#!/usr/bin/env python

# # DuckDB + Parquet Data Exploration Template

import os
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd

from src.util import create_config

# # --- Configuration ---


BASE = Path("/Users/kyledorman/data/planet_coverage/points_30km/")  # <-- update this
config_file = BASE / "dove" / "config.yaml"  # <-- update this
config = create_config(config_file)

# Example path patterns
f_pattern = "*/results/*/*/*/*/ocean.parquet"
all_files_pattern = str(BASE / f_pattern)

# Combined list used later when we search individual files
all_parquets = list(BASE.glob(f_pattern))

GRID_ID = 31565
hex_id = f"{GRID_ID:06x}"  # unique 6‑digit hex, e.g. '0f1a2b'
d1, d2, d3 = hex_id[:2], hex_id[2:4], hex_id[4:6]
GRID_PATH = BASE / "dove" / "results" / "2024" / d1 / d2 / d3
FILE = GRID_PATH / "ocean.parquet"


# Create the base map centered on the calculated location
ca_ocean = gpd.read_file(BASE / "ca_ocean.geojson")
assert ca_ocean.crs is not None
grids_df = gpd.read_file(BASE / "ocean_grids.gpkg").to_crs(ca_ocean.crs)

# --- Connect to DuckDB ---
con = duckdb.connect()


# Register a view for all files
con.execute(
    f"""
    CREATE OR REPLACE VIEW samples_all AS
    SELECT * FROM read_parquet('{all_files_pattern}');
"""
)


if not os.path.exists("extracted/dove_skysay_grid_counts.csv"):
    df_dove = con.execute(
        """
        SELECT cell_id, date_trunc('year', acquired) AS year, COUNT(*) AS dove_count
        FROM samples_all
        WHERE item_type = 'PSScene'
        GROUP BY cell_id, year
        ORDER BY cell_id, year
    """
    ).fetchdf()
    df_sky = con.execute(
        """
        SELECT cell_id, date_trunc('year', acquired) AS year, COUNT(*) AS sky_count
        FROM samples_all
        WHERE item_type = 'SkySatCollect'
        GROUP BY cell_id, year
        ORDER BY cell_id, year
    """
    ).fetchdf()

    merged = df_dove.set_index(["cell_id", "year"]).join(df_sky.set_index(["cell_id", "year"]), how="inner")

    merged.to_csv("extracted/dove_skysay_grid_counts.csv")


def create_bool_df(column_name, bool_logic_str, merge_df, nafill=0.0):
    df_pct = con.execute(
        f"""
        SELECT cell_id,
               SUM({bool_logic_str})::DOUBLE  / COUNT(*) AS frac_{column_name}
        FROM samples_all
        WHERE item_type = 'PSScene'
        GROUP BY cell_id
    """
    ).fetchdf()

    geo_pct = (
        merge_df.set_index("cell_id")
        .join(df_pct.set_index("cell_id"), how="left")
        .fillna({f"frac_{column_name}": nafill})
    )

    return geo_pct


# --- Load Geo Points and Join ---

# Sample count per grid cell
df_counts = con.execute(
    """
    SELECT cell_id, COUNT(*) as sample_count
    FROM samples_all
    WHERE item_type = 'SkySatCollect'
    GROUP BY cell_id
"""
).fetchdf()

if not os.path.exists("extracted/skysat_sample_count.gpkg"):
    geo_plot = (
        grids_df.set_index("cell_id").join(df_counts.set_index("cell_id"), how="left").fillna({"sample_count": 0})
    )
    geo_plot.to_file("extracted/skysat_sample_count.gpkg", driver="GPKG")


# --- Load Geo Points and Join ---

# Sample count per grid cell
if not os.path.exists("extracted/dove_sample_count.gpkg"):
    df_counts = con.execute(
        """
        SELECT cell_id, COUNT(*) as sample_count
        FROM samples_all
        WHERE item_type = 'PSScene'
        GROUP BY cell_id
    """
    ).fetchdf()

    geo_plot = (
        grids_df.set_index("cell_id").join(df_counts.set_index("cell_id"), how="left").fillna({"sample_count": 0})
    )

    geo_plot.to_file("extracted/dove_sample_count.gpkg", driver="GPKG")


def compute_histogram(column: str, nbins: int = 30) -> pd.DataFrame:
    """
    Runs DuckDB's histogram() table function on `column` in samples_all (filtered to PSScene)
    and returns a DataFrame with columns: bin_upper, frequency.
    """
    sql = f"""
        WITH bounds AS (
          SELECT
            MIN({column}) AS mn,
            MAX({column}) AS mx
          FROM samples_all
          WHERE item_type = 'PSScene'
        )
        SELECT
          -- histogram() returns a MAP<upper_boundary, count>
          histogram(
            {column},
            equi_width_bins(bounds.mn::DOUBLE, bounds.mx::DOUBLE, {nbins}::BIGINT, True)
          ) AS hist_map
        FROM samples_all
        CROSS JOIN bounds
        WHERE item_type = 'PSScene';
    """
    hist_map = con.execute(sql).fetchdf().iloc[0]["hist_map"]

    # Unpack into a two-column DataFrame
    df = pd.DataFrame({"bin_upper": list(hist_map.keys()), "count": list(hist_map.values())})
    df = df.sort_values("bin_upper").reset_index(drop=True)
    uppers = df["bin_upper"].tolist()
    bin_size = uppers[1] - uppers[0]
    # Compute lower edge from previous upper
    lowest = uppers[0] - bin_size
    lowers = [lowest] + uppers[:-1]
    df["bin_lower"] = pd.Series(lowers)
    df["centers"] = (df["bin_lower"] + df["bin_upper"]) / 2
    df["widths"] = df["bin_upper"] - df["bin_lower"]
    return df


# 2. Plotting all four angle columns
for col in ["satellite_azimuth", "sun_azimuth", "sun_elevation", "view_angle"]:
    if not os.path.exists(f"extracted/hist_{col}.csv"):
        df_hist = compute_histogram(col, nbins=30)
        df_hist.to_csv(f"extracted/hist_{col}.csv")


# --- Fraction of preview per Grid Point ---
if not os.path.exists("extracted/publishing_stage_pct.gpkg"):
    create_bool_df("publishing_stage", "publishing_stage = 'preview'", grids_df, nafill=0.0).to_file(
        "extracted/publishing_stage_pct.gpkg", driver="GPKG"
    )


# --- Fraction analysis ready data per Grid Point ---
if not os.path.exists("extracted/pct_analysis_ready.gpkg"):
    create_bool_df(
        "has_sr_asset",
        "has_sr_asset",
        grids_df,
        nafill=0.0,
    ).to_file("extracted/pct_analysis_ready.gpkg", driver="GPKG")


# --- Fraction analysis ready data per Grid Point ---
if not os.path.exists("extracted/pct_ground_control.gpkg"):
    create_bool_df(
        "ground_control",
        "ground_control",
        grids_df,
        nafill=0.0,
    ).to_file("extracted/pct_ground_control.gpkg", driver="GPKG")


if os.path.exists("extracted/pct_preview_with_ground_control.gpkg"):
    df_pct = con.execute(
        """
        SELECT cell_id,
            SUM(ground_control)::DOUBLE  / COUNT(*) AS frac_preview_gc
        FROM samples_all
        WHERE item_type = 'PSScene' AND publishing_stage = 'preview'
        GROUP BY cell_id
    """
    ).fetchdf()

    geo_pct = (
        grids_df.set_index("cell_id").join(df_pct.set_index("cell_id"), how="left").fillna({"frac_preview_gc": 0.0})
    )

    geo_pct.to_file("extracted/pct_preview_with_ground_control.gpkg", driver="GPKG")
