import logging
import warnings
from pathlib import Path

import duckdb
import geopandas as gpd

from src.plotting.util import (
    load_grids,
    make_high_frequency_query,
    make_max_daily_captures_query,
    make_multiple_captures_query,
    plot_gdf_column,
)

warnings.filterwarnings("ignore")  # hide every warning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


BASE = Path("/Users/kyledorman/data/planet_coverage/points_30km/")  # <-- update this
FIG_DIR = BASE.parent / "figs" / BASE.name / "multi_capture"
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Example path patterns
f_pattern = "*/coastal_results/*/*/*/coastal_points.parquet"
all_files_pattern = str(BASE / f_pattern)

# Combined list used later when we search individual files
all_parquets = list(BASE.glob(f_pattern))

if not all_parquets:
    logger.error("No parquet files found matching pattern %s", all_files_pattern)
    raise FileNotFoundError("No parquet files found")
logger.info("Found %d parquet files", len(all_parquets))


query_df, grids_df, hex_grid = load_grids(BASE)

# --- Connect to DuckDB ---
con = duckdb.connect()

# Register a view for all files
con.execute(
    f"""
    CREATE OR REPLACE VIEW samples_all AS
    SELECT * FROM read_parquet('{all_files_pattern}');
"""
)
logger.info("Registered DuckDB view 'samples_all'")

year = 2023
disp_year = year + 1

for valid in [True, False]:
    valid_str = "valid" if valid else "all"

    query = make_multiple_captures_query(year, valid_only=valid)

    df = con.execute(query).fetchdf().set_index("grid_id")
    hex_df = grids_df.join(df, how="left").fillna({"multi_capture_days": 0})

    logger.info("Query finished")

    logger.info(f"Plotting {valid_str}")
    agg = hex_df.groupby("hex_id").agg(
        max_multi_capture_days=("multi_capture_days", "max"),
        median_multi_capture_days=("multi_capture_days", "median"),
    )
    agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
    gdf = gpd.GeoDataFrame(agg, geometry="geometry")

    plot_gdf_column(
        gdf,
        "max_multi_capture_days",
        title=f"Max Multi Capture Days {disp_year} ({valid_str.capitalize()})",
        show_coastlines=True,
        save_path=FIG_DIR / f"max_multi_capture_days_{valid_str}.png",
        show=False,
    )
    plot_gdf_column(
        gdf,
        "median_multi_capture_days",
        title=f"Median Multi Capture Days {disp_year} ({valid_str.capitalize()})",
        show_coastlines=True,
        save_path=FIG_DIR / f"median_multi_capture_days_{valid_str}.png",
        show=False,
    )


for freqency in [2, 10, 60]:
    for valid in [True, False]:
        valid_str = "valid" if valid else "all"

        query = make_high_frequency_query(year, freqency, valid_only=valid)

        df = con.execute(query).fetchdf().set_index("grid_id")
        hex_df = grids_df.join(df, how="left").fillna({"high_freq_count": 0})

        logger.info("Query finished")

        logger.info(f"Plotting {valid_str}")
        agg = hex_df.groupby("hex_id").agg(
            max_high_freq_count=("high_freq_count", "max"),
            median_high_freq_count=("high_freq_count", "median"),
        )
        agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
        gdf = gpd.GeoDataFrame(agg, geometry="geometry")

        plot_gdf_column(
            gdf,
            "max_high_freq_count",
            title=f"Max Captures within {freqency} minutes {disp_year} ({valid_str.capitalize()})",
            show_coastlines=True,
            save_path=FIG_DIR / f"max_high_freq_count_{freqency}_{valid_str}.png",
            show=False,
        )
        plot_gdf_column(
            gdf,
            "median_high_freq_count",
            title=f"Median Captures within {freqency} minutes {disp_year} ({valid_str.capitalize()})",
            show_coastlines=True,
            save_path=FIG_DIR / f"median_high_freq_count_{freqency}_{valid_str}.png",
            show=False,
        )


for valid in [True, False]:
    valid_str = "valid" if valid else "all"

    query = make_max_daily_captures_query(year, valid_only=valid)

    df = con.execute(query).fetchdf().set_index("grid_id")
    hex_df = grids_df.join(df, how="left").fillna({"max_daily_captures": 0})

    logger.info("Query finished")

    logger.info(f"Plotting {valid_str}")
    agg = hex_df.groupby("hex_id").agg(
        max_max_daily_captures=("max_daily_captures", "max"),
        median_max_daily_captures=("max_daily_captures", "median"),
    )
    agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
    gdf = gpd.GeoDataFrame(agg, geometry="geometry")

    plot_gdf_column(
        gdf,
        "max_max_daily_captures",
        title=f"Max Max Captures in a Day {disp_year} ({valid_str.capitalize()})",
        show_coastlines=True,
        save_path=FIG_DIR / f"max_max_daily_captures_{valid_str}.png",
        show=False,
    )
    plot_gdf_column(
        gdf,
        "median_max_daily_captures",
        title=f"Median Max Captures in a Day {disp_year} ({valid_str.capitalize()})",
        show_coastlines=True,
        save_path=FIG_DIR / f"median_max_daily_captures_{valid_str}.png",
        show=False,
    )
