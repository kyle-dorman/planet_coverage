import logging
import warnings
from datetime import datetime
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.plotting.util import load_grids, plot_gdf_column

warnings.filterwarnings("ignore")  # hide every warning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


BASE = Path("/Users/kyledorman/data/planet_coverage/points_30km/")  # <-- update this
FIG_DIR = BASE.parent / "figs" / BASE.name / "time_between"
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


def make_time_between_query(year: int, valid_only: bool) -> str:
    """
    Build the fiscal-year query for a single 12-month window.
    """

    # ------------------------------------------------------------------
    # Compute end date = start + 1 year  (no extra deps needed)
    # ------------------------------------------------------------------
    start_dt = datetime(year, 12, 1).date()
    end_dt = start_dt.replace(year=start_dt.year + 1)
    end_date = end_dt.isoformat()
    start_date = start_dt.isoformat()

    valid_filter = (
        """
        AND publishing_stage = 'finalized'
        AND quality_category = 'standard'
        AND clear_percent    > 75.0
        AND has_sr_asset
        AND ground_control
    """
        if valid_only
        else ""
    )

    return f"""
    WITH ordered AS (                       -- 1️⃣  rows in time order
        SELECT
            grid_id,
            acquired,
            EXTRACT(epoch FROM acquired) AS ts          -- seconds-since-epoch
        FROM samples_all
        WHERE
            item_type    = 'PSScene'
        AND coverage_pct > 0.5
        AND acquired BETWEEN TIMESTAMP '{start_date}' AND TIMESTAMP '{end_date}'
        {valid_filter}
    ),

    deltas AS (                           -- 2️⃣  Δt between consecutive samples
        SELECT
            grid_id,
            (ts - LAG(ts) OVER (PARTITION BY grid_id
                                ORDER BY ts)) / 86400.0    AS days_between
        FROM ordered
    ),

    filtered AS (                         -- 3️⃣  keep only gaps ≥ 12 h
        SELECT grid_id, days_between
        FROM deltas
        WHERE days_between >= 0.5
    )

    -- 4️⃣  p95 per grid
    SELECT
        grid_id,
        quantile_cont(days_between, 0.95) AS p95_days_between
    FROM filtered
    WHERE days_between IS NOT NULL
    GROUP BY grid_id
    ORDER BY grid_id;
    """


processed = []
nbins = 20
bin_edges = np.linspace(0.0, 365.0, nbins + 1)
bin_widths = np.diff(bin_edges)
bin_left = bin_edges[:-1]  # left edge of each bar

for year in tqdm(range(2015, 2025), total=10):
    for valid in [True, False]:
        query = make_time_between_query(year, valid)

        valid_str = "valid" if valid else "all"

        df = con.execute(query).fetchdf().set_index("grid_id")
        hex_df = grids_df[["hex_id"]].join(df, how="left").fillna({"p95_days_between": 365})

        hist, _ = np.histogram(hex_df.p95_days_between, bin_edges)
        row = {f"count_{i}": hist[i] for i in range(nbins)}
        row["year"] = year + 1
        row["valid"] = valid
        processed.append(row)

        agg = hex_df.groupby("hex_id").agg(
            median_p95_days_between=("p95_days_between", "median"),
            max_p95_days_between=("p95_days_between", "max"),
            min_p95_days_between=("p95_days_between", "min"),
        )
        agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
        gdf = gpd.GeoDataFrame(agg, geometry="geometry")

        plot_gdf_column(
            gdf,
            "median_p95_days_between",
            title=f"Median p95 Time Between Samples {year + 1} ({valid_str.capitalize()})",
            show_coastlines=True,
            save_path=FIG_DIR / f"median_p95_time_between_samples_{valid_str}_{year + 1}.png",
            vmax=365,
            vmin=1,
            scale="log",
            show=False,
        )
        plot_gdf_column(
            gdf,
            "max_p95_days_between",
            title=f"Max p95 Time Between Samples {year + 1} ({valid_str.capitalize()})",
            show_coastlines=True,
            save_path=FIG_DIR / f"max_p95_time_between_samples_{valid_str}_{year + 1}.png",
            vmax=365,
            vmin=1,
            scale="log",
            show=False,
        )
        plot_gdf_column(
            gdf,
            "min_p95_days_between",
            title=f"Min p95 Time Between Samples {year + 1} ({valid_str.capitalize()})",
            show_coastlines=True,
            save_path=FIG_DIR / f"min_p95_time_between_samples_{valid_str}_{year + 1}.png",
            vmax=365,
            vmin=1,
            scale="log",
            show=False,
        )

df = pd.DataFrame(processed)
for valid_flag, group in df.groupby("valid"):
    valid_str = "valid" if valid_flag else "all"
    # sort rows by fiscal year so the sub-plots are in order
    group = group.sort_values("year")

    # create a 2 × 5 grid of axes (ten panels)
    fig, axes = plt.subplots(
        2,
        5,
        figsize=(15, 6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = axes.ravel()  # flatten 2-D array → 1-D iterator

    # loop over the ten rows in this validity group
    for ax, (_, row) in zip(axes, group.iterrows()):
        counts = [row[f"count_{i}"] for i in range(nbins)]

        ax.bar(
            bin_left,  # left edge of each bar
            counts,
            width=bin_widths,
            align="edge",
        )
        ax.set_title(f"{valid_str.capitalize()} {row['year']}")

        # thin tick labels for space
        ax.tick_params(axis="both", labelsize=8)

    # add a single y-label on the figure border
    fig.supylabel("Frequency (grid cells)", fontsize=12)
    fig.supxlabel("p95 Days Between Samples", fontsize=12)

    # overall figure title
    fig.suptitle("Histogram of p95 Days Between Samples per Grid – " f"{valid_str.capitalize()}", fontsize=14, y=1.04)

    plt.savefig(FIG_DIR / f"histogram_p95_time_between_samples_by_year_{valid_str}.png")

    # plt.show()
