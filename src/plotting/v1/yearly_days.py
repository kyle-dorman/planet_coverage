import logging
import warnings
from datetime import datetime
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import NullLocator
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
FIG_DIR = BASE.parent / "figs" / BASE.name / "days_with_sample"
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
MIN_DIST = 20.0
valid = ~grids_df.is_land & (grids_df.dist_km.isna() | (grids_df.dist_km < MIN_DIST))
grids_df = grids_df[valid].copy()

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


all_grids_query = """
    SELECT
        grid_id,
        COUNT(*) AS sample_count,
    FROM samples_all
    WHERE
        item_type = 'PSScene'
        AND coverage_pct > 0.5
    GROUP BY grid_id
"""

all_grid_ids = con.execute(all_grids_query).fetchdf().grid_id.tolist()


valid_grids_query = """
    SELECT
        grid_id,
        COUNT(*) AS sample_count,
    FROM samples_all
    WHERE
        item_type = 'PSScene'
        AND coverage_pct > 0.5
        AND clear_percent > 75.0
        AND publishing_stage = 'finalized'
        AND quality_category = 'standard'
        AND has_sr_asset
        AND ground_control
    GROUP BY grid_id
"""

valid_grid_ids = con.execute(valid_grids_query).fetchdf().grid_id.tolist()


def make_year_query(year: int, valid_only: bool) -> str:
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
        SELECT
            grid_id,
            COUNT(DISTINCT DATE_TRUNC('day', acquired)) AS days_with_sample
        FROM samples_all
        WHERE acquired >= TIMESTAMP '{start_date}'
        AND acquired <  TIMESTAMP '{end_date}'
        AND item_type   = 'PSScene'
        AND coverage_pct > 0.5
        {valid_filter}
        GROUP BY grid_id
        ORDER BY grid_id;
    """


processed = []
nbins = 10
bin_edges = np.ceil(np.logspace(np.log10(1.0), np.log10(366.0), nbins + 1)).astype(np.int32)
bin_widths = np.diff(bin_edges)
bin_left = bin_edges[:-1]  # left edge of each bar

for year in tqdm(range(2015, 2025), total=10):
    for valid in [True, False]:
        query = make_year_query(year, valid)

        valid_str = "valid" if valid else "all"

        df = con.execute(query).fetchdf().set_index("grid_id")
        hex_df = grids_df[["hex_id"]].join(df, how="left").fillna({"days_with_sample": 0})

        if valid:
            hex_df = hex_df.loc[hex_df.index.intersection(valid_grid_ids)]
        else:
            hex_df = hex_df.loc[hex_df.index.intersection(all_grid_ids)]

        hist, _ = np.histogram(hex_df.days_with_sample, bin_edges)
        row = {f"count_{i}": hist[i] for i in range(nbins)}
        row["year"] = year + 1
        row["valid"] = valid
        processed.append(row)

        agg = hex_df.groupby("hex_id").agg(
            median_days_with_sample=("days_with_sample", "median"), max_days_with_sample=("days_with_sample", "max")
        )
        agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
        gdf = gpd.GeoDataFrame(agg, geometry="geometry")

        plot_gdf_column(
            gdf,
            "median_days_with_sample",
            title=f"Days with at least one sample (Year: {year + 1} Agg: Median)",
            show_land_ocean=True,
            save_path=FIG_DIR / f"median_days_with_sample_{valid_str}_{year + 1}.png",
            vmax=365,
            vmin=0,
            show=False,
        )
        plot_gdf_column(
            gdf,
            "max_days_with_sample",
            title=f"Days with at least one sample (Year: {year + 1} Agg: Max)",
            show_land_ocean=True,
            save_path=FIG_DIR / f"max_days_with_sample_{valid_str}_{year + 1}.png",
            vmax=365,
            vmin=1,
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
        ax.set_xscale("log")
        ax.set_xticks(bin_edges)
        ax.set_xticklabels(
            [f"{b:g}" for b in bin_edges],
            rotation=45,
            ha="right",
        )
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_title(f"{row['year']}")

        # thin tick labels for space
        ax.tick_params(axis="both", labelsize=8)

    # add a single y-label on the figure border
    fig.supylabel("Frequency (grid cells)", fontsize=12)
    fig.supxlabel("Days with ≥1 sample in bin", fontsize=12)

    # overall figure title
    fig.suptitle("Histogram of Sampling Days per Grid", fontsize=14, y=1.04)

    plt.savefig(FIG_DIR / f"histogram_sample_count_by_year_{valid_str}.png")

    plt.close(fig)

    # # Cumulative sum plot
    # fig, axes = plt.subplots(
    #     2,
    #     5,
    #     figsize=(15, 6),
    #     sharex=True,
    #     constrained_layout=True,
    # )
    # axes = axes.ravel()  # flatten 2-D array → 1-D iterator

    # for ax, (_, row) in zip(axes, group.iterrows()):
    #     counts = [row[f"count_{i}"] for i in range(nbins)]
    #     cumsum = np.cumsum(counts)
    #     total = cumsum[-1] if cumsum[-1] else 1.0  # prevent divide-by-zero
    #     cumsum_pct = cumsum / total * 100.0  # → 0-100 %

    #     ax.plot(bin_left, cumsum_pct, marker="o", linestyle="-")
    #     ax.set_ylim(0, 100)
    #     ax.set_xscale("log")
    #     ax.set_xticks(bin_edges)
    #     ax.set_xticklabels([f"{edge:.0f}" for edge in bin_edges], rotation=45, ha="right")
    #     ax.set_title(f"{valid_str.capitalize()} {row['year']}")
    #     ax.tick_params(axis="both", labelsize=8)

    # fig.supylabel("Cumulative Grid Cell %", fontsize=12)
    # fig.supxlabel("Days with ≥1 sample in bin", fontsize=12)
    # fig.suptitle(
    #     f"Cumulative Distribution of Sampling Days per Grid – {valid_str.capitalize()}",
    #     fontsize=14,
    #     y=1.04,
    # )

    # plt.savefig(FIG_DIR / f"cumsum_sample_count_by_year_{valid_str}.png")

    # plt.close(fig)
