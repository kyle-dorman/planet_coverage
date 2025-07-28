import logging
import warnings
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
from tqdm import tqdm

from src.plotting.util import (
    load_grids,
    make_daily_time_between_hist_query,
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


BASE = Path("/Users/kyledorman/data/planet_coverage/points_30km/")
SHORELINES = BASE.parent / "shorelines"
FIG_DIR = BASE.parent / "figs_v2" / "multi_capture"
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


query_df, grids_df, hex_grid = load_grids(SHORELINES)
MIN_DIST = 20.0
lats = grids_df.centroid.y
valid = ~grids_df.is_land & ~grids_df.dist_km.isna() & (grids_df.dist_km < MIN_DIST) & (lats > -81.5) & (lats < 81.5)
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
# con.install_extension("sql_functions")
# con.load_extension("sql_functions")
logger.info("Registered DuckDB view 'samples_all'")


# Histogram edges in minutes (log axis cannot include 0)
def round_up(n):
    if n < 5:
        return n
    return int(10 * round(n / 10, 0))


def plot_histogram():
    # Pleasant default style
    plt.style.use("seaborn-v0_8-darkgrid")

    bins = np.floor(np.logspace(np.log10(1.0), np.log10(12.0 * 60), 10)).astype(np.int32)
    minute_edges = [0.1] + [round_up(n) for n in bins]
    day_edges = [m / 1440.0 for m in minute_edges]  # minutes → days
    valid = True

    grid_ids = grids_df.index.to_list()
    grid_tbl = pd.DataFrame({"grid_id": grid_ids})
    con.register("grid_ids_tbl", grid_tbl)

    all_year_counts = np.zeros(len(minute_edges) - 1)

    for year in tqdm(range(2016, 2025), total=2025 - 2016):
        query = make_daily_time_between_hist_query(
            year,
            bins=day_edges,
            valid_only=valid,
        )

        hist_dict = con.execute(query).fetchall()[0][0]

        counts = np.array(list(hist_dict.values()))
        bins = np.array(list(hist_dict.keys()))
        # Ensure bins are in correct order (they should be!)
        order = np.argsort(bins)
        bins = bins[order]
        counts = counts[order]

        all_year_counts += counts[1:-1]  # skip the <0.1‑min bin

    logger.info("Query finished")

    # create a grid
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(6, 4),
        constrained_layout=True,
    )

    # ── build centres & widths (skip the <1‑minute bin) ────────────────────
    widths = np.diff(minute_edges)
    widths[0] = widths[1]  # shrink the first visible bar
    centers = np.array([right - w / 2 for right, w in zip(minute_edges[1:], widths)])

    # ── plot, with edge & alpha for depth ───────────────────────────────────
    ax.bar(
        centers,
        all_year_counts,
        width=widths,
        color="#4c78a8",
        edgecolor="black",
        alpha=0.85,
        zorder=3,
    )

    # ── axes styling ───────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_xticks(minute_edges[1:])
    ax.set_xticklabels([f"{b:g}" for b in minute_edges[1:]])
    ax.xaxis.set_minor_locator(NullLocator())

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)

    fig.supylabel("Frequency", fontsize=12)
    fig.supxlabel("Time Between Samples (minutes)", fontsize=12)
    fig.suptitle("Time Between Same-Day Samples", fontsize=14)

    plt.savefig(FIG_DIR / "histogram_time_between_samples.png")
    plt.close(fig)


def plot_geo():
    year = 2023
    query = make_multiple_captures_query(year, valid_only=True)

    df = con.execute(query).fetchdf().set_index("grid_id")
    hex_df = grids_df.join(df, how="left")

    logger.info("Query finished")

    logger.info("Plotting")
    agg = hex_df.groupby("hex_id").agg(
        max_multi_capture_days=("multi_capture_days", "max"),
    )
    agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
    gdf = gpd.GeoDataFrame(agg, geometry="geometry")

    plot_gdf_column(
        gdf=gdf,
        column="max_multi_capture_days",
        title="Count Multi-Capture Days",
        save_path=FIG_DIR / "max_multi_capture_days.png",
        use_cbar_label=False,
        vmax=70,
    )


plot_geo()
# plot_histogram()

logger.info("Done")
