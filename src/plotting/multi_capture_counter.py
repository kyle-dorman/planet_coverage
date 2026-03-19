import logging
import warnings
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator, PercentFormatter
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
FIG_DIR = BASE.parent / "figs" / "multi_capture"
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Example path patterns
f_pattern = "dove/coastal_results/*/*/*/coastal_points.parquet"
all_files_pattern = str(BASE / f_pattern)

# Combined list used later when we search individual files
all_parquets = list(BASE.glob(f_pattern))

if not all_parquets:
    logger.error("No parquet files found matching pattern %s", all_files_pattern)
    raise FileNotFoundError("No parquet files found")
logger.info("Found %d parquet files", len(all_parquets))


query_df, grids_df, hex_grid = load_grids(SHORELINES)
MIN_DIST = 20.0
valid = ~grids_df.is_land & grids_df.dist_km.notna() & (grids_df.dist_km < MIN_DIST)
grids_df = grids_df[valid].copy()

# --- Connect to DuckDB ---
con = duckdb.connect()

# Register a view for all files
con.execute(f"""
    CREATE OR REPLACE VIEW samples_all AS
    SELECT * FROM read_parquet('{all_files_pattern}');
""")
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
    minute_edges = [1e-3] + [round_up(n) for n in bins]
    bin_right = minute_edges[1:]  # right edge of each bar
    day_edges = [m / 1440.0 for m in bin_right]  # minutes → days
    valid = True

    grid_ids = grids_df.index.to_list()
    grid_tbl = pd.DataFrame({"grid_id": grid_ids})
    con.register("grid_ids_tbl", grid_tbl)

    hist_rows = []  # tidy (long) rows for CSV export

    start_year = 2016
    end_year = 2025
    num_years = end_year - start_year
    for year in tqdm(range(start_year, end_year), total=num_years):
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

        # Last bin is inf, ignore
        int_bins = np.array(bins[:-1] * 1440.0, dtype=np.int32).tolist()
        counts = counts[:-1]
        if not np.allclose(bins[:-1], day_edges):
            print(year)
            print(int_bins, bin_right)

        # Tidy/long rows for CSV: one row per (year, bin)
        # Keep the existing "+ 1" year convention from the previous code.
        year_out = year + 1

        # Lower/upper edges in minutes for each bin (upper edges are `int_bins` here)
        bin_lowers = [0] + int_bins[:-1]

        for lo, hi, c in zip(bin_lowers, int_bins, counts):
            hist_rows.append(
                {
                    "year": year_out,
                    "bin_lower_edge_min": int(lo),
                    "bin_upper_edge_min": int(hi),
                    "count": int(c),
                }
            )

    logger.info("Query finished")
    hist_df = pd.DataFrame(hist_rows).sort_values(["year", "bin_upper_edge_min"])
    hist_df.to_csv(FIG_DIR / "hist_data.csv", index=False)

    # Pivot to (bin x year) for plotting: each x-bin is stacked by year
    pivot = hist_df.pivot(index="bin_upper_edge_min", columns="year", values="count").fillna(0).sort_index()

    bin_uppers = pivot.index.to_numpy(dtype=float)  # minute upper edges
    bin_lowers = np.r_[0.0, bin_uppers[:-1]]
    widths = bin_uppers - bin_lowers
    centers = bin_uppers - widths / 2.0

    years = list(pivot.columns)

    # ── Stacked histogram by bin (stacked by year) ─────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)

    bottom = np.zeros(len(bin_uppers), dtype=float)

    # Use a colormap for year colors (no hard-coded palette)
    cmap_years = plt.cm.get_cmap("viridis", num_years)
    year_colors = [cmap_years(year - start_year - 1) for year in years]

    for yr, color in zip(years, year_colors):
        y = pivot[yr].to_numpy(dtype=float)
        ax.bar(
            centers,
            y,
            width=widths,
            bottom=bottom,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.9,
            label=str(int(yr)),
            zorder=3,
        )
        bottom += y

    # Axes styling: log x-axis like before
    ax.set_xscale("log")
    ax.set_xticks(minute_edges[1:])
    ax.set_xticklabels([f"{b:g}" for b in minute_edges[1:]])
    ax.xaxis.set_minor_locator(NullLocator())

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)

    ax.set_ylabel("Count", fontsize=12)
    ax.set_xlabel("Time Between Samples (minutes)", fontsize=12)
    ax.set_title("Time Between Same-Day Samples (counts; stacked by year)", fontsize=14)

    ax.legend(
        title="Year",
        frameon=False,
        ncol=3,
        fontsize=8,
        title_fontsize=9,
        loc="upper right",
    )

    plt.savefig(FIG_DIR / "histogram_time_between_samples_stacked_by_year.png")
    plt.close(fig)

    # ── Per-year cumulative distributions ──────────────────────────────────
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)

    # x-axis in minutes (upper edge of each bin)
    x_minutes = bin_uppers.astype(float)

    max_total = max(pivot[yr].to_numpy(dtype=float).sum() for yr in years)
    for yr in years:
        row = pivot[yr].to_numpy(dtype=float)
        total = float(row.sum())
        if total <= 0:
            continue
        cdf = np.cumsum(row) / max_total
        color = cmap_years(yr - start_year - 1)
        ax2.plot(x_minutes, cdf, linewidth=1.8, linestyle="-", zorder=4, label=str(int(yr)), color=color)

    ax2.set_xscale("log")
    ax2.set_xticks(minute_edges[1:])
    ax2.set_xticklabels([f"{b:g}" for b in minute_edges[1:]])
    ax2.xaxis.set_minor_locator(NullLocator())

    ax2.set_ylim(0.0, 1.0)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.grid(axis="both", linestyle="--", alpha=0.7, zorder=0)

    ax2.axhline(0.50, linestyle="--", linewidth=0.8, color="black", alpha=0.7)
    ax2.axhline(0.95, linestyle=":", linewidth=0.8, color="black", alpha=0.7)

    ax2.set_ylabel("Cumulative % of samples", fontsize=12)
    ax2.set_xlabel("Time Between Samples (minutes)", fontsize=12)
    ax2.set_title("CDF by Year: Time Between Same-Day Samples", fontsize=14)

    ax2.legend(
        title="Year",
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="lower right",
    )

    plt.savefig(FIG_DIR / "cumsum_time_between_samples_cdf_by_year.png")
    plt.close(fig2)


def plot_geo():
    year = 2024
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

    logger.info("Saving grid results to ShapeFile")
    (FIG_DIR / "per_grid_multi_capture_days").mkdir(exist_ok=True)
    gdf.to_file(FIG_DIR / "per_grid_multi_capture_days" / "data.shp")

    plot_gdf_column(
        gdf=gdf,
        column="max_multi_capture_days",
        title="Count Multi-Capture Days",
        save_path=FIG_DIR / "max_multi_capture_days.png",
        use_cbar_label=False,
        vmax=70,
    )
    logger.info("Done plotting geo")


# plot_geo()
plot_histogram()

logger.info("Done")
