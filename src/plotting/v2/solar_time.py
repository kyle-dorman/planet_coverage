import logging
import warnings
from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.util import load_grids, plot_gdf_column

warnings.filterwarnings("ignore")  # hide every warning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

logger.info("Computing solar stats and plots")

BASE = Path("/Users/kyledorman/data/planet_coverage/points_30km/")
SHORELINES = BASE.parent / "shorelines"
FIG_DIR = BASE.parent / "figs_v2" / "solar_time"
FIG_DIR.mkdir(exist_ok=True, parents=True)

query_df, grids_df, hex_grid = load_grids(SHORELINES)
MIN_DIST = 20.0
LAT = 81.5
lats = grids_df.centroid.y
valid = ~grids_df.is_land & ~grids_df.dist_km.isna() & (grids_df.dist_km < MIN_DIST) & (lats > -LAT) & (lats < LAT)
grids_df = grids_df[valid].copy()
logger.info("Loaded grid dataframes")

# --- Connect to DuckDB ---
con = duckdb.connect()


def fiscal_year_expr(col: str = "acquired") -> str:
    """Return a DuckDB SQL expression for a December-based fiscal year label.
    FY is labeled by the *ending* year: Dec YYYY → Nov (YYYY+1) maps to FY (YYYY+1).
    """
    return (
        f"CASE WHEN EXTRACT(month FROM {col}) >= 12 THEN EXTRACT(year FROM {col}) + 1 ELSE EXTRACT(year FROM {col}) END"
    )


def solar_time_offset_by_fy_plot():
    """Plot mean and IQR (25-75%) of solar_time_offset_seconds by fiscal year (Dec→Dec),
    split by instrument. Produces a grouped bar chart with asymmetric IQR error bars.
    """
    logger.info("Creating solar offset plots")
    # Search both Dove and SkySat solar outputs (coastal_results_solar)
    f_pat = "*/coastal_results_solar/*/*/*/coastal_points.parquet"
    pat = str(BASE / f_pat)

    all_files = list(BASE.glob(f_pat))
    if not all_files:
        logger.error("No solar parquet files found in coastal_results_solar paths.")
        raise FileNotFoundError("No coastal_results_solar parquet files found")

    con.execute(
        f"""
        CREATE OR REPLACE VIEW samples_all AS
        SELECT * FROM read_parquet('{pat}')
        """
    )
    logger.info("Registered DuckDB view 'samples_all' for solar_time_offset plot")

    grid_ids = grids_df.index.to_list()
    grid_tbl = pd.DataFrame({"grid_id": grid_ids})
    con.register("grid_ids_tbl", grid_tbl)  # exposes it as a DuckDB view

    fy = fiscal_year_expr("s.acquired")

    # Aggregate mean + quantiles by FY and instrument
    query = f"""
        SELECT
            s.instrument,
            CAST({fy} AS INTEGER)                                       AS fy,
            avg(s.solar_time_offset_seconds) / 3600.0                   AS mean_hours,
            quantile_cont(s.solar_time_offset_seconds, 0.25) / 3600.0   AS q25_hours,
            median(s.solar_time_offset_seconds) / 3600.0                AS median_hours,
            quantile_cont(s.solar_time_offset_seconds, 0.75) / 3600.0   AS q75_hours,
            count(*)                                                    AS sample_count
        FROM samples_all as s
        JOIN grid_ids_tbl USING (grid_id)
        WHERE s.item_type = 'PSScene'
            AND s.coverage_pct > 0.5
            AND s.acquired >= '2015-12-01'
            AND s.acquired < '2024-12-01'
            AND s.publishing_stage = 'finalized'
            AND s.quality_category = 'standard'
            AND s.clear_percent    > 75.0
            AND s.has_sr_asset
            AND s.ground_control
        GROUP BY instrument, fy
        ORDER BY instrument, fy
    """
    df = con.execute(query).fetchdf()
    if df.empty:
        logger.warning("No rows returned for solar_time_offset_seconds aggregation")
        return

    # Ensure instruments are stable order
    inst_order = sorted(df["instrument"].unique())
    years = list(range(2016, 2025))  # FY 2016 (Dec 2015→Nov 2016) through FY 2024

    # Pivot for plotting grouped bars (mean as height, IQR as error bars)
    # We'll create one bar per instrument per FY with asymmetric errors: [mean - q25, q75 - mean]
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)

    # order years on x-axis
    x_years = np.array(years)

    # Assign fixed colors (requested): blue, green, red; fall back to cycle if more
    base_colors = ["tab:blue", "tab:green", "tab:red"]

    for idx, inst in enumerate(inst_order):
        color = base_colors[idx % len(base_colors)]
        sub = df[df["instrument"] == inst].set_index("fy").reindex(years)

        med = sub["median_hours"].to_numpy()
        q25 = sub["q25_hours"].to_numpy()
        q75 = sub["q75_hours"].to_numpy()

        # Shaded IQR ribbon
        ax.fill_between(x_years, q25, q75, alpha=0.25, label=f"{inst} IQR", color=color, linewidth=0)
        # Median line
        ax.plot(x_years, med, marker="o", ms=3, label=f"{inst} median", color=color)

    ax.set_xticks(x_years)
    ax.set_xticklabels(list(map(str, years)), rotation=0)
    ax.set_xlabel("Fiscal Year")
    ax.set_ylabel("Solar Time Offset (hours)")
    ax.set_title("Solar Time Offset (hours) by Fiscal Year — Median and IQR per Instrument")
    ax.grid(True, which="both", linestyle=":", linewidth=0.3)

    # Build a concise legend: group IQR/median per instrument
    ax.legend(ncols=min(3, max(1, len(inst_order))), fontsize=9)

    out_path = FIG_DIR / "solar_time_offset_by_fy_instrument_hours.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote figure: %s", out_path)

    # --- Additional figure: scatter plot of sample counts (n) per FY and instrument ---
    fig2, ax2 = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
    for idx, inst in enumerate(inst_order):
        color = base_colors[idx % len(base_colors)]
        sub = df[df["instrument"] == inst].set_index("fy").reindex(years)
        nvals = sub["sample_count"].to_numpy()
        ax2.scatter(x_years, nvals, s=16, label=f"{inst} n", color=color)

    ax2.set_xticks(x_years)
    ax2.set_xticklabels(list(map(str, years)), rotation=0)
    ax2.set_xlabel("Fiscal Year")
    ax2.set_ylabel("Sample count")
    ax2.set_title("Samples per Fiscal Year by Instrument")
    ax2.grid(True, which="both", linestyle=":", linewidth=0.3)
    ax2.legend(ncols=min(3, max(1, len(inst_order))), fontsize=9)
    ax2.set_yscale("log")  # counts often vary widely; log makes patterns clearer

    out_path_counts = FIG_DIR / "solar_time_offset_by_fy_instrument_counts.png"
    fig2.savefig(out_path_counts, dpi=150)
    plt.close(fig2)
    logger.info("Wrote figure: %s", out_path_counts)


def solar_time_offset_by_year_month_plot():
    """Plot median and IQR (25-75%) of solar_time_offset (hours) by **month**, split by instrument.
    Output: ribbon plot per instrument over monthly timeline.
    """
    logger.info("Creating monthly solar offset plot")
    # Search both Dove and SkySat solar outputs (coastal_results_solar)
    f_pat = "*/coastal_results_solar/*/*/*/coastal_points.parquet"
    pat = str(BASE / f_pat)

    all_files = list(BASE.glob(f_pat))
    if not all_files:
        logger.error("No solar parquet files found in coastal_results_solar paths.")
        raise FileNotFoundError("No coastal_results_solar parquet files found")

    con.execute(
        f"""
        CREATE OR REPLACE VIEW samples_all AS
        SELECT * FROM read_parquet('{pat}')
        """
    )
    logger.info("Registered DuckDB view 'samples_all' for solar_time_offset plot")

    grid_ids = grids_df.index.to_list()
    grid_tbl = pd.DataFrame({"grid_id": grid_ids})
    con.register("grid_ids_tbl", grid_tbl)  # exposes it as a DuckDB view

    # Aggregate mean + quantiles by FY and instrument
    query = """
        SELECT
            s.instrument,
            DATE_TRUNC('month', s.acquired)                             AS month,
            avg(s.solar_time_offset_seconds) / 3600.0                   AS mean_hours,
            quantile_cont(s.solar_time_offset_seconds, 0.25) / 3600.0   AS q25_hours,
            median(s.solar_time_offset_seconds) / 3600.0                AS median_hours,
            quantile_cont(s.solar_time_offset_seconds, 0.75) / 3600.0   AS q75_hours,
            count(*)                                                    AS sample_count
        FROM samples_all as s
        JOIN grid_ids_tbl USING (grid_id)
        WHERE s.item_type = 'PSScene'
            AND s.coverage_pct > 0.5
            AND s.acquired >= '2015-12-01'
            AND s.acquired < '2024-12-01'
            AND s.publishing_stage = 'finalized'
            AND s.quality_category = 'standard'
            AND s.clear_percent    > 75.0
            AND s.has_sr_asset
            AND s.ground_control
        GROUP BY instrument, month
        ORDER BY instrument, month
    """
    df = con.execute(query).fetchdf()
    if df.empty:
        logger.warning("No rows returned for solar_time_offset_seconds aggregation")
        return

    # Ensure datetime for plotting
    df["month"] = pd.to_datetime(df["month"])  # pandas datetime64[ns]

    # Define continuous monthly domain: 2016-01 through 2023-12 (8 years × 12 months)
    months = pd.date_range(df.month.min(), "2024-12-01", freq="MS")

    # Ensure instruments are stable order
    inst_order = sorted(df["instrument"].unique())

    # Pivot for plotting grouped bars (mean as height, IQR as error bars)
    # We'll create one bar per instrument per FY with asymmetric errors: [mean - q25, q75 - mean]
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)

    # Assign fixed colors (requested): blue, green, red; fall back to cycle if more
    base_colors = ["tab:blue", "tab:green", "tab:red"]

    for idx, inst in enumerate(inst_order):
        color = base_colors[idx % len(base_colors)]
        sub = df[df["instrument"] == inst].set_index("month").reindex(months)

        med = sub["median_hours"].to_numpy()
        q25 = sub["q25_hours"].to_numpy()
        q75 = sub["q75_hours"].to_numpy()

        # Avoid NaN spans in ribbon
        finite = np.isfinite(q25) & np.isfinite(q75)
        if finite.any():
            ax.fill_between(
                months[finite], q25[finite], q75[finite], alpha=0.25, label=f"{inst} IQR", color=color, linewidth=0
            )
        # Median line (NaNs will break the line where data is missing)
        ax.plot(months, med, marker="o", ms=2.5, label=f"{inst} median", color=color)

    # Axis formatting
    ax.set_xlabel("Month")
    ax.set_ylabel("Solar Time Offset (hours)")
    ax.set_title("Monthly Solar Time Offset (hours) — Median and IQR per Instrument")
    ax.grid(True, which="both", linestyle=":", linewidth=0.3)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))

    # Build a concise legend: group IQR/median per instrument
    ax.legend(ncols=min(3, max(1, len(inst_order))), fontsize=9)

    out_path = FIG_DIR / "solar_time_offset_by_year_month_instrument_hours.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote figure: %s", out_path)

    # --- Additional figure: scatter plot of sample counts (n) per month and instrument ---
    fig2, ax2 = plt.subplots(figsize=(10, 3.6), constrained_layout=True)
    for idx, inst in enumerate(inst_order):
        color = base_colors[idx % len(base_colors)]
        sub = df[df["instrument"] == inst].set_index("month").reindex(months)
        nvals = sub["sample_count"].to_numpy()
        ax2.scatter(months, nvals, s=8, label=f"{inst} n", color=color)

    ax2.set_xlabel("Month")
    ax2.set_ylabel("Sample count")
    ax2.set_title("Samples per Month by Instrument")
    ax2.grid(True, which="both", linestyle=":", linewidth=0.3)

    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))

    ax2.legend(ncols=min(3, max(1, len(inst_order))), fontsize=9)
    ax2.set_yscale("log")  # optional; remove if you prefer linear scale

    out_path_counts_m = FIG_DIR / "solar_time_offset_by_year_month_instrument_counts.png"
    fig2.savefig(out_path_counts_m, dpi=150)
    plt.close(fig2)
    logger.info("Wrote figure: %s", out_path_counts_m)


def plot_median_and_std_geo():
    f_pat = "*/coastal_results_solar/*/*/*/coastal_points.parquet"
    pat = str(BASE / f_pat)

    all_files = list(BASE.glob(f_pat))
    if not all_files:
        logger.error("No solar parquet files found in coastal_results_solar paths.")
        raise FileNotFoundError("No coastal_results_solar parquet files found")

    con.execute(
        f"""
        CREATE OR REPLACE VIEW samples_all AS
        SELECT * FROM read_parquet('{pat}')
        """
    )
    logger.info("Registered DuckDB view 'samples_all' for solar_time_offset plot")

    query = """
        SELECT
            grid_id,
            median(solar_time_offset_seconds) / 3600.0      AS median_hours,
            stddev_samp(solar_time_offset_seconds) / 60.0   AS std_minutes
        FROM samples_all
        WHERE
            item_type           = 'PSScene'
            AND coverage_pct    > 0.5
            AND acquired        >=  TIMESTAMP '2015-12-01'
            AND acquired        <   TIMESTAMP '2024-12-01'
            AND publishing_stage = 'finalized'
            AND quality_category = 'standard'
            AND clear_percent    > 75.0
            AND has_sr_asset
            AND ground_control
        GROUP BY grid_id
    """

    df = con.execute(query).fetchdf().set_index("grid_id")

    logger.info("Query finished")

    hex_df = grids_df[["geometry", "hex_id"]].join(df, how="inner")
    hex_df["std_minutes"] += 1

    logger.info("Plotting Counts")
    agg = hex_df.groupby("hex_id").agg(
        median_median_hours=("median_hours", "median"),
        median_std_minutes=("std_minutes", "median"),
    )
    agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
    gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=grids_df.crs)

    plot_gdf_column(
        gdf=gdf,
        column="median_median_hours",
        title="Median Solar Time Offset (hours)",
        title_fontsize=15,
        title_padsize=10,
        save_path=FIG_DIR / "median_geo.png",
        use_cbar_label=False,
        vmin=8.5,
        vmax=12.5,
    )
    vmin = np.percentile(gdf.median_std_minutes[~gdf.median_std_minutes.isna()], 5).astype(np.int32).item()
    vmin = vmin - (vmin % 10)
    plot_gdf_column(
        gdf=gdf,
        column="median_std_minutes",
        title="Std Solar Time Offset (minutes)",
        title_fontsize=15,
        title_padsize=10,
        save_path=FIG_DIR / "std_geo.png",
        use_cbar_label=False,
        vmin=vmin,
        vmax=80,
    )


plot_median_and_std_geo()
# solar_time_offset_by_fy_plot()
# solar_time_offset_by_year_month_plot()
logger.info("Done")
