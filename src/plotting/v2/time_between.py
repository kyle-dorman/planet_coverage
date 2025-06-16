import logging
import warnings
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.ticker import NullLocator
from tqdm import tqdm

from src.plotting.util import load_grids, make_time_between_query, plot_gdf_column

warnings.filterwarnings("ignore")  # hide every warning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


BASE = Path("/Users/kyledorman/data/planet_coverage/points_30km/")  # <-- update this
SHORELINES = BASE.parent / "shorelines"
FIG_DIR = BASE.parent / "figs_v2" / "time_between"
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


# nbins = 10
# Use human readable log scale edges
bin_edges = np.array([0, 1, 2, 7, 14, 30, 60, 90, 180, 365], dtype=np.int32)
bin_right = bin_edges[1:]  # right edge of each bar
pct = 90


# --------------------- Yearly Comparision ------------------------
def yearly_plots():
    start_year = 2015
    end_year = 2024
    valid = True

    all_years_df = []

    for year in tqdm(range(start_year, end_year), total=end_year - start_year):
        disp_year = year + 1

        query = make_time_between_query(year, pct, valid)

        df = con.execute(query).fetchdf().set_index("grid_id")
        hex_df = grids_df[["geometry", "hex_id", "dist_km"]].join(df, how="left").fillna({f"p{pct}_days_between": 365})

        hex_df["year"] = year

        # Filter out Antartica
        hex_df = hex_df[~hex_df.dist_km.isna()]

        all_years_df.append(hex_df.copy().reset_index())

        agg = hex_df.groupby("hex_id").agg(
            median_days_between=(f"p{pct}_days_between", "median"),
            min_days_between=(f"p{pct}_days_between", "min"),
        )
        agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
        gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=grids_df.crs)

        title = f"p{pct} Days Between Samples (Year: {disp_year}, Agg: Median)"
        plot_gdf_column(
            gdf,
            "median_days_between",
            title=title,
            show_land_ocean=True,
            save_path=FIG_DIR / f"median_days_between_{disp_year}.png",
            vmax=180,
        )
        title = f"p{pct} Days Between Samples (Year: {disp_year}, Agg: Min)"
        plot_gdf_column(
            gdf,
            "min_days_between",
            title=title,
            show_land_ocean=True,
            save_path=FIG_DIR / f"min_days_between_{disp_year}.png",
            vmax=180,
        )

    # Aggregate over all years per grid and then group by hex_id.
    df = pd.concat(all_years_df, ignore_index=True)
    gdf = gpd.GeoDataFrame(hex_df.copy(), geometry="geometry", crs=grids_df.crs)
    logger.info("Saving yearly grid results to ShapeFile")
    (FIG_DIR / "per_grid_per_year").mkdir(exist_ok=True)
    gdf.to_file(FIG_DIR / "per_grid_per_year" / "data.shp")

    # Prepare colormap for years
    num_years = end_year - start_year
    cmap_years = cm.get_cmap("viridis", num_years)

    # Cumulative sum plot
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(6, 4),
        constrained_layout=True,
    )

    for year, group in df.groupby("year"):
        year = int(year)  # type: ignore
        disp_year = year + 1
        counts, _ = np.histogram(group[f"p{pct}_days_between"], bin_edges)
        cumsum = np.cumsum(counts)
        total = cumsum[-1] if cumsum[-1] else 1.0  # prevent divide-by-zero
        cumsum_pct = cumsum / total * 100.0  # → 0-100 %

        color = cmap_years(year - start_year)  # consistent color per year
        ax.plot(
            bin_right,
            cumsum_pct,
            marker="o",
            linestyle="-",
            color=color,
            label=str(disp_year),
        )

    ax.set_ylim(0, 100)
    ax.set_xscale("log")
    ax.set_xticks(bin_edges[1:])
    ax.set_xticklabels(
        [f"{b:g}" for b in bin_edges[1:]],
        rotation=45,
        ha="right",
    )
    ax.xaxis.set_minor_locator(NullLocator())
    # ax.tick_params(axis="both", labelsize=8)
    ax.axhline(50, linestyle="--", linewidth=0.8, label="50 %", color="red")
    ax.axhline(95, linestyle=":", linewidth=0.8, label="95 %", color="green")
    ax.legend()

    fig.supylabel("Cumulative Grid Cell %", fontsize=12)
    fig.supxlabel("Time Between Samples (days)", fontsize=12)
    fig.suptitle(
        f"Cumulative Distribution p{pct} Time Between Samples",
        fontsize=14,
    )

    plt.savefig(FIG_DIR / f"cumsum_p{pct}_time_between_samples_by_year.png")
    plt.close(fig)

    logger.info("Created Yearly Cumulative distribution plot")

    grid_grouped = df.groupby("grid_id").agg(
        median_days_between=(f"p{pct}_days_between", "median"),
    )
    hex_df = (
        grids_df[["geometry", "hex_id", "dist_km"]].join(grid_grouped, how="left").fillna({f"p{pct}_days_between": 365})
    )
    # Filter out Antartica
    hex_df = hex_df[~hex_df.dist_km.isna()]
    gdf = gpd.GeoDataFrame(hex_df.copy(), geometry="geometry", crs=grids_df.crs)
    # Save data
    logger.info("Saving per grid yearly aggregated grid results to ShapeFile")
    (FIG_DIR / "per_grid_year_agg").mkdir(exist_ok=True)
    gdf.to_file(FIG_DIR / "per_grid_year_agg" / "data.shp")

    agg = hex_df.groupby("hex_id").agg(
        median_median_days_between=("median_days_between", "median"),
        min_median_days_between=("median_days_between", "min"),
    )
    agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
    gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=grids_df.crs)

    # Save data
    logger.info("Saving yearly aggregated results to ShapeFile")
    (FIG_DIR / "hex_year_agg").mkdir(exist_ok=True)
    gdf.to_file(FIG_DIR / "hex_year_agg" / "data.shp")

    title = f"p{pct} Days Between Samples (Year Agg: Median, Agg: Median)"
    plot_gdf_column(
        gdf,
        "median_median_days_between",
        title=title,
        show_land_ocean=True,
        save_path=FIG_DIR / "median_median_days_between.png",
        vmax=180,
    )
    title = f"p{pct} Days Between Samples (Year Agg: Median, Agg: Min)"
    plot_gdf_column(
        gdf,
        "min_median_days_between",
        title=title,
        show_land_ocean=True,
        save_path=FIG_DIR / "min_median_days_between.png",
        vmax=180,
    )

    logger.info("Done with yearly plots")


# --------------------- Clear % Comparision ------------------------
def clear_pct_plots():
    # Prepare colormap for clear %
    clear_pcts = [None, 0, 25, 50, 75, 100]
    num_rows = len(clear_pcts)
    cmap_clear = cm.get_cmap("viridis", num_rows)
    year = 2023
    disp_year = year + 1

    # Cumulative sum plot
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(6, 4),
        constrained_layout=True,
    )

    all_levels_df = []

    for i, clear_pct in tqdm(enumerate(clear_pcts), total=len(clear_pcts)):
        if clear_pct is None:
            valid_filter = ""
            label = "all"
        else:
            valid_filter = f"""
                AND publishing_stage = 'finalized'
                AND quality_category = 'standard'
                AND clear_percent    >= {clear_pct}
                AND has_sr_asset
                AND ground_control
            """
            label = f"clear={clear_pct}%"
        query = make_time_between_query(year, pct, valid_only=False, extra_filter=valid_filter)

        df = con.execute(query).fetchdf().set_index("grid_id")
        hex_df = grids_df[["geometry", "dist_km"]].join(df, how="left").fillna({f"p{pct}_days_between": 365})
        hex_df["clear_pct"] = clear_pct

        # Filter out Antartica
        hex_df = hex_df[~hex_df.dist_km.isna()]

        all_levels_df.append(hex_df.reset_index().copy())

        counts, _ = np.histogram(hex_df[f"p{pct}_days_between"], bin_edges)
        cumsum = np.cumsum(counts)
        total = cumsum[-1] if cumsum[-1] else 1.0  # prevent divide-by-zero
        cumsum_pct = cumsum / total * 100.0  # → 0-100 %

        color = cmap_clear(i)
        ax.plot(
            bin_right,
            cumsum_pct,
            marker="o",
            linestyle="-",
            color=color,
            label=label,
        )

    ax.set_ylim(0, 100)
    ax.set_xscale("log")
    ax.set_xticks(bin_edges[1:])
    ax.set_xticklabels(
        [f"{b:g}" for b in bin_edges[1:]],
        rotation=45,
        ha="right",
    )
    ax.xaxis.set_minor_locator(NullLocator())
    # ax.tick_params(axis="both", labelsize=8)
    ax.axhline(50, linestyle="--", linewidth=0.8, label="50 %", color="red")
    ax.axhline(95, linestyle=":", linewidth=0.8, label="95 %", color="green")
    ax.legend()

    fig.supylabel("Cumulative Grid Cell %", fontsize=12)
    fig.supxlabel("Time Between Samples (days)", fontsize=12)
    fig.suptitle(
        f"Cumulative Distribution p{pct} Time Between Samples ({disp_year})",
        fontsize=14,
    )

    plt.savefig(FIG_DIR / f"cumsum_p{pct}_time_between_samples_by_clear_pct_{disp_year}.png")
    plt.close(fig)

    df = pd.concat(all_levels_df, ignore_index=True)
    gdf = gpd.GeoDataFrame(hex_df.copy(), geometry="geometry", crs=grids_df.crs)
    # Save data
    logger.info("Saving per grid cloud cover results to ShapeFile")
    (FIG_DIR / "per_grid_clear_pct").mkdir(exist_ok=True)
    gdf.to_file(FIG_DIR / "per_grid_clear_pct" / "data.shp")

    logger.info("Done with clear % plots")


yearly_plots()
clear_pct_plots()
