import logging
import warnings
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np

from src.plotting.util import load_grids, plot_gdf_column

warnings.filterwarnings("ignore")  # hide every warning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


BASE = Path("/Users/kyledorman/data/planet_coverage/points_30km/")
SHORELINES = BASE.parent / "shorelines"
FIG_DIR = BASE.parent / "figs_v2" / "total_coverage"
FIG_DIR.mkdir(exist_ok=True, parents=True)

# path patterns
f_pattern = "*/coastal_results_old/*/*/*/coastal_points.parquet"
all_files_pattern = str(BASE / f_pattern)

# Combined list used later when we search individual files
all_parquets = list(BASE.glob(f_pattern))

if not all_parquets:
    logger.error("No parquet files found matching pattern %s", all_files_pattern)
    raise FileNotFoundError("No parquet files found")
logger.info("Found %d parquet files", len(all_parquets))


query_df, grids_df, hex_grid = load_grids(SHORELINES)
MIN_DIST = 20.0
# lats = grids_df.centroid.y
valid = grids_df.dist_km < MIN_DIST  # & ~grids_df.is_land & ~grids_df.dist_km.isna() & (lats > -81.5) & (lats < 81.5)
grids_df = grids_df[valid].copy()

assert grids_df.crs is not None
inter = gpd.sjoin(
    hex_grid.to_crs(grids_df.crs).reset_index()[["geometry", "hex_id"]], grids_df.reset_index()[["geometry", "grid_id"]]
)
counts = inter[["hex_id", "grid_id"]].groupby("hex_id").count()
hex_grid["grid_count"] = 0.0
hex_grid.loc[counts.index, "grid_count"] = counts.grid_id

logger.info("Loaded grid dataframes")

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


def dove_coverage():
    logger.info("Plotting PSScene")

    query = """
        SELECT
            grid_id,
            COUNT(id) AS sample_count,
        FROM samples_all
        WHERE
            item_type           = 'PSScene'
            AND coverage_pct    > 0.5
            AND acquired        <  TIMESTAMP '2024-12-01'
            AND acquired        >  TIMESTAMP '2015-12-01'
        GROUP BY grid_id
    """

    df = con.execute(query).fetchdf().set_index("grid_id")

    logger.info("Query finished")

    grids_data_df = grids_df.join(df, how="left").fillna(0.0)

    logger.info("Plotting Counts")
    agg = grids_data_df.groupby("hex_id").agg(
        median_count=("sample_count", "median"),
        max_count=("sample_count", "max"),
        sum_count=("sample_count", "sum"),
    )
    agg = agg[agg.index >= 0].join(hex_grid[["geometry", "grid_count"]])
    for key in ["median_count", "max_count", "sum_count"]:
        agg.loc[agg.sum_count == 0, key] = np.nan
    gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=grids_df.crs)

    plot_gdf_column(
        gdf=gdf,
        column="median_count",
        title="Median PlanetScope Sample Count (12/2015-12/2024)",
        title_fontsize=15,
        vmin=10,
        vmax=3000,
        save_path=FIG_DIR / "median_dove.png",
        use_cbar_label=False,
    )

    logger.info("Saving results to ShapeFile")
    (FIG_DIR / "hex_data").mkdir(exist_ok=True)
    gdf.to_file(FIG_DIR / "hex_data" / "data.shp")

    (FIG_DIR / "grid_data").mkdir(exist_ok=True)
    gdf = gpd.GeoDataFrame(grids_data_df, geometry="geometry", crs=grids_df.crs)
    gdf.to_file(FIG_DIR / "grid_data" / "data.shp")


def skysat_coverage():
    logger.info("Plotting Skysat")

    query = """
        SELECT
            grid_id,
            COUNT(id) AS sample_count,
        FROM samples_all
        WHERE
            item_type           = 'SkySatCollect'
            AND coverage_pct    > 0.5
            AND acquired        <  TIMESTAMP '2024-12-01'
            AND acquired        >  TIMESTAMP '2015-12-01'
        GROUP BY grid_id
    """

    df = con.execute(query).fetchdf().set_index("grid_id")

    logger.info("Query finished")

    near_grids_df = grids_df[(grids_df.dist_km < 4.0) & ~grids_df.is_land]
    grids_data_df = near_grids_df.join(df, how="left").fillna(0.0)

    print("% Grids with atleast 1 samples")
    print(round(100 * (grids_data_df.sample_count > 0).sum() / len(grids_data_df), 2))

    print("% Grids with atleast 5 samples")
    print(round(100 * (grids_data_df.sample_count > 4).sum() / len(grids_data_df), 2))

    print("% Grids with atleast 100 samples")
    print(round(100 * (grids_data_df.sample_count > 99).sum() / len(grids_data_df), 2))

    logger.info("Plotting Counts")
    agg = grids_data_df.groupby("hex_id").agg(
        sum_count=("sample_count", "sum"),
        max_count=("sample_count", "max"),
    )
    agg = agg[agg.index >= 0].join(hex_grid[["geometry", "grid_count"]])
    for key in ["sum_count", "max_count"]:
        agg.loc[agg.sum_count == 0, key] = np.nan
    gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=grids_df.crs)

    plot_gdf_column(
        gdf=gdf,
        column="sum_count",
        title="Sum SkySat Sample Count (12/2015-12/2024)",
        title_fontsize=15,
        scale="log",
        # vmin=10,
        # vmax=4500,
        save_path=FIG_DIR / "sky_sat_sum.png",
        use_cbar_label=False,
    )
    plot_gdf_column(
        gdf=gdf,
        column="max_count",
        title="Max SkySat Sample Count (12/2015-12/2024)",
        title_fontsize=15,
        scale="log",
        # vmin=10,
        vmax=1800,
        save_path=FIG_DIR / "sky_sat_max.png",
        use_cbar_label=False,
    )

    logger.info("Saving results to ShapeFile")
    (FIG_DIR / "skysat_hex_data").mkdir(exist_ok=True)
    gdf.to_file(FIG_DIR / "hex_data" / "data.shp")

    (FIG_DIR / "skysat_grid_data").mkdir(exist_ok=True)
    gdf = gpd.GeoDataFrame(grids_data_df, geometry="geometry", crs=grids_df.crs)
    gdf.to_file(FIG_DIR / "grid_data" / "data.shp")


# dove_coverage()
skysat_coverage()

logger.info("Done")
