import logging
import warnings
from pathlib import Path

import duckdb
import geopandas as gpd

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
lats = grids_df.centroid.y
valid = ~grids_df.is_land & ~grids_df.dist_km.isna() & (grids_df.dist_km < MIN_DIST) & (lats > -81.5) & (lats < 81.5)
grids_df = grids_df[valid].copy()
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

    hex_df = grids_df[["geometry", "hex_id"]].join(df, how="left")

    logger.info("Plotting Counts")
    agg = hex_df.groupby("hex_id").agg(
        median_count=("sample_count", "median"),
        max_count=("sample_count", "max"),
    )
    agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
    gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=grids_df.crs)

    plot_gdf_column(
        gdf=gdf,
        column="median_count",
        title="Median PlanetScope Sample Count (12/2015-12/2024)",
        title_fontsize=15,
        vmin=10,
        vmax=5500,
        save_path=FIG_DIR / "median.png",
        use_cbar_label=False,
    )

    logger.info("Saving results to ShapeFile")
    (FIG_DIR / "hex_data").mkdir(exist_ok=True)
    gdf.to_file(FIG_DIR / "hex_data" / "data.shp")

    (FIG_DIR / "grid_data").mkdir(exist_ok=True)
    gdf = gpd.GeoDataFrame(hex_df, geometry="geometry", crs=grids_df.crs)
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

    hex_df = grids_df[["geometry", "hex_id", "dist_km"]].join(df, how="left")

    logger.info("Plotting Counts")
    agg = hex_df.groupby("hex_id").agg(
        sum_count=("sample_count", "sum"),
        max_count=("sample_count", "max"),
    )
    agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
    gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=grids_df.crs)

    plot_gdf_column(
        gdf=gdf,
        column="sum_count",
        title="Sum SkySat Sample Count (12/2015-12/2024)",
        title_fontsize=15,
        scale="log",
        # vmin=10,
        vmax=4500,
        save_path=FIG_DIR / "sky_sat_sum.png",
        use_cbar_label=False,
    )

    logger.info("Saving results to ShapeFile")
    (FIG_DIR / "skysat_hex_data").mkdir(exist_ok=True)
    gdf.to_file(FIG_DIR / "hex_data" / "data.shp")

    (FIG_DIR / "skysat_grid_data").mkdir(exist_ok=True)
    gdf = gpd.GeoDataFrame(hex_df, geometry="geometry", crs=grids_df.crs)
    gdf.to_file(FIG_DIR / "grid_data" / "data.shp")


dove_coverage()
# skysat_coverage()

logger.info("Done")
