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


BASE = Path("/Users/kyledorman/data/planet_coverage/points_30km/")  # <-- update this
FIG_DIR = BASE.parent / "figs" / BASE.name / "sample_count"
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


query = """
    SELECT
        grid_id,
        COUNT(*) AS sample_count,
        MIN(acquired) as first_sample,
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

df = con.execute(query).fetchdf().set_index("grid_id")

logger.info("Query finished")

hex_df = grids_df.join(df, how="left").fillna({"sample_count": 0})

logger.info("Plotting Valid First Date Seen")
agg = (
    hex_df.dropna(subset=["first_sample"])
    .groupby("hex_id")
    .agg(
        min_first_sample=("first_sample", "min"),
        median_first_sample=("first_sample", "median"),
    )
)
agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
gdf = gpd.GeoDataFrame(agg, geometry="geometry")

plot_gdf_column(
    gdf,
    "min_first_sample",
    title="Minimum First Sample Recorded (Valid)",
    show_land_ocean=True,
    save_path=FIG_DIR / "min_first_sample_valid.png",
)

logger.info("Plotting Valid Counts")
agg = hex_df.groupby("hex_id").agg(
    median_count=("sample_count", "median"),
    min_count=("sample_count", "min"),
    max_count=("sample_count", "max"),
)
agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
agg["min_count_binary"] = agg["min_count"] > 0
gdf = gpd.GeoDataFrame(agg, geometry="geometry")

plot_gdf_column(
    gdf[gdf.median_count > 0],
    "median_count",
    title="All Time Sample Count (Agg: Median)",
    show_land_ocean=True,
    scale="log",
    save_path=FIG_DIR / "median_count_valid.png",
)

plot_gdf_column(
    gdf[gdf.max_count > 0],
    "max_count",
    title="All Time Sample Count (Agg: Max)",
    show_land_ocean=True,
    scale="log",
    save_path=FIG_DIR / "max_count_valid.png",
)

plot_gdf_column(
    gdf,
    "min_count_binary",
    title="Has A Valid Sample",
    show_land_ocean=True,
    save_path=FIG_DIR / "bool_count_valid.png",
)

# ______________________________________________________

query = """
    SELECT
        grid_id,
        MIN(acquired) as first_sample,
        COUNT(*) AS sample_count,
    FROM samples_all
    WHERE
        item_type = 'PSScene'
        AND coverage_pct > 0.5
    GROUP BY grid_id
"""

df = con.execute(query).fetchdf().set_index("grid_id")
hex_df = grids_df.join(df, how="left").fillna({"sample_count": 0})

logger.info("Query finished")

logger.info("Plotting All First Date Seen")
agg = (
    hex_df.dropna(subset=["first_sample"])
    .groupby("hex_id")
    .agg(
        min_first_sample=("first_sample", "min"),
        median_first_sample=("first_sample", "median"),
    )
)
agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
gdf = gpd.GeoDataFrame(agg, geometry="geometry")

plot_gdf_column(
    gdf,
    "min_first_sample",
    title="Minimum First Sample Recorded (All)",
    show_land_ocean=True,
    save_path=FIG_DIR / "min_first_sample_all.png",
)

logger.info("Plotting All Counts")
agg = hex_df.groupby("hex_id").agg(
    median_count=("sample_count", "median"),
    max_count=("sample_count", "max"),
    min_count=("sample_count", "min"),
)
agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
agg["min_count_binary"] = agg["min_count"] > 0
gdf = gpd.GeoDataFrame(agg, geometry="geometry")

plot_gdf_column(
    gdf[gdf.median_count > 0],
    "median_count",
    title="All Time Sample Count (Agg: Median)",
    show_land_ocean=True,
    scale="log",
    save_path=FIG_DIR / "median_count_all.png",
)

plot_gdf_column(
    gdf[gdf.max_count > 0],
    "max_count",
    title="All Time Sample Count (Agg: max)",
    show_land_ocean=True,
    scale="log",
    save_path=FIG_DIR / "max_count_all.png",
)

plot_gdf_column(
    gdf,
    "min_count_binary",
    title="Has A Sample",
    show_land_ocean=True,
    save_path=FIG_DIR / "bool_count_all.png",
)
