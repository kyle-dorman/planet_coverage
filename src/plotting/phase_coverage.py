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
FIG_DIR = BASE.parent / "figs" / BASE.name
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


query = """
SELECT
    grid_id,
    MIN(tide_height) AS obs_min_tide_height,
    MAX(tide_height) AS obs_max_tide_height
FROM samples_all
WHERE
    acquired >= TIMESTAMP '2023-12-01'
    AND acquired <  TIMESTAMP '2025-01-01'
    AND tide_height IS NOT NULL
    AND item_type        = 'PSScene'
    AND publishing_stage = 'finalized'
    AND quality_category = 'standard'
    AND has_sr_asset
    AND ground_control
GROUP BY grid_id
ORDER BY grid_id;
"""

df = con.execute(query).fetchdf().set_index("grid_id").dropna(subset=["obs_min_tide_height", "obs_max_tide_height"])
logger.info("Queried observed tide extremes for %d grids", len(df))

geo_tide = grids_df.join(df, how="left").dropna(subset=["obs_min_tide_height", "obs_max_tide_height", "tide_range"])
geo_tide["obs_tide_range"] = geo_tide.obs_max_tide_height - geo_tide.obs_min_tide_height
geo_tide["phase_coverage"] = geo_tide["obs_tide_range"] / geo_tide["tide_range"]
geo_tide.to_file("grid_tide.gpkg")
logger.info("Joined tide extremes with coastal grids: geo_tide has %d rows", len(geo_tide))

# plot_gdf_column(geo_tide, "phase_coverage", title="phase_coverage", show_coastlines=True)

hex_tide = geo_tide.groupby("hex_id").agg(  # keep one row per hex_id
    obs_min_tide_height=("obs_min_tide_height", "min"),  # lowest observed tide
    obs_max_tide_height=("obs_max_tide_height", "max"),  # highest observed tide
    tide_min=("tide_min", "min"),  # lowest tide
    tide_max=("tide_max", "max"),  # highest tide
)
hex_tide["tide_range"] = hex_tide.tide_max - hex_tide.tide_min
hex_tide["obs_tide_range"] = hex_tide.obs_max_tide_height - hex_tide.obs_min_tide_height
hex_tide["phase_coverage"] = hex_tide.obs_tide_range / hex_tide.tide_range

logger.info("Aggregated tide data to %d hexagons", len(hex_tide))

hex_tide = hex_tide[hex_tide.index >= 0]
hex_tide = hex_tide.join(hex_grid[["geometry"]])
gdf = gpd.GeoDataFrame(hex_tide, geometry="geometry")

logger.info("Plotting phase_coverage map ➜ %s", FIG_DIR / "phase_coverage.png")
plot_gdf_column(
    gdf,
    "phase_coverage",
    title="phase_coverage",
    show_coastlines=True,
    save_path=str(FIG_DIR / "phase_coverage.png"),
)

gdf.to_file("hex_tide.gpkg")
