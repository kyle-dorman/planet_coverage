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
FIG_DIR = BASE.parent / "figs" / BASE.name / "tide"
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
    MAX(tide_height) AS obs_max_tide_height,
    COUNT(*)         AS sample_count,
FROM samples_all
WHERE
        acquired         >= TIMESTAMP '2023-12-01'
    AND acquired         <  TIMESTAMP '2025-01-01'
    AND tide_height      IS NOT NULL
    AND item_type        = 'PSScene'
    AND publishing_stage = 'finalized'
    AND quality_category = 'standard'
    AND coverage_pct     > 0.5
    AND clear_percent    > 75.0
    AND has_sr_asset
    AND ground_control
GROUP BY grid_id
ORDER BY grid_id;
"""

df = con.execute(query).fetchdf().set_index("grid_id")
logger.info("Queried observed tide extremes for %d grids", len(df))

geo_tide = grids_df.join(df, how="left").dropna(subset=["obs_min_tide_height", "obs_max_tide_height", "tide_range"])
geo_tide["obs_tide_range"] = geo_tide.obs_max_tide_height - geo_tide.obs_min_tide_height
geo_tide["obs_high_tide_offset"] = geo_tide.tide_max - geo_tide.obs_max_tide_height
geo_tide["obs_low_tide_offset"] = geo_tide.obs_min_tide_height - geo_tide.tide_min
geo_tide["phase_coverage"] = geo_tide["obs_tide_range"] / geo_tide["tide_range"]

logger.info("Joined tide extremes with coastal grids: geo_tide has %d rows", len(geo_tide))

# plot_gdf_column(geo_tide, "phase_coverage", title="phase_coverage", show_coastlines=True)

hex_tide = geo_tide.groupby("hex_id").agg(  # keep one row per hex_id
    min_phase_coverage=("phase_coverage", "min"),
    max_phase_coverage=("phase_coverage", "max"),
    median_phase_coverage=("phase_coverage", "median"),
    min_obs_high_tide_offset=("obs_high_tide_offset", "min"),
    max_obs_high_tide_offset=("obs_high_tide_offset", "max"),
    median_obs_high_tide_offset=("obs_high_tide_offset", "median"),
    min_obs_low_tide_offset=("obs_low_tide_offset", "min"),
    max_obs_low_tide_offset=("obs_low_tide_offset", "max"),
    median_obs_low_tide_offset=("obs_low_tide_offset", "median"),
)

logger.info("Aggregated tide data to %d hexagons", len(hex_tide))

hex_tide = hex_tide[hex_tide.index >= 0]
hex_tide = hex_tide.join(hex_grid[["geometry"]])
gdf = gpd.GeoDataFrame(hex_tide, geometry="geometry")

logger.info("Plotting phase_coverage maps")

plot_gdf_column(
    gdf,
    "min_phase_coverage",
    title="Minimum Phase Coverage",
    show_coastlines=True,
    save_path=str(FIG_DIR / "min_phase_coverage.png"),
    show=False,
)
plot_gdf_column(
    gdf,
    "max_phase_coverage",
    title="Maximum Phase Coverage",
    show_coastlines=True,
    save_path=str(FIG_DIR / "max_phase_coverage.png"),
    show=False,
)
plot_gdf_column(
    gdf,
    "median_phase_coverage",
    title="Median Phase Coverage",
    show_coastlines=True,
    save_path=str(FIG_DIR / "median_phase_coverage.png"),
    show=False,
)

# plot_gdf_column(
#     gdf,
#     "min_obs_high_tide_offset",
#     title="Minimum High Tide Offset",
#     show_coastlines=True,
#     save_path=str(FIG_DIR / "min_obs_high_tide_offset.png"),
# )
# plot_gdf_column(
#     gdf,
#     "max_obs_high_tide_offset",
#     title="Maximum High Tide Offset",
#     show_coastlines=True,
#     save_path=str(FIG_DIR / "max_obs_high_tide_offset.png"),
# )
plot_gdf_column(
    gdf,
    "median_obs_high_tide_offset",
    title="Median High Tide Offset",
    show_coastlines=True,
    save_path=str(FIG_DIR / "median_obs_high_tide_offset.png"),
    show=False,
)

# plot_gdf_column(
#     gdf,
#     "min_obs_low_tide_offset",
#     title="Minimum Low Tide Offset",
#     show_coastlines=True,
#     save_path=str(FIG_DIR / "min_obs_low_tide_offset.png"),
# )
# plot_gdf_column(
#     gdf,
#     "max_obs_low_tide_offset",
#     title="Maximum Low Tide Offset",
#     show_coastlines=True,
#     save_path=str(FIG_DIR / "max_obs_low_tide_offset.png"),
# )
plot_gdf_column(
    gdf,
    "median_obs_low_tide_offset",
    title="Median Low Tide Offset",
    show_coastlines=True,
    save_path=str(FIG_DIR / "median_obs_low_tide_offset.png"),
    show=False,
)
