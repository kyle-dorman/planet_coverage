import logging
import warnings
from pathlib import Path

import cartopy.crs as ccrs
import duckdb
import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt

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
valid = ~grids_df.is_land & ~grids_df.dist_km.isna() & (grids_df.dist_km < MIN_DIST) & ~grids_df.tide_range.isna()
grids_df = grids_df[valid].copy()


pct = 90
FIG_DIR = BASE.parent / "figs_v2" / f"tide_{pct}"
FIG_DIR.mkdir(exist_ok=True, parents=True)


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
    MIN(tide_height)    AS obs_min_tide_height,
    MAX(tide_height)    AS obs_max_tide_height,
    COUNT(*)            AS sample_count,
FROM samples_all
WHERE
        acquired         >= TIMESTAMP '2023-12-01'
    AND acquired         <  TIMESTAMP '2024-12-01'
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

# Keep all tidal grids
geo_tide = grids_df.join(df, how="left")
geo_tide["obs_tide_range"] = geo_tide.obs_max_tide_height - geo_tide.obs_min_tide_height
geo_tide["obs_high_tide_offset"] = geo_tide.tide_max - geo_tide.obs_max_tide_height
geo_tide["obs_low_tide_offset"] = geo_tide.obs_min_tide_height - geo_tide.tide_min
geo_tide["tide_range_coverage"] = geo_tide["obs_tide_range"] / geo_tide["tide_range"]

logger.info("Joined tide extremes with coastal grids: geo_tide has %d rows", len(geo_tide))

hex_tide = geo_tide.groupby("hex_id").agg(  # keep one row per hex_id
    min_tide_range_coverage=("tide_range_coverage", "min"),
    max_tide_range_coverage=("tide_range_coverage", "max"),
    median_tide_range_coverage=("tide_range_coverage", "median"),
    min_obs_high_tide_offset=("obs_high_tide_offset", "min"),
    max_obs_high_tide_offset=("obs_high_tide_offset", "max"),
    median_obs_high_tide_offset=("obs_high_tide_offset", "median"),
    min_obs_low_tide_offset=("obs_low_tide_offset", "min"),
    max_obs_low_tide_offset=("obs_low_tide_offset", "max"),
    median_obs_low_tide_offset=("obs_low_tide_offset", "median"),
)

logger.info("Aggregated tide data to %d hexagons", len(hex_tide))

hex_tide = hex_tide[hex_tide.index >= 0].join(hex_grid[["geometry"]])
gdf = gpd.GeoDataFrame(hex_tide, geometry="geometry")

logger.info("Saving results to ShapeFile")
(FIG_DIR / "hex_tide_range").mkdir(exist_ok=True)
gdf.to_file(FIG_DIR / "hex_tide_range" / "data.shp")

logger.info("Plotting tide_range_coverage maps")


fig, axes = plt.subplots(
    2, 2, figsize=(6 * 2, 3.4 * 2), constrained_layout=True, subplot_kw={"projection": ccrs.Robinson()}
)

plot_gdf_column(
    gdf,
    "median_obs_high_tide_offset",
    title="High Tide Offset (m)",
    show_land_ocean=True,
    ax=axes[1, 0],
    vmax=1.5,
    use_cbar_label=False,
)
plot_gdf_column(
    gdf,
    "median_obs_low_tide_offset",
    title="Low Tide Offset (m)",
    show_land_ocean=True,
    ax=axes[1, 1],
    vmax=1.5,
    use_cbar_label=False,
)
plot_gdf_column(
    gdf,
    "median_tide_range_coverage",
    title="Tide Range Coverage (%)",
    show_land_ocean=True,
    ax=axes[0, 1],
    vmin=0.0,
    vmax=1.0,
    use_cbar_label=False,
)


# Use human readable log scale edges
bin_edges = np.array([0, 1, 2, 7, 14, 30, 60, 90, 180, 365], dtype=np.int32)
bin_right = bin_edges[1:]  # right edge of each bar
year = 2023

extra_filter = "AND is_mid_tide AND has_tide_data"
query = make_time_between_query(year, pct, valid_only=True, extra_filter=extra_filter)
df = con.execute(query).fetchdf().set_index("grid_id")
hex_df = grids_df[["hex_id", "dist_km"]].join(df, how="left")

agg = hex_df.groupby("hex_id").agg(
    median_days_between=(f"p{pct}_days_between", "median"),
    max_days_between=(f"p{pct}_days_between", "max"),
    min_days_between=(f"p{pct}_days_between", "min"),
)
agg = agg[agg.index >= 0].join(hex_grid[["geometry"]], how="inner")
gdf = gpd.GeoDataFrame(agg, geometry="geometry")

plt_bin_edges = np.array([0, 4, 7, 14, 30, 60, 90, 366], dtype=np.int32)
plot_gdf_column(
    gdf,
    "median_days_between",
    title=f"p{pct} Time Between Mid-Tide Samples",
    show_land_ocean=True,
    # vmax=180,
    ax=axes[0, 0],
    use_cbar_label=False,
    bins=plt_bin_edges.tolist(),
)

fig.suptitle("Tidal Coverage", fontsize=14)

plt.savefig(FIG_DIR / "four_pannel.png", dpi=300)
plt.close(fig)

logger.info("Saving results to ShapeFile")
(FIG_DIR / "hex_days_between").mkdir(exist_ok=True)
gdf.to_file(FIG_DIR / "hex_days_between" / "data.shp")
