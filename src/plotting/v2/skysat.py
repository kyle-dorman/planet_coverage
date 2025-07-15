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
FIG_DIR = BASE.parent / "figs_v2" / "skysat_dove_2_hr"
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Example path patterns
f_pattern = "skysat_dove_v2/*/*/*/*/data.parquet"
all_files_pattern = str(BASE / f_pattern)

# Combined list used later when we search individual files
all_parquets = list(BASE.glob(f_pattern))

if not all_parquets:
    logger.error("No parquet files found matching pattern %s", all_files_pattern)
    raise FileNotFoundError("No parquet files found")
logger.info("Found %d parquet files", len(all_parquets))


query_df, grids_df, hex_grid = load_grids(SHORELINES)
MIN_DIST = 5.0
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
logger.info("Registered DuckDB view 'samples_all'")


query = """
SELECT
    grid_id,
    COUNT(DISTINCT (skysat_id)) AS sample_count
FROM
    samples_all
GROUP BY
    grid_id
"""

df = con.execute(query).fetchdf()
df.grid_id = df.grid_id.map(int)
df = df.set_index("grid_id")
hex_df = grids_df[["hex_id", "dist_km", "geometry"]].join(df, how="left")
gdf = gpd.GeoDataFrame(hex_df, geometry="geometry", crs=hex_grid.crs)

logger.info("Saving results to ShapeFile")
(FIG_DIR / "grid_data").mkdir(exist_ok=True)
gdf.to_file(FIG_DIR / "grid_data" / "data.shp")

logger.info("Plotting Counts")
agg = hex_df.groupby("hex_id").agg(
    median_sample_count=("sample_count", "median"),
    max_sample_count=("sample_count", "max"),
    sum_sample_count=("sample_count", "sum"),
)
agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=hex_grid.crs)

plot_gdf_column(
    gdf,
    "sum_sample_count",
    title="SkySat/Dove Intersection Counts",
    save_path=FIG_DIR / "sum_sample_count.png",
    scale="log",
    use_cbar_label=False,
    vmax=2200,
)

logger.info("Saving results to ShapeFile")
(FIG_DIR / "hex_data").mkdir(exist_ok=True)
gdf.to_file(FIG_DIR / "hex_data" / "data.shp")

print("TOTAL SKYSAT/DOVE INTERSECTIONS")
print(int(hex_df.sample_count.sum()))

logger.info("Done")
