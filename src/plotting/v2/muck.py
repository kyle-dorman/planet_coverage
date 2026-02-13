import logging
import pdb
import warnings
from pathlib import Path

import duckdb
import geopandas as gpd

from src.plotting.util import load_grids

warnings.filterwarnings("ignore")  # hide every warning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


BASE = Path("/Users/kyledorman/data/planet_coverage/points_30km/")
SHORELINES = BASE.parent / "shorelines"

# Example path patterns
f_pattern = "skysat_dove/*/*/*/*/data.parquet"
all_files_pattern = str(BASE / f_pattern)

# Combined list used later when we search individual files
all_parquets = list(BASE.glob(f_pattern))

if not all_parquets:
    logger.error("No parquet files found matching pattern %s", all_files_pattern)
    raise FileNotFoundError("No parquet files found")
logger.info("Found %d parquet files", len(all_parquets))


query_df, grids_df, hex_grid = load_grids(SHORELINES)
valid = grids_df.is_coast
grids_df = grids_df[valid].copy()

logger.info("Loaded grid dataframes")

# --- Connect to DuckDB ---
con = duckdb.connect()

# Register a view for all files
con.execute(f"""
    CREATE OR REPLACE VIEW samples_all AS
    SELECT * FROM read_parquet('{all_files_pattern}');
""")
logger.info("Registered DuckDB view 'samples_all'")


query = """
SELECT
    grid_id,
    COUNT(skysat_id) AS sample_count
FROM
    samples_all
GROUP BY
    grid_id
"""

df = con.execute(query).fetchdf()
df.grid_id = df.grid_id.map(int)

df = df.set_index("grid_id")
hex_df = grids_df[["geometry"]].join(df, how="left").fillna(0.0)
gdf = gpd.GeoDataFrame(hex_df, geometry="geometry", crs=hex_grid.crs)


pdb.set_trace()

print("MAX SKYSAT/DOVE INTERSECTIONS GRID CELL")
print(gdf[gdf.sample_count == gdf.sample_count.max()])
print(gdf[gdf.sample_count == gdf.sample_count.max()].geometry.centroid)

print("% Grids with atleast 1 samples")
print(round(100 * (gdf.sample_count > 0).sum() / len(gdf)))

print("% Grids with atleast 5 samples")
print(round(100 * (gdf.sample_count > 4).sum() / len(gdf)))
