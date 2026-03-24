import logging
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
f_pattern = "*/coastal_results/*/*/*/coastal_points.parquet"
all_files_pattern = str(BASE / f_pattern)

# Combined list used later when we search individual files
all_parquets = list(BASE.glob(f_pattern))

if not all_parquets:
    logger.error("No parquet files found matching pattern %s", all_files_pattern)
    raise FileNotFoundError("No parquet files found")
logger.info("Found %d parquet files", len(all_parquets))


query_df, grids_df, hex_grid = load_grids(SHORELINES)
MIN_DIST = 3.0
valid = (
    ~grids_df.is_land
    & grids_df.dist_km.notna()
    & ((grids_df.dist_km < MIN_DIST) | grids_df.is_coast)
    & ~grids_df.tide_range.isna()
)
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
    COUNT(id) AS sample_count,
FROM
    samples_all
WHERE
    item_type           = 'PSScene'
    AND coverage_pct    > 0.5
    AND acquired        <  TIMESTAMP '2025-01-01'
    AND acquired        >=  TIMESTAMP '2024-01-01'
    AND publishing_stage = 'finalized'
    AND quality_category = 'standard'
    AND clear_percent    > 75.0
    AND has_sr_asset
    AND ground_control
GROUP BY grid_id
ORDER BY grid_id
"""

df = con.execute(query).fetchdf().set_index("grid_id")

df = grids_df[["dist_km", "is_coast", "geometry"]].join(df, how="left")
df["sample_count"] = df["sample_count"].fillna(0.0)

gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=grids_df.crs)

logger.info("Saving results to ShapeFile")
gdf.to_file("/Users/kyledorman/Desktop/no_midtide.gpkg")

# df.to_csv("/Users/kyledorman/Desktop/midtide.csv")

logger.info("Done")
