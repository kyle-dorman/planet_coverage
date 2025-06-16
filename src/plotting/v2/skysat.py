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
SHORELINES = BASE.parent / "shorelines"
FIG_DIR = BASE.parent / "figs_v2" / "skysat_dove"
FIG_DIR.mkdir(exist_ok=True, parents=True)

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
MIN_DIST = 5.0
valid = ~grids_df.is_land & (grids_df.dist_km.isna() | (grids_df.dist_km < MIN_DIST)) & ~grids_df.tide_range.isna()
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
    COUNT(DISTINCT (dove_id, skysat_id)) AS sample_count
FROM
    samples_all
GROUP BY
    grid_id
"""

df = con.execute(query).fetchdf()
df.grid_id = df.grid_id.map(int)
df = df.set_index("grid_id")
hex_df = grids_df[["hex_id", "dist_km"]].join(df, how="left")
# Filter Antartica NaN rows
to_remove = hex_df.sample_count.isna() & hex_df.dist_km.isna()
hex_df = hex_df[~to_remove].copy()

logger.info("Plotting Counts")
agg = hex_df.groupby("hex_id").agg(
    median_sample_count=("sample_count", "median"),
    max_sample_count=("sample_count", "max"),
)
agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=hex_grid.crs)

plot_gdf_column(
    gdf,
    "max_sample_count",
    title="SkySat/Dove Intersection Counts",
    show_land_ocean=True,
    save_path=FIG_DIR / "max_sample_count.png",
    scale="log",
    use_cbar_label=False,
)

logger.info("Saving results to ShapeFile")
(FIG_DIR / "data").mkdir(exist_ok=True)
gdf.to_file(FIG_DIR / "data" / "data.shp")
