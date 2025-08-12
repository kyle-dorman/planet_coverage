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


BASE = Path("/Users/kyledorman/data/planet_coverage/points_30km/")  # <-- update this
SHORELINES = BASE.parent / "shorelines"
FIG_DIR = BASE.parent / "figs_v2" / "ocean"
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Example path patterns
f_pattern = "dove/results/*/*/*/*/data.parquet"
all_files_pattern = str(BASE / f_pattern)

# Combined list used later when we search individual files
all_parquets = list(BASE.glob(f_pattern))

if not all_parquets:
    logger.error("No parquet files found matching pattern %s", all_files_pattern)
    raise FileNotFoundError("No parquet files found")
logger.info("Found %d parquet files", len(all_parquets))


query_df, _, hex_grid = load_grids(SHORELINES)
ignore_region = gpd.read_file(SHORELINES / "invalid_region.geojson")
# Filter out the North East Russia area.
query_df = query_df[~query_df.to_crs(ignore_region.crs).within(ignore_region.geometry.iloc[0])]  # type: ignore
assert query_df.crs is not None
inter = gpd.sjoin(
    hex_grid.to_crs(query_df.crs).reset_index()[["geometry", "hex_id"]], query_df.reset_index()[["geometry", "cell_id"]]
)
counts = inter[["hex_id", "cell_id"]].groupby("hex_id").count()
hex_grid["grid_count"] = 0.0
hex_grid.loc[counts.index, "grid_count"] = counts.cell_id

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
WITH daily_counts AS (
    -- one count per satellite_id per day
    SELECT
        cell_id,
        DATE_TRUNC('day', acquired)             AS sample_day,
        approx_count_distinct(satellite_id)     AS daily_count
    FROM samples_all
    WHERE
        acquired >= TIMESTAMP '2023-07-19'
        AND has_8_channel
        AND item_type        = 'PSScene'
        AND clear_percent    > 75.0
    GROUP BY cell_id, sample_day
)

SELECT
    cell_id,
    SUM(daily_count) AS sample_count
FROM daily_counts
GROUP BY cell_id
ORDER BY cell_id;
"""

df = con.execute(query).fetchdf().set_index("cell_id")
query_data_df = query_df.join(df, how="left").fillna(0.0)

logger.info("Query finished")

logger.info("Plotting Counts")
agg = query_data_df.groupby("hex_id").agg(
    max_sample_count=("sample_count", "max"),
    median_sample_count=("sample_count", "median"),
    sum_sample_count=("sample_count", "sum"),
)
agg = agg[agg.index >= 0].join(hex_grid[["geometry", "grid_count"]])
gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=hex_grid.crs)
for key in ["max_sample_count", "median_sample_count", "sum_sample_count"]:
    gdf.loc[gdf.max_sample_count < 0.5, key] = np.nan

logger.info("Saving results to ShapeFile")
(FIG_DIR / "hex_data").mkdir(exist_ok=True)
gdf.to_file(FIG_DIR / "hex_data" / "data.shp")


gdf = gdf[~gdf.max_sample_count.isna()]
plot_gdf_column(
    gdf=gdf,
    column="max_sample_count",
    title="Open Ocean Sample Count (Since 7/19/2023)",
    save_path=FIG_DIR / "max_sample_count.png",
    # scale="log",
    use_cbar_label=False,
    # vmax=5000,
)

(FIG_DIR / "query_data").mkdir(exist_ok=True)
gdf = gpd.GeoDataFrame(query_data_df, geometry="geometry", crs=query_df.crs)
gdf.to_file(FIG_DIR / "query_data" / "data.shp")

logger.info("Done")
