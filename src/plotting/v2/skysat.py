import logging
import warnings
from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

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
lats = grids_df.centroid.y
valid = ~grids_df.is_land & ~grids_df.dist_km.isna() & (grids_df.dist_km < MIN_DIST) & (lats > -81.5) & (lats < 81.5)
grids_df = grids_df[valid].copy()

assert grids_df.crs is not None
inter = gpd.sjoin(
    hex_grid.to_crs(grids_df.crs).reset_index()[["geometry", "hex_id"]], grids_df.reset_index()[["geometry", "grid_id"]]
)
counts = inter[["hex_id", "grid_id"]].groupby("hex_id").count()
hex_ids = counts[counts.grid_id > 10].index
hex_grid = hex_grid.loc[hex_ids]

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


query = """
SELECT
    grid_id,
    approx_count_distinct(skysat_id) AS sample_count
FROM
    samples_all
WHERE
    dove_acquired < '2024-12-01'
    AND dove_acquired > '2015-12-01'
GROUP BY
    grid_id
"""

df = con.execute(query).fetchdf()
df.grid_id = df.grid_id.map(int)

print("MAX SKYSAT/DOVE INTERSECTIONS GRID CELL")
print(int(df.sample_count.max()))

df = df.set_index("grid_id")
hex_df = grids_df[["hex_id", "geometry"]].join(df, how="left").fillna(0.0)
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

print("MAX SKYSAT/DOVE INTERSECTIONS HEX CELL")
print(gdf[gdf.sum_sample_count == gdf.sum_sample_count.max()])
print(gdf[gdf.sum_sample_count == gdf.sum_sample_count.max()].geometry.centroid)

plot_gdf_column(
    gdf,
    "sum_sample_count",
    title="SkySat/Dove Intersection Counts",
    save_path=FIG_DIR / "sum_sample_count.png",
    scale="log",
    use_cbar_label=False,
)

logger.info("Saving results to ShapeFile")
(FIG_DIR / "hex_data").mkdir(exist_ok=True)
gdf.to_file(FIG_DIR / "hex_data" / "data.shp")

print("TOTAL SKYSAT/DOVE INTERSECTIONS")
print(int(hex_df.sample_count.sum()))


# --- Plot intersections over time by month ---

logger.info("Creating scatter plot of intersections over time by month")

# Fetch date-wise data
time_query = """
SELECT
    DATE_TRUNC('month', dove_acquired) AS month,
    approx_count_distinct(skysat_id) AS intersection_count
FROM
    samples_all
GROUP BY
    month
ORDER BY
    month
"""
time_df = con.execute(time_query).fetchdf()
time_df["month"] = pd.to_datetime(time_df["month"])

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(time_df["month"], time_df["intersection_count"], alpha=0.7)
plt.title("Monthly SkySat/Dove Intersections Over Time")
plt.xlabel("Month")
plt.ylabel("Intersection Count")
plt.grid(True)
plt.tight_layout()
plt.savefig(FIG_DIR / "monthly_intersections_scatter.png", dpi=300)
plt.close()
logger.info("Saved scatter plot to monthly_intersections_scatter.png")

logger.info("Done")
