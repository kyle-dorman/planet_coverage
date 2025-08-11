import logging
import warnings
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import NullLocator

from src.plotting.util import load_grids, plot_gdf_column

warnings.filterwarnings("ignore")  # hide every warning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


BASE = Path("/Users/kyledorman/data/planet_coverage/points_30km/")  # <-- update this
FIG_DIR = BASE.parent / "figs" / BASE.name / "skysat_dove"
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


query_df, grids_df, hex_grid = load_grids(BASE)
MIN_DIST = 5.0
valid = ~grids_df.is_land & ~grids_df.dist_km.isna() & (grids_df.dist_km < MIN_DIST)
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
hex_df = grids_df[["hex_id"]].join(df, how="left").fillna({"sample_count": 0})

# --- log-scale histogram with human-readable ticks --------------------
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sample_counts = hex_df["sample_count"].values
# log scale needs positives
sample_counts = sample_counts[sample_counts > 0]  # type: ignore

# Bins: 1, 2, 5 per decade (≈3 bins per power of 10)
if sample_counts.size:
    max_exp = int(np.ceil(np.log10(sample_counts.max())))
else:
    max_exp = 1
bins = np.floor(np.logspace(0, max_exp, num=max_exp * 3 + 1, base=10.0)).astype(np.int32)

ax.hist(sample_counts, bins=bins, alpha=0.75, edgecolor="black")  # type: ignore
ax.set_xscale("log")
ax.set_xticks(bins)
ax.set_xticklabels(
    [f"{int(b):,}" if b >= 1000 else f"{b:g}" for b in bins],
    rotation=45,
    ha="right",
)
ax.xaxis.set_minor_locator(NullLocator())
ax.set_title("SkySat–Dove Pair Histogram (All)")
ax.set_xlabel("Distinct pairs per grid cell")
ax.set_ylabel("Number of grid cells")
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

plt.tight_layout()
plt.savefig(FIG_DIR / "hist_sample_count.png", dpi=200)

logger.info("Query finished")

logger.info("Plotting Counts")
agg = (
    hex_df.dropna(subset=["sample_count"])
    .groupby("hex_id")
    .agg(
        median_sample_count=("sample_count", "median"),
        max_sample_count=("sample_count", "max"),
    )
)
agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=hex_grid.crs)

median_gdf = gdf[gdf.median_sample_count > 0]
plot_gdf_column(
    median_gdf,
    "median_sample_count",
    title="SkySat/Dove Intersection Counts (Agg: Median)",
    show_land_ocean=True,
    save_path=FIG_DIR / "median_sample_count.png",
    scale="log",
)

max_gdf = gdf[gdf.max_sample_count > 0]
plot_gdf_column(
    max_gdf,
    "max_sample_count",
    title="SkySat/Dove Intersection Counts (Agg: Max)",
    show_land_ocean=True,
    save_path=FIG_DIR / "max_sample_count.png",
    scale="log",
)

# ------------------ Filtered -----------------------

query = """
SELECT
    grid_id,
    COUNT(DISTINCT (dove_id, skysat_id)) AS sample_count
FROM
    samples_all
WHERE
    tide_height_delta < 0.1
GROUP BY
    grid_id
"""

df = con.execute(query).fetchdf()
df.grid_id = df.grid_id.map(int)
df = df.set_index("grid_id")
hex_df = grids_df[["hex_id"]].join(df, how="left").fillna({"sample_count": 0})

logger.info("Query finished")

logger.info("Plotting Counts")

# --- log-scale histogram with human-readable ticks --------------------
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sample_counts = hex_df["sample_count"].values
# log scale needs positives
sample_counts = sample_counts[sample_counts > 0]  # type: ignore

if sample_counts.size:
    max_exp = int(np.ceil(np.log10(sample_counts.max())))
else:
    max_exp = 1
bins = np.floor(np.logspace(0, max_exp, num=max_exp * 3 + 1, base=10.0)).astype(np.int32)

ax.hist(sample_counts, bins=bins, alpha=0.75, edgecolor="black")  # type: ignore
ax.set_xscale("log")
ax.set_xticks(bins)
ax.set_xticklabels(
    [f"{int(b):,}" if b >= 1000 else f"{b:g}" for b in bins],
    rotation=45,
    ha="right",
)
ax.xaxis.set_minor_locator(NullLocator())
ax.set_title("SkySat–Dove Pair Histogram (Filtered, Δtide < 0.1 m)")
ax.set_xlabel("Distinct pairs per grid cell")
ax.set_ylabel("Number of grid cells")
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

plt.tight_layout()
plt.savefig(FIG_DIR / "hist_sample_count_filtered.png", dpi=200)

agg = (
    hex_df.dropna(subset=["sample_count"])
    .groupby("hex_id")
    .agg(
        median_sample_count=("sample_count", "median"),
        max_sample_count=("sample_count", "max"),
    )
)
agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=hex_grid.crs)

median_gdf = gdf[gdf.median_sample_count > 0]
plot_gdf_column(
    median_gdf,
    "median_sample_count",
    title="SkySat/Dove Intersection Counts (Agg: Median, Filtered)",
    show_land_ocean=True,
    save_path=FIG_DIR / "median_sample_count_filtered.png",
    scale="log",
)

max_gdf = gdf[gdf.max_sample_count > 0]
plot_gdf_column(
    max_gdf,
    "max_sample_count",
    title="SkySat/Dove Intersection Counts (Agg: Max, Filtered)",
    show_land_ocean=True,
    save_path=FIG_DIR / "max_sample_count_filtered.png",
    scale="log",
)
