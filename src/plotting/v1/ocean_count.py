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
FIG_DIR = BASE.parent / "figs" / BASE.name / "ocean"
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Example path patterns
f_pattern = "*/results/*/*/*/*/data.parquet"
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


# query = """
#     SELECT
#         cell_id,
#         COUNT(*) AS sample_count,
#     FROM samples_all
#     WHERE
#         item_type = 'PSScene'
#         AND acquired >= '2023-07-19'
#         AND has_8_channel
#     GROUP BY cell_id
# """

query = """
WITH daily_counts AS (
    SELECT
        cell_id,
        DATE_TRUNC('day', acquired)              AS sample_day,
        -- one count per satellite_id per day
        COUNT(DISTINCT satellite_id)             AS daily_count
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
hex_df = query_df[["hex_id"]].join(df, how="inner")

logger.info("Query finished")

logger.info("Plotting Counts")
agg = (
    hex_df.dropna(subset=["sample_count"])
    .groupby("hex_id")
    .agg(
        median_sample_count=("sample_count", "median"),
    )
)
agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=hex_grid.crs)

vmax = gdf.median_sample_count.max()

plot_gdf_column(
    gdf,
    "median_sample_count",
    title="Open Ocean Sample Count (Since 7/19/2023)",
    show_land_ocean=True,
    save_path=FIG_DIR / "median_sample_count.png",
    scale="log",
)


query = """
WITH daily_counts AS (
    SELECT
        cell_id,
        DATE_TRUNC('day', acquired)              AS sample_day,
        -- one count per satellite_id per day
        COUNT(DISTINCT satellite_id)             AS daily_count
    FROM samples_all
    WHERE
        acquired >= TIMESTAMP '2023-07-19'
        AND has_8_channel
        AND item_type        = 'PSScene'
        AND publishing_stage = 'finalized'
        AND quality_category = 'standard'
        AND clear_percent    > 75.0
        AND has_sr_asset
        AND ground_control
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
hex_df = query_df[["hex_id"]].join(df, how="inner")

logger.info("Query finished")

logger.info("Plotting Counts")
agg = (
    hex_df.dropna(subset=["sample_count"])
    .groupby("hex_id")
    .agg(
        median_sample_count=("sample_count", "median"),
    )
)
agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=hex_grid.crs)

plot_gdf_column(
    gdf,
    "median_sample_count",
    title="Open Ocean Valid Sample Count (Since 7/19/2023)",
    show_land_ocean=True,
    save_path=FIG_DIR / "median_sample_count_valid.png",
    scale="log",
    vmax=vmax,
)


# _____________________________________________________


query = """
WITH daily_counts AS (
    SELECT
        cell_id,
        DATE_TRUNC('day', acquired)              AS sample_day,
        -- one count per satellite_id per day
        COUNT(DISTINCT satellite_id)             AS daily_count
    FROM samples_all
    WHERE
        item_type        = 'PSScene'
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
hex_df = query_df[["hex_id"]].join(df, how="inner")

logger.info("Query finished")

logger.info("Plotting Counts")
agg = (
    hex_df.dropna(subset=["sample_count"])
    .groupby("hex_id")
    .agg(
        median_sample_count=("sample_count", "median"),
    )
)
agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=hex_grid.crs)

vmax = gdf.median_sample_count.max()

plot_gdf_column(
    gdf,
    "median_sample_count",
    title="Open Ocean Sample Count (Full Record)",
    show_land_ocean=True,
    save_path=FIG_DIR / "median_sample_count_all_time.png",
    scale="log",
)


query = """
WITH daily_counts AS (
    SELECT
        cell_id,
        DATE_TRUNC('day', acquired)              AS sample_day,
        -- one count per satellite_id per day
        COUNT(DISTINCT satellite_id)             AS daily_count
    FROM samples_all
    WHERE
        item_type        = 'PSScene'
        AND publishing_stage = 'finalized'
        AND quality_category = 'standard'
        AND clear_percent    > 75.0
        AND has_sr_asset
        AND ground_control
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
hex_df = query_df[["hex_id"]].join(df, how="inner")

logger.info("Query finished")

logger.info("Plotting Counts")
agg = (
    hex_df.dropna(subset=["sample_count"])
    .groupby("hex_id")
    .agg(
        median_sample_count=("sample_count", "median"),
    )
)
agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=hex_grid.crs)

plot_gdf_column(
    gdf,
    "median_sample_count",
    title="Open Ocean Valid Sample Count (Full Record)",
    show_land_ocean=True,
    save_path=FIG_DIR / "median_sample_count_valid_all_time.png",
    scale="log",
    vmax=vmax,
)
