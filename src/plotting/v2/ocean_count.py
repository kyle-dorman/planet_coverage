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
FIG_DIR = BASE.parent / "figs_v2" / "ocean"
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


query_df, grids_df, hex_grid = load_grids(SHORELINES)

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
hex_df = query_df[["hex_id"]].join(df, how="inner")

logger.info("Query finished")

logger.info("Plotting Counts")
agg = (
    hex_df.dropna(subset=["sample_count"])
    .groupby("hex_id")
    .agg(
        max_sample_count=("sample_count", "max"),
    )
)
agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=hex_grid.crs)

plot_gdf_column(
    gdf=gdf,
    column="max_sample_count",
    title="Open Ocean Sample Count (Since 7/19/2023)",
    save_path=FIG_DIR / "max_sample_count.png",
    scale="log",
    use_cbar_label=False,
    vmax=5000,
)

logger.info("Saving results to ShapeFile")
(FIG_DIR / "sample_count").mkdir(exist_ok=True)
gdf.to_file(FIG_DIR / "sample_count" / "data.shp")

logger.info("Done")
