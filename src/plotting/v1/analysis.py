import logging
import pdb
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
FIG_DIR = BASE.parent / "figs" / BASE.name
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Example path patterns
f_pattern = "*/coastal_results/*/*/*/coastal_points.parquet"
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

query = """
SELECT
    grid_id,
    COUNT_IF(has_8_channel) AS count_8_channel,
    COUNT_IF(NOT has_8_channel) AS count_4_channel,
    COUNT(*) AS sample_count,
FROM samples_all
WHERE item_type = 'PSScene'
GROUP BY grid_id
"""


df = con.execute(query).fetchdf().set_index("grid_id")


pdb.set_trace()

assert False

query = """
-- one row per grid_id × calendar-month
SELECT
    grid_id,
    /* month_start = first day of the month, keeps it sortable & readable */
    DATE_TRUNC('month', acquired) AS month_start,
    COUNT(*)                       AS sample_count,
    COUNT_IF(has_8_channel)        AS count_8_channel      -- rows where flag = TRUE
FROM samples_all
WHERE
    item_type        = 'PSScene'
    AND coverage_pct > 0.5
GROUP BY grid_id, month_start
ORDER BY grid_id, month_start;
"""

monthly_counts = con.execute(query).fetchdf().set_index("grid_id")
monthly_counts["pct_8_channel"] = monthly_counts.count_8_channel / monthly_counts.sample_count

logger.info("Queried monthly 8 channel samples")

first_month_8_channel = (
    monthly_counts[monthly_counts.pct_8_channel > 0.5]
    .reset_index()
    .drop_duplicates(subset=["grid_id"])
    .set_index("grid_id")
)
grid_first_month_8_channel = grids_df.join(first_month_8_channel[["month_start"]], how="left").dropna(
    subset=["month_start"]
)
# plot_gdf_column(grid_first_month_8_channel, "month_start", title="First Month with more than 50% 8 channel", show_land_ocean=True)

hex_counts = grids_df[["hex_id"]].join(monthly_counts, how="left")

agg = (
    hex_counts.groupby(["hex_id", "month_start"], as_index=False, sort=True)[["count_8_channel", "sample_count"]]
    .sum()  # ← sums within each hex
    .assign(pct_8_channel=lambda d: d["count_8_channel"] / d["sample_count"])  # or * 100 for %
)

agg = agg[agg.index >= 0]
agg = agg[agg.pct_8_channel > 0.5].drop_duplicates(subset=["hex_id"]).set_index("hex_id")
agg = agg.join(hex_grid[["geometry"]])
gdf = gpd.GeoDataFrame(agg, geometry="geometry")

plot_gdf_column(
    gdf,
    "month_start",
    title="First Month with more than 50% 8 channel",
    show_land_ocean=True,
    save_path=FIG_DIR / "first_month_with_half_8_channel.png",
)
