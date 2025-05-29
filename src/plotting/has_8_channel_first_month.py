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
FIG_DIR = BASE.parent / "figs" / BASE.name / "8_channel"
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
hex_df = grids_df.join(df, how="left")

logger.info("Queried 4/8 channel samples")

hex_df["has_4_channel_no_8"] = (hex_df.count_4_channel > 0) & (hex_df.count_8_channel == 0)
hex_df["has_8_channel_no_4"] = (hex_df.count_8_channel > 0) & (hex_df.count_4_channel == 0)

agg = hex_df.groupby("hex_id").agg(
    has_4_channel_no_8=("has_4_channel_no_8", "any"),
    has_8_channel_no_4=("has_8_channel_no_4", "any"),
)
agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
gdf = gpd.GeoDataFrame(agg, geometry="geometry")

plot_gdf_column(
    gdf,
    "has_4_channel_no_8",
    title="Has 4 channel but no 8 channel",
    show_coastlines=True,
    save_path=FIG_DIR / "has_4_channel_no_8.png",
)

plot_gdf_column(
    gdf,
    "has_8_channel_no_4",
    title="Has 8 channel but no 4 channel",
    show_coastlines=True,
    save_path=FIG_DIR / "has_8_channel_no_4.png",
)

query = """
-- adjust today() if you want a fixed cutoff rather than “now”
WITH bounds_raw AS (  --------------------------------------------------------------------
    /* 1️⃣  compute candidate boundary months per grid_id */
    SELECT
        grid_id,

        -- first month containing ≥1 8-channel sample
        MIN(DATE_TRUNC('month', acquired))
            FILTER (WHERE has_8_channel)                 AS first_8_month,

        -- last month containing ≥1 4-channel sample
        MAX(DATE_TRUNC('month', acquired))
            FILTER (WHERE NOT has_8_channel)             AS last_4_month,

        -- first + last month of *any* sample (after filter)
        MIN(DATE_TRUNC('month', acquired))               AS first_any_month,
        MAX(DATE_TRUNC('month', acquired))               AS last_any_month
    FROM samples_all
    WHERE
          item_type        = 'PSScene'
      AND coverage_pct     > 0.5
    GROUP BY grid_id
),

bounds AS (        ----------------------------------------------------------------------
    /* 2️⃣  finalise start / end, ensuring NON-NULL for every grid_id */
    SELECT
        grid_id,

        /* If a grid never had 8-channel, use CURRENT month (“still waiting”) */
        COALESCE(first_8_month,
                 DATE_TRUNC('month', CURRENT_DATE))      AS start_month,

        /* If a grid never had 4-channel after it started 8-channel,
           fall back to the grid's own last data month                */
        COALESCE(last_4_month, last_any_month)           AS end_month,

        /* flags for later logic */
        first_8_month IS NULL                            AS only_4_flag
    FROM bounds_raw
),

monthly AS (       ----------------------------------------------------------------------
    /* 3️⃣  regular month-by-month counts within each grid’s window */
    SELECT
        s.grid_id,
        DATE_TRUNC('month', s.acquired)                  AS month_start,
        COUNT(*)                                         AS sample_count,
        COUNT_IF(has_8_channel)                          AS count_8_channel
    FROM samples_all AS s
    JOIN bounds      AS b
      ON s.grid_id = b.grid_id
    WHERE
          s.item_type        = 'PSScene'
      AND s.coverage_pct     > 0.5
      AND DATE_TRUNC('month', s.acquired)
          BETWEEN b.start_month AND b.end_month
    GROUP BY s.grid_id, month_start
),

waiting AS (       ----------------------------------------------------------------------
    /* 4️⃣  placeholder row for “only-4-channel so far” grids */
    SELECT
        grid_id,
        start_month          AS month_start,   -- this is CURRENT month
        0                    AS sample_count,
        0                    AS count_8_channel
    FROM bounds
    WHERE only_4_flag
)

-- ---------------------------------------------------------------------------
-- 5️⃣  final output  (real months + waiting rows)
-- ---------------------------------------------------------------------------
SELECT *
FROM monthly
UNION ALL
SELECT *
FROM waiting
ORDER BY grid_id, month_start;
"""

df = con.execute(query).fetchdf().set_index("grid_id")
df["pct_8_channel"] = df.count_8_channel / df.sample_count

logger.info("Queried monthly 8 channel samples")

first_month_8_channel = (
    df[df.pct_8_channel > 0.5].reset_index().drop_duplicates(subset=["grid_id"]).set_index("grid_id")
)

# Tried to include grids with channel but it distorts the plotting.
# no_months_8_channel = df[df.sample_count == 0]

# merged_df = pd.concat(
#     [
#         first_month_8_channel[["month_start"]],
#         no_months_8_channel[["month_start"]],
#     ],
#     ignore_index=False,
# )

grid_first_month_8_channel = grids_df.join(first_month_8_channel, how="left").dropna()

# plot_gdf_column(grid_first_month_8_channel, "month_start", title="First Month with more than 50% 8 channel", show_coastlines=True)

agg = grid_first_month_8_channel.groupby("hex_id").agg(month_start=("month_start", "median"))

agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
gdf = gpd.GeoDataFrame(agg, geometry="geometry")

plot_gdf_column(
    gdf,
    "month_start",
    title="First Month with more than 50% 8 channel",
    show_coastlines=True,
    save_path=FIG_DIR / "first_month_with_half_8_channel.png",
)
