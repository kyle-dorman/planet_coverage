import logging
import warnings
from pathlib import Path

import duckdb
import pandas as pd

from src.plotting.util import load_grids

warnings.filterwarnings("ignore")  # hide every warning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

logger.info("Computing basic stats on dataset")

BASE = Path("/Users/kyledorman/data/planet_coverage/points_30km/")
SHORELINES = BASE.parent / "shorelines"

query_df, grids_df, hex_grid = load_grids(SHORELINES)
MIN_DIST = 20.0
lats = grids_df.centroid.y
valid = ~grids_df.is_land & ~grids_df.dist_km.isna() & (grids_df.dist_km < MIN_DIST) & (lats > -81.5) & (lats < 81.5)
grids_df = grids_df[valid].copy()
logger.info("Loaded grid dataframes")

# --- Connect to DuckDB ---
con = duckdb.connect()


# ----------------------- Query Grids ----------------------------
def query_grid_stats():
    # path patterns
    f_pattern = "*/results/*/*/*/*/data.parquet"
    all_files_pattern = str(BASE / f_pattern)

    # Combined list used later when we search individual files
    all_parquets = list(BASE.glob(f_pattern))

    if not all_parquets:
        logger.error("No parquet files found matching pattern %s", all_files_pattern)
        raise FileNotFoundError("No parquet files found")
    logger.info("Found %d query parquet files", len(all_parquets))

    # Register a view for all files
    con.execute(
        f"""
        CREATE OR REPLACE VIEW query_view AS
        SELECT * FROM read_parquet('{all_files_pattern}');
    """
    )
    logger.info("Registered DuckDB view 'query_view'")

    query = """
        SELECT
            approx_count_distinct(id)
                FILTER (WHERE NOT has_8_channel)    AS count_4_channel,
            approx_count_distinct(id)
                FILTER (WHERE has_8_channel)        AS count_8_channel,
            MIN(acquired)
                FILTER (WHERE has_8_channel)        AS first_8_date,
            MAX(acquired)
                FILTER (WHERE NOT has_8_channel)    AS last_4_date,
            MIN(acquired)                           AS first_sample_date,
            MAX(acquired)                           AS last_sample_date,
            approx_count_distinct(id)               AS sample_count,
        FROM query_view
        WHERE
            item_type = 'PSScene'
            AND acquired < '2024-12-01'
    """
    df = con.execute(query).fetchdf()

    print("FIRST SAMPLE DATE")
    print(df.first_sample_date.iloc[0])
    print("")

    print("LAST SAMPLE DATE")
    print(df.last_sample_date.iloc[0])
    print("")

    print("SAMPLE COUNT (Including Ocean)")
    print(df.sample_count.iloc[0])
    print("")

    print("FIRST 8 CHANNEL SAMPLE DATE")
    print(df.first_8_date.iloc[0])
    print("")

    print("LAST 4 CHANNEL SAMPLE DATE")
    print(df.last_4_date.iloc[0])
    print("")

    print("% 4 CHANNEL")
    pct_4_channel = df.count_4_channel.iloc[0] / (df.count_4_channel.iloc[0] + df.count_8_channel.iloc[0])
    print(round(100 * pct_4_channel, 1), "%")
    print("")

    query = """
    SELECT
        DATE_TRUNC('month', acquired)           AS month_start,
        approx_count_distinct(id)
            FILTER (WHERE has_8_channel)        AS count_8_channel,
        approx_count_distinct(id)
            FILTER (WHERE NOT has_8_channel)    AS count_4_channel,
    FROM query_view
    WHERE
        item_type        = 'PSScene'
        AND acquired < '2024-12-01'
    GROUP BY month_start
    ORDER BY month_start
    """

    df = con.execute(query).fetchdf()
    df["sample_count"] = df["count_8_channel"] + df["count_4_channel"]
    df["pct_8_channel"] = df["count_8_channel"] / df["sample_count"]
    first_month_8_channel = df[df.pct_8_channel > 0.5].iloc[0].month_start

    print("FIRST MONTH > 50% 8 CHANNEL")
    print(first_month_8_channel)
    print("")


# ---------------------- Small Grids -------------------------
def coastal_cell_stats():
    # path patterns
    f_pattern = "*/coastal_results/*/*/*/coastal_points.parquet"
    all_files_pattern = str(BASE / f_pattern)

    # Combined list used later when we search individual files
    all_parquets = list(BASE.glob(f_pattern))

    if not all_parquets:
        logger.error("No parquet files found matching pattern %s", all_files_pattern)
        raise FileNotFoundError("No parquet files found")
    logger.info("Found %d small grid parquet files", len(all_parquets))

    # Register a view for all files
    con.execute(
        f"""
        CREATE OR REPLACE VIEW samples_all AS
        SELECT * FROM read_parquet('{all_files_pattern}');
    """
    )
    logger.info("Registered DuckDB view 'samples_all'")

    grid_ids = grids_df.index.to_list()
    grid_tbl = pd.DataFrame({"grid_id": grid_ids})
    con.register("grid_ids_tbl", grid_tbl)  # exposes it as a DuckDB view

    query = """
        SELECT
            approx_count_distinct(id) AS sample_count
        FROM samples_all AS s
        JOIN grid_ids_tbl USING (grid_id)
        WHERE
            s.item_type        = 'PSScene'
            AND s.coverage_pct > 0.5
            AND acquired < '2024-12-01'
    """

    df = con.execute(query).fetchdf()

    print("SAMPLE COUNT GRIDS WITHIN 20 KM OF SHORELINE")
    total_sample_count = df.sample_count.iloc[0]
    print(total_sample_count)
    print("")

    query = """
        SELECT
            approx_count_distinct(id) AS sample_count
        FROM samples_all AS s
        JOIN grid_ids_tbl USING (grid_id)
        WHERE
            s.item_type        = 'PSScene'
            AND s.coverage_pct > 0.5
            AND acquired < '2024-12-01'
            AND s.publishing_stage = 'finalized'
            AND s.quality_category = 'standard'
            AND s.has_sr_asset
            AND s.ground_control
    """

    df = con.execute(query).fetchdf()

    print("VALID SAMPLE COUNT GRIDS WITHIN 20 KM OF SHORELINE")
    valid_sample_count = df.sample_count.iloc[0]
    print(valid_sample_count)
    print(round(100.0 * valid_sample_count / total_sample_count, 1), "%")
    print("")

    query = """
        SELECT
            approx_count_distinct(id) AS sample_count
        FROM samples_all AS s
        JOIN grid_ids_tbl USING (grid_id)
        WHERE
            s.item_type        = 'PSScene'
            AND s.coverage_pct > 0.5
            AND acquired < '2024-12-01'
            AND clear_percent    > 75.0
            AND s.publishing_stage = 'finalized'
            AND s.quality_category = 'standard'
            AND s.has_sr_asset
            AND s.ground_control
    """

    df = con.execute(query).fetchdf()

    print("VALID SAMPLE COUNT (75% clear) GRIDS WITHIN 20 KM OF SHORELINE")
    valid_sample_count = df.sample_count.iloc[0]
    print(valid_sample_count)
    print(round(100.0 * valid_sample_count / total_sample_count, 1), "%")
    print("")


query_grid_stats()
coastal_cell_stats()
logger.info("Done")
