import logging
import warnings
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
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
    f_pattern = "dove/results/*/*/*/*/data.parquet"
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
    f_pattern = "dove/coastal_results/*/*/*/coastal_points.parquet"
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


def query_skysat_stats():
    # path patterns
    f_pattern = "skysat/results/*/*/*/*/data.parquet"
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
            MIN(acquired)               AS first_sample_date,
            MAX(acquired)               AS last_sample_date,
            approx_count_distinct(id)   AS sample_count,
        FROM query_view
        WHERE
            acquired < '2024-12-01'
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

    query = """
        SELECT
            approx_count_distinct(id)       AS sample_count,
            DATE_TRUNC('year', acquired)   AS year,
            satellite_id
        FROM query_view
        GROUP BY
            satellite_id, year
    """
    df = con.execute(query).fetchdf()
    df["year"] = pd.to_datetime(df["year"])

    logger.info("Creating stacked bar chart of SkySat samples per year per satellite")

    pivot_df = df.pivot_table(index="year", columns="satellite_id", values="sample_count", fill_value=0).sort_index()

    plt.figure(figsize=(12, 6))
    pivot_df.plot(kind="bar", stacked=True, colormap="tab20", width=1.0, edgecolor="none", figsize=(12, 6))

    # Format x-axis as plain 4-digit year
    ax = plt.gca()
    ax.set_xticklabels([d.year for d in pivot_df.index])

    plt.title("Yearly SkySat Samples per Satellite")
    plt.xlabel("Year")
    plt.ylabel("Sample Count")
    plt.legend(title="Satellite ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.grid(True, axis="y")
    output_path = BASE / "skysat_yearly_samples_stacked_bar.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved stacked bar chart to {output_path}")


query_grid_stats()
coastal_cell_stats()
query_skysat_stats()
logger.info("Done")
