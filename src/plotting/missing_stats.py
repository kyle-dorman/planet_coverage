import logging
import warnings
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # hide every warning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

logger.info("Computing basic stats on dataset")

BASE = Path("/Users/kyledorman/data/planet_coverage/count_missing/")
SHORELINES = BASE.parent / "shorelines"

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
            COUNT(*)
                FILTER (WHERE NOT instrument)       AS count_instrument,
            COUNT(*)
                FILTER (WHERE NOT clear_percent)    AS count_clear_percent,
            COUNT(*)
                FILTER (WHERE NOT cloud_cover)      AS count_cloud_cover,
            COUNT(*)
                FILTER (WHERE NOT ground_control)   AS count_ground_control,
            COUNT(*)
                FILTER (WHERE NOT publishing_stage) AS count_publishing_stage,
            COUNT(*)
                FILTER (WHERE NOT assets)           AS count_assets,
            item_type,
            DATE_TRUNC('year', acquired) AS year,
        FROM query_view
        GROUP BY year, item_type
        ORDER BY year, item_type;
    """
    df = con.execute(query).fetchdf()

    for item_type, ddf in df.groupby("item_type"):
        print(item_type)
        print(ddf)

    return df


def plot_stats(df):
    """
    Create a stacked‑bar chart for each item_type showing counts of missing
    attributes per year. Saves PNGs in BASE/plots and shows the plots.
    """
    categories = [
        ("count_instrument", "Instrument"),
        ("count_clear_percent", "Clear Percent"),
        ("count_cloud_cover", "Cloud Cover"),
        ("count_ground_control", "Ground Control"),
        ("count_publishing_stage", "Publishing Stage"),
        ("count_assets", "Assets"),
    ]

    # Ensure 'year' is an int (e.g. 2020) rather than a Timestamp
    df = df.copy()
    df["year"] = pd.to_datetime(df["year"]).dt.year.astype(int)

    out_dir = BASE / "plots"
    out_dir.mkdir(exist_ok=True)

    for item_type, sub_df in df.groupby("item_type"):
        sub_df = sub_df.sort_values("year")
        years = sub_df["year"].tolist()

        bottom = np.zeros(len(years))
        fig, ax = plt.subplots(figsize=(10, 6))

        for col, label in categories:
            values = sub_df[col].values
            ax.bar(years, values, bottom=bottom, label=label)
            bottom += values

        ax.set_title(f"Missing Attribute Counts per Year – {item_type}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()

        save_path = out_dir / f"{item_type}_missing_stats.png"
        fig.savefig(save_path, dpi=300)

    plt.show()


df = query_grid_stats()
plot_stats(df)

logger.info("Done")
