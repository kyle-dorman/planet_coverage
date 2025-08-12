import logging
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from src.create_coastal_df import SCHEMA as COASTAL_SCHEMA
from src.plotting.util import load_grids
from src.query_udms import DataFrameRow

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
all_grid_ids = grids_df.index.tolist()
logger.info("Loaded grid dataframes")


# ----------------------- Query Grids ----------------------------
def query_grid_stats():
    logger.info("Creating Query Grid Dove Stats")

    # path patterns
    f_pattern = "dove/results/*/*/*/*/data.parquet"
    all_files_pattern = str(BASE / f_pattern)

    all_lazy = pl.scan_parquet(
        all_files_pattern,
        schema=DataFrameRow.polars_schema(),
    )
    filtered = all_lazy.filter(
        (pl.col("acquired") < datetime(2024, 12, 1)) & (pl.col("acquired") > datetime(2015, 12, 1))
    )
    # aggregate metrics
    df = filtered.select(
        [
            # count of id where has_8_channel is False
            pl.col("id").filter(~pl.col("has_8_channel")).approx_n_unique().alias("count_4_channel"),
            # count of id where has_8_channel is True
            pl.col("id").filter(pl.col("has_8_channel")).approx_n_unique().alias("count_8_channel"),
            # count *distinct* scene IDs (deduplicated)
            pl.col("id").approx_n_unique().alias("sample_count"),
            # earliest acquired for 8-channel
            pl.col("acquired").filter(pl.col("has_8_channel")).min().alias("first_8_date"),
            # latest acquired for 4-channel
            pl.col("acquired").filter(~pl.col("has_8_channel")).max().alias("last_4_date"),
            # overall first/last sample dates
            pl.col("acquired").min().alias("first_sample_date"),
            pl.col("acquired").max().alias("last_sample_date"),
        ]
    ).collect()

    print("FIRST SAMPLE DATE")
    print(df["first_sample_date"][0])
    print("")

    print("LAST SAMPLE DATE")
    print(df["last_sample_date"][0])
    print("")

    print("SAMPLE COUNT (Including Ocean)")
    print(df["sample_count"][0])
    print("")

    print("FIRST 8 CHANNEL SAMPLE DATE")
    print(df["first_8_date"][0])
    print("")

    print("LAST 4 CHANNEL SAMPLE DATE")
    print(df["last_4_date"][0])
    print("")

    print("% 4 CHANNEL")
    pct_4_channel = df["count_4_channel"][0] / (df["count_4_channel"][0] + df["count_8_channel"][0])
    print(round(100 * pct_4_channel, 1), "%")
    print("")

    # ---- Monthly counts (Polars) -------------------------------------------
    df = (
        filtered.with_columns(
            # truncate each acquisition date to the first day of its month
            pl.col("acquired")
            .dt.truncate("1mo")
            .alias("month_start")
        )
        .group_by("month_start")
        .agg(
            # approximate distinct counts per month for 8- and 4-channel scenes
            pl.col("id").filter(pl.col("has_8_channel")).approx_n_unique().alias("count_8_channel"),
            pl.col("id").filter(~pl.col("has_8_channel")).approx_n_unique().alias("count_4_channel"),
        )
        .sort("month_start")
        .collect()
    ).to_pandas()

    df["sample_count"] = df["count_8_channel"] + df["count_4_channel"]
    df["pct_8_channel"] = df["count_8_channel"] / df["sample_count"]
    first_month_8_channel = df[df.pct_8_channel > 0.5].iloc[0].month_start

    print("FIRST MONTH > 50% 8 CHANNEL")
    print(first_month_8_channel)
    print("")


# ---------------------- Small Grids -------------------------
def coastal_cell_stats():
    logger.info("Creating Coastal Dove Grid Stats")

    # path patterns
    f_pattern = "dove/coastal_results/*/*/*/coastal_points.parquet"
    all_files_pattern = str(BASE / f_pattern)

    all_lazy = pl.scan_parquet(
        all_files_pattern,
        schema=COASTAL_SCHEMA,
    )
    filtered = all_lazy.filter(
        (pl.col("acquired") < datetime(2024, 12, 1))
        & (pl.col("acquired") > datetime(2015, 12, 1))
        & (pl.col("coverage_pct") > 0.5)
        & pl.col("grid_id").is_in(all_grid_ids)
    )

    df = filtered.select(
        [
            pl.col("id").approx_n_unique().alias("image_count"),
            pl.col("id")
            .filter(
                (pl.col("publishing_stage") == "finalized")
                & (pl.col("quality_category") == "standard")
                & pl.col("has_sr_asset")
                & pl.col("ground_control")
            )
            .approx_n_unique()
            .alias("valid_image_count"),
            pl.col("id")
            .filter(
                (pl.col("publishing_stage") == "finalized")
                & (pl.col("quality_category") == "standard")
                & pl.col("has_sr_asset")
                & pl.col("ground_control")
                & (pl.col("clear_percent") > 75.0)
            )
            .approx_n_unique()
            .alias("clear_image_count"),
        ]
    ).collect()

    print("IMAGE COUNT GRIDS WITHIN 20 KM OF SHORELINE")
    print(df["image_count"][0])
    print("")
    print("VALID IMAGE COUNT GRIDS WITHIN 20 KM OF SHORELINE")
    print(df["valid_image_count"][0])
    print(round(100.0 * df["valid_image_count"][0] / df["image_count"][0], 1), "%")
    print("")

    print("VALID IMAGE COUNT (75% clear) GRIDS WITHIN 20 KM OF SHORELINE")
    print(df["clear_image_count"][0])
    print(round(100.0 * df["clear_image_count"][0] / df["image_count"][0], 1), "%")
    print("")

    df = filtered.select(
        [
            pl.len().alias("sample_count"),
            pl.col("id")
            .filter(
                (pl.col("publishing_stage") == "finalized")
                & (pl.col("quality_category") == "standard")
                & pl.col("has_sr_asset")
                & pl.col("ground_control")
            )
            .len()
            .alias("valid_count"),
            pl.col("id")
            .filter(
                (pl.col("publishing_stage") == "finalized")
                & (pl.col("quality_category") == "standard")
                & pl.col("has_sr_asset")
                & pl.col("ground_control")
                & (pl.col("clear_percent") > 75.0)
            )
            .len()
            .alias("clear_count"),
        ]
    ).collect()

    print("SAMPLE COUNT GRIDS WITHIN 20 KM OF SHORELINE")
    print(df["sample_count"][0])
    print("")

    print("VALID SAMPLE COUNT GRIDS WITHIN 20 KM OF SHORELINE")
    print(df["valid_count"][0])
    print(round(100.0 * df["valid_count"][0] / df["sample_count"][0], 1), "%")
    print("")

    print("VALID SAMPLE COUNT (75% clear) GRIDS WITHIN 20 KM OF SHORELINE")
    print(df["clear_count"][0])
    print(round(100.0 * df["clear_count"][0] / df["sample_count"][0], 1), "%")
    print("")


def query_skysat_stats():
    logger.info("Creating Query Grid SkySat Stats")
    # path patterns
    f_pattern = "skysat/results/*/*/*/*/data.parquet"
    all_files_pattern = str(BASE / f_pattern)

    all_lazy = pl.scan_parquet(
        all_files_pattern,
        schema=DataFrameRow.polars_schema(),
    )
    filtered = all_lazy.filter(
        (pl.col("acquired") < datetime(2024, 12, 1)) & (pl.col("acquired") > datetime(2015, 12, 1))
    )
    # aggregate metrics
    df = (
        filtered.select(
            [
                # count *distinct* scene IDs (deduplicated)
                pl.col("id").approx_n_unique().alias("sample_count"),
                # overall first/last sample dates
                pl.col("acquired").min().alias("first_sample_date"),
                pl.col("acquired").max().alias("last_sample_date"),
            ]
        )
        .collect()
        .to_pandas()
    )

    print("FIRST SAMPLE DATE")
    print(df.first_sample_date.iloc[0])
    print("")

    print("LAST SAMPLE DATE")
    print(df.last_sample_date.iloc[0])
    print("")

    print("SAMPLE COUNT (Including Ocean)")
    print(df.sample_count.iloc[0])
    print("")

    # ---- Yearly counts per satellite (Polars) ------------------------------
    yearly_df = (
        filtered.with_columns(
            # truncate each acquisition to Jan-01 of its year
            pl.col("acquired")
            .dt.truncate("1y")
            .alias("year")
        )
        .group_by(["satellite_id", "year"])
        .agg(pl.col("id").approx_n_unique().alias("sample_count"))  # dedup by id
        .sort(["year", "satellite_id"])
        .collect()
        .to_pandas()
    )

    logger.info("Creating stacked bar chart of SkySat samples per year per satellite")

    pivot_df = yearly_df.pivot_table(
        index="year", columns="satellite_id", values="sample_count", fill_value=0
    ).sort_index()

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


# query_grid_stats()
coastal_cell_stats()
# query_skysat_stats()
logger.info("Done")
