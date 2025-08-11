import logging
import warnings
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.plotting.util import load_grids, make_time_between_query, plot_gdf_column

warnings.filterwarnings("ignore")  # hide every warning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


BASE = Path("/Users/kyledorman/data/planet_coverage/points_30km/")  # <-- update this
FIG_DIR = BASE.parent / "figs" / BASE.name / "tide_time_between"
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


nbins = 10
# Use human readable log scale edges
bin_edges = np.ceil(np.logspace(np.log10(1.0), np.log10(366.0), 10 + 1)).astype(np.int32)
bin_widths = np.diff(bin_edges)
bin_left = bin_edges[:-1]  # left edge of each bar

start_year = 2023
end_year = 2024


def plot_hist(df: pd.DataFrame, valid_flag: bool, pct: float) -> None:
    valid_str = "valid" if valid_flag else "all"
    # sort rows by tide so the sub-plots are in order
    df = df.sort_values("tide")

    # create a 3 × 1 grid of axes (2 panels)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6, 2 * 3 + 1),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = axes.ravel()

    # loop over the rows in this validity group
    for ax, (_, row) in zip(axes, df.iterrows()):
        counts = [row[f"count_{i}"] for i in range(nbins)]

        tide = row.tide

        ax.bar(
            bin_left,  # left edge of each bar
            counts,
            width=bin_widths,
            align="edge",
        )
        # Show ticks at all bin edges
        ax.set_xscale("log")
        ax.set_xticks(bin_edges)
        ax.set_xticklabels([f"{edge:.0f}" for edge in bin_edges], rotation=45, ha="right")
        ax.set_title(f"{tide.capitalize()} Tide - {row['year']}")

        # thin tick labels for space
        ax.tick_params(axis="both", labelsize=8)

    # add a single y-label on the figure border
    fig.supylabel("Frequency (grid cells)", fontsize=10)
    fig.supxlabel(f"p{pct} Days Between Samples", fontsize=10)

    # overall figure title
    fig.suptitle(f"p{pct} Days Between Tidal Samples", fontsize=12)
    plt.savefig(FIG_DIR / f"histogram_p{pct}_time_between_samples_by_year_{valid_str}.png")
    plt.close(fig)


def plot_cumsum(df: pd.DataFrame, valid_flag: bool, pct: float) -> None:
    valid_str = "valid" if valid_flag else "all"
    # sort rows by tide so the sub-plots are in order
    df = df.sort_values("tide")

    # Cumulative sum plot
    fig, axes = plt.subplots(3, 1, figsize=(6, 3 * 3 + 1), sharex=True, constrained_layout=True)
    axes = axes.ravel()

    for i, (ax, (_, row)) in enumerate(zip(axes, df.iterrows())):
        tide = row.tide
        counts = [row[f"count_{i}"] for i in range(nbins)]
        cumsum = np.cumsum(counts)
        total = cumsum[-1] if cumsum[-1] else 1.0  # prevent divide-by-zero
        cumsum_pct = cumsum / total * 100.0  # → 0-100 %

        ax.plot(bin_left, cumsum_pct, marker="o", linestyle="-")
        ax.set_ylim(0, 100)
        ax.set_xscale("log")
        ax.set_xticks(bin_edges)
        ax.set_xticklabels([f"{edge:.0f}" for edge in bin_edges], rotation=45, ha="right")
        ax.set_title(f"{tide.capitalize()} Tide - {row['year']}")
        ax.tick_params(axis="both", labelsize=8)
        # Highlight p50 and p95
        ax.axhline(50, linestyle="--", linewidth=0.8, label="50 %", color="red")
        ax.axhline(95, linestyle=":", linewidth=0.8, label="95 %", color="green")
        if i == 0:
            ax.legend()

    fig.supylabel("Cumulative Grid Cell %", fontsize=10)
    fig.supxlabel("Time Between Samples (days)", fontsize=10)
    fig.suptitle(
        f"Cumulative Distribution of p{pct} Time Between Tidal Samples",
        fontsize=12,
    )
    plt.savefig(FIG_DIR / f"cumsum_p{pct}_time_between_samples_by_year_{valid_str}.png")
    plt.close(fig)


def plot_ribbon(df: pd.DataFrame, valid_flag: bool, pct: float) -> None:
    valid_str = "valid" if valid_flag else "all"

    # sort rows by tide level so the sub-plots are in order
    df = df.sort_values("tide")

    # create a 3 × 1 grid of axes (2 panels)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6, 3 * 3 + 1),
        sharex=True,
        constrained_layout=True,
    )
    axes = axes.ravel()

    # loop over the rows in this validity group
    for i, (ax, (tide, summary)) in enumerate(zip(axes, df.groupby("tide"))):
        tide = str(tide)
        year = summary.year.iloc[0]
        summary = summary.sort_values("dist_km")
        ax.fill_between(summary["centre_km"], summary["q25"], summary["q75"], alpha=0.3, label="25-75 % IQR")
        ax.plot(summary["centre_km"], summary["median"], marker="o", ms=3, label="Median")
        ax.set_yscale("log")
        ax.set_yticks(bin_edges)
        ax.set_yticklabels([f"{edge:.0f}" for edge in bin_edges])
        ax.grid(True, which="both", linestyle=":", linewidth=0.3)
        ax.set_title(f"{tide.capitalize()} Tide - {year}")
        if i == 0:
            ax.legend()

    fig.supylabel("Time Between Samples (days)", fontsize=10)
    fig.supxlabel("Distance from Shore (km)", fontsize=10)
    fig.suptitle(
        f"Distance vs. p{pct} Days Between Tidal Samples per Grid",
        fontsize=10,
    )
    plt.savefig(FIG_DIR / f"ribbon_p{pct}_dist_vs_days_between_{valid_str}.png")
    plt.close(fig)


def run():
    query_df, grids_df, hex_grid = load_grids(BASE)
    MIN_DIST = 20.0
    valid = ~grids_df.is_land & (grids_df.dist_km.isna() | (grids_df.dist_km < MIN_DIST)) & ~grids_df.tide_range.isna()
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

    all_grids_query = """
        SELECT
            grid_id,
            COUNT(*) AS sample_count,
        FROM samples_all
        WHERE
            item_type = 'PSScene'
            AND coverage_pct > 0.5
            AND has_tide_data
            AND acquired         >= TIMESTAMP '2023-12-01'
            AND acquired         <  TIMESTAMP '2024-12-01'
        GROUP BY grid_id
    """

    all_grid_ids = con.execute(all_grids_query).fetchdf().grid_id.tolist()

    valid_grids_query = """
        SELECT
            grid_id,
            COUNT(*) AS sample_count,
        FROM samples_all
        WHERE
            item_type = 'PSScene'
            AND coverage_pct > 0.5
            AND clear_percent > 75.0
            AND publishing_stage = 'finalized'
            AND quality_category = 'standard'
            AND has_sr_asset
            AND ground_control
            AND has_tide_data
            AND acquired         >= TIMESTAMP '2023-12-01'
            AND acquired         <  TIMESTAMP '2024-12-01'
        GROUP BY grid_id
    """

    valid_grid_ids = con.execute(valid_grids_query).fetchdf().grid_id.tolist()

    for pct in [50, 75, 95]:
        processed = []
        ribbon_plots = []
        for year in tqdm(range(start_year, end_year), total=end_year - start_year):
            for tide in ["low", "mid", "high"]:
                if tide == "mid":
                    extra_filter = "AND is_mid_tide AND has_tide_data"
                elif tide == "high":
                    extra_filter = "AND tide_height_bin = 9 AND has_tide_data"
                else:
                    extra_filter = "AND tide_height_bin = 0 AND has_tide_data"

                for valid in [True, False]:
                    query = make_time_between_query(year, pct, valid, extra_filter=extra_filter)

                    valid_str = "valid" if valid else "all"
                    disp_year = year + 1

                    df = con.execute(query).fetchdf().set_index("grid_id")
                    hex_df = grids_df[["hex_id", "dist_km"]].join(df, how="left")
                    if valid:
                        hex_df = hex_df.loc[hex_df.index.intersection(valid_grid_ids)]
                    else:
                        hex_df = hex_df.loc[hex_df.index.intersection(all_grid_ids)]
                    # # Skip Antartica that has no data
                    # keep_rows = ~hex_df[f"p{pct}_days_between"].isna() | ~hex_df.dist_km.isna()
                    # hex_df = hex_df.loc[keep_rows].copy()
                    # hex_df.loc[fill_rows, f"p{pct}_days_between"] = 365.0
                    # Remove remaining NaN rows
                    # hex_df = hex_df.dropna(subset=[f"p{pct}_days_between"])
                    hex_df.fillna({f"p{pct}_days_between": 365.0}, inplace=True)

                    # Filter Antartica
                    plt_df = hex_df[~hex_df.dist_km.isna()]
                    # 1-km bins (adjust width as needed)
                    bins = pd.cut(plt_df["dist_km"], np.arange(0, int(MIN_DIST) + 1, 1))  # type: ignore

                    summary = (
                        plt_df.groupby(bins)[f"p{pct}_days_between"]
                        .agg(
                            q25=lambda s: s.quantile(0.25),
                            median="median",
                            q75=lambda s: s.quantile(0.75),
                            count="count",
                        )
                        .reset_index()
                    )
                    summary["centre_km"] = summary["dist_km"].apply(lambda i: i.left + 0.5)
                    summary["valid"] = valid
                    summary["year"] = disp_year
                    summary["tide"] = tide

                    ribbon_plots.append(summary)

                    hist, _ = np.histogram(hex_df[f"p{pct}_days_between"], bin_edges)
                    row = {f"count_{i}": hist[i] for i in range(nbins)}
                    row["year"] = disp_year
                    row["valid"] = valid
                    row["tide"] = tide
                    processed.append(row)

                    agg = hex_df.groupby("hex_id").agg(
                        median_days_between=(f"p{pct}_days_between", "median"),
                        max_days_between=(f"p{pct}_days_between", "max"),
                        min_days_between=(f"p{pct}_days_between", "min"),
                    )
                    agg = agg[agg.index >= 0].join(hex_grid[["geometry"]], how="inner")
                    gdf = gpd.GeoDataFrame(agg, geometry="geometry")

                    # title = f"p{pct} Days Between {tide.capitalize()} Tide Samples (Year: {disp_year}, Valid: {valid_str.capitalize()}, Agg: Max)"
                    # plot_gdf_column(
                    #     gdf,
                    #     "max_days_between",
                    #     title=title,
                    #     show_land_ocean=True,
                    #     save_path=FIG_DIR / f"max_p{pct}_time_between_samples_{tide}_{valid_str}_{disp_year}.png",
                    #     vmax=365,
                    #     vmin=1,
                    #     scale="log",
                    # )
                    title = f"p{pct} Days Between {
                        tide.capitalize()} Tide Samples (Year: {disp_year}, Valid: {
                        valid_str.capitalize()}, Agg: Min)"
                    plot_gdf_column(
                        gdf,
                        "min_days_between",
                        title=title,
                        show_land_ocean=True,
                        save_path=FIG_DIR / f"min_p{pct}_time_between_samples_{tide}_{valid_str}_{disp_year}.png",
                        vmax=365,
                        vmin=1,
                        scale="log",
                    )
                    title = f"p{pct} Days Between {
                        tide.capitalize()} Tide Samples (Year: {disp_year}, Valid: {
                        valid_str.capitalize()}, Agg: Median)"
                    plot_gdf_column(
                        gdf,
                        "median_days_between",
                        title=title,
                        show_land_ocean=True,
                        save_path=FIG_DIR / f"median_p{pct}_time_between_samples_{tide}_{valid_str}_{disp_year}.png",
                        vmax=365,
                        vmin=1,
                        scale="log",
                    )

        df = pd.DataFrame(processed)
        for valid_flag, group in df.groupby("valid"):
            plot_hist(group, bool(valid_flag), pct)
            plot_cumsum(group, bool(valid_flag), pct)

        df = pd.concat(ribbon_plots, ignore_index=True)
        for valid_flag, group in df.groupby("valid"):
            valid_str = "valid" if valid_flag else "all"
            plot_ribbon(group, bool(valid_flag), pct)


if __name__ == "__main__":
    run()
