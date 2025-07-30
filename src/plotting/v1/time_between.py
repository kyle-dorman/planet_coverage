import logging
import warnings
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import NullLocator
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
FIG_DIR = BASE.parent / "figs" / BASE.name / "time_between"
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
MIN_DIST = 20.0
valid = ~grids_df.is_land & (grids_df.dist_km.isna() | (grids_df.dist_km < MIN_DIST))
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
    GROUP BY grid_id
"""

valid_grid_ids = con.execute(valid_grids_query).fetchdf().grid_id.tolist()


nbins = 10
# Use human readable log scale edges
bin_edges = np.ceil(np.logspace(np.log10(1.0), np.log10(366.0), nbins + 1)).astype(np.int32)
bin_widths = np.diff(bin_edges)
bin_left = bin_edges[:-1]  # left edge of each bar

start_year = 2015
end_year = 2025

for pct in [50, 95]:
    processed = []
    ribbon_plots = []
    for year in tqdm(range(start_year, end_year), total=end_year - start_year):
        disp_year = year + 1

        for valid in [True, False]:
            query = make_time_between_query(year, pct, valid)

            valid_str = "valid" if valid else "all"

            df = con.execute(query).fetchdf().set_index("grid_id")
            hex_df = grids_df[["hex_id", "dist_km"]].join(df, how="left").fillna({f"p{pct}_days_between": 365})

            if valid:
                hex_df = hex_df.loc[hex_df.index.intersection(valid_grid_ids)]
            else:
                hex_df = hex_df.loc[hex_df.index.intersection(all_grid_ids)]

            plt_df = hex_df[~hex_df.dist_km.isna()]
            hist, _ = np.histogram(plt_df[f"p{pct}_days_between"], bin_edges)
            row = {f"count_{i}": hist[i] for i in range(nbins)}
            row["year"] = disp_year
            row["valid"] = valid
            processed.append(row)

            agg = hex_df.groupby("hex_id").agg(
                median_days_between=(f"p{pct}_days_between", "median"),
                max_days_between=(f"p{pct}_days_between", "max"),
                min_days_between=(f"p{pct}_days_between", "min"),
            )
            agg = agg[agg.index >= 0].join(hex_grid[["geometry"]])
            gdf = gpd.GeoDataFrame(agg, geometry="geometry")

            plot_gdf_column(
                gdf,
                "median_days_between",
                title=f"p{pct} Time Between Samples (Year: {year}, Valid: {valid_str.capitalize()}, Agg: Median)",
                show_land_ocean=True,
                save_path=FIG_DIR / f"median_p{pct}_time_between_samples_{valid_str}_{disp_year}.png",
                vmax=365,
                vmin=1,
                scale="log",
            )
            # plot_gdf_column(
            #     gdf,
            #     "max_days_between",
            #     title=f"p{pct} Time Between Samples (Year: {year}, Valid: {valid_str.capitalize()}, Agg: Max)",
            #     show_land_ocean=True,
            #     save_path=FIG_DIR / f"max_p{pct}_time_between_samples_{valid_str}_{disp_year}.png",
            #     vmax=365,
            #     vmin=1,
            #     scale="log",
            # )
            plot_gdf_column(
                gdf,
                "min_days_between",
                title=f"p{pct} Time Between Samples (Year: {year}, Valid: {valid_str.capitalize()}, Agg: Min)",
                show_land_ocean=True,
                save_path=FIG_DIR / f"min_p{pct}_time_between_samples_{valid_str}_{disp_year}.png",
                vmax=365,
                vmin=1,
                scale="log",
            )

            # 1-km bins (adjust width as needed)
            plt_df = hex_df[~hex_df.dist_km.isna()]
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

            ribbon_plots.append(summary)

    df = pd.DataFrame(processed)
    for valid_flag, group in df.groupby("valid"):
        valid_str = "valid" if valid_flag else "all"
        # sort rows by fiscal year so the sub-plots are in order
        group = group.sort_values("year")

        # create a 2 × 5 grid of axes (ten panels)
        fig, axes = plt.subplots(
            5,
            2,
            figsize=(5 * 2, 3 * 5),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = axes.ravel()  # flatten 2-D array → 1-D iterator

        # loop over the ten rows in this validity group
        for ax, (_, row) in zip(axes, group.iterrows()):
            counts = [row[f"count_{i}"] for i in range(nbins)]

            ax.bar(
                bin_left,  # left edge of each bar
                counts,
                width=bin_widths,
                align="edge",
            )
            # Show ticks at all bin edges
            ax.set_xscale("log")
            ax.set_xticklabels(
                [f"{b:g}" for b in bin_edges],
                rotation=45,
                ha="right",
            )
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_title(f"{row['year']}")

            # thin tick labels for space
            ax.tick_params(axis="both", labelsize=8)

        # add a single y-label on the figure border
        fig.supylabel("Frequency (grid cells)", fontsize=12)
        fig.supxlabel("Time Between Samples (days)", fontsize=12)
        fig.suptitle(f"p{pct} Time Between Samples", fontsize=14)

        plt.savefig(FIG_DIR / f"histogram_p{pct}_time_between_samples_by_year_{valid_str}.png")
        plt.close(fig)

        # Cumulative sum plot
        fig, axes = plt.subplots(
            5,
            2,
            figsize=(4 * 2, 2 * 5),
            sharex=True,
            constrained_layout=True,
        )
        axes = axes.ravel()  # flatten 2-D array → 1-D iterator

        for i, (ax, (_, row)) in enumerate(zip(axes, group.iterrows())):
            counts = [row[f"count_{i}"] for i in range(nbins)]
            cumsum = np.cumsum(counts)
            total = cumsum[-1] if cumsum[-1] else 1.0  # prevent divide-by-zero
            cumsum_pct = cumsum / total * 100.0  # → 0-100 %

            ax.plot(bin_left, cumsum_pct, marker="o", linestyle="-")
            ax.set_ylim(0, 100)
            ax.set_xscale("log")
            ax.set_xticks(bin_edges)
            ax.set_xticks(bin_edges)
            ax.set_xticklabels(
                [f"{b:g}" for b in bin_edges],
                rotation=45,
                ha="right",
            )
            ax.xaxis.set_minor_locator(NullLocator())
            # ax.set_xticklabels([f"{edge:.0f}" for edge in bin_edges], rotation=45, ha="right")
            ax.set_title(f"{row['year']}")
            ax.tick_params(axis="both", labelsize=8)
            ax.axhline(50, linestyle="--", linewidth=0.8, label="50 %", color="red")
            ax.axhline(95, linestyle=":", linewidth=0.8, label="95 %", color="green")
            if i == 0:
                ax.legend()

        fig.supylabel("Cumulative Grid Cell %", fontsize=12)
        fig.supxlabel("Time Between Samples (days)", fontsize=12)
        fig.suptitle(
            f"Cumulative Distribution of p{pct} Time Between Samples",
            fontsize=14,
        )

        plt.savefig(FIG_DIR / f"cumsum_p{pct}_time_between_samples_by_year_{valid_str}.png")
        plt.close(fig)

    df = pd.concat(ribbon_plots, ignore_index=True)
    for valid_flag, group in df.groupby("valid"):
        valid_str = "valid" if valid_flag else "all"
        # sort rows by fiscal year so the sub-plots are in order
        group = group.sort_values("year")

        # create a 2 × 5 grid of axes (ten panels)
        fig, axes = plt.subplots(
            5,
            2,
            figsize=(4 * 2, 2 * 5),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = axes.ravel()  # flatten 2-D array → 1-D iterator

        # loop over the ten rows in this validity group
        for ax, (disp_year, summary) in zip(axes, group.groupby("year")):
            summary = summary.sort_values("dist_km")
            year = summary.year.iloc[0]
            ax.fill_between(summary["centre_km"], summary["q25"], summary["q75"], alpha=0.3, label="25-75 % IQR")
            ax.plot(summary["centre_km"], summary["median"], marker="o", ms=3, label="Median")
            ax.set_yscale("log")
            ax.set_yticks(bin_edges)
            ax.set_yticklabels([f"{edge:.0f}" for edge in bin_edges])
            ax.grid(True, which="both", linestyle=":", linewidth=0.3)
            ax.set_title(str(year))
            ax.legend()

        fig.supylabel("Time Between Samples (days)", fontsize=10)
        fig.supxlabel("Distance from Shore (km)", fontsize=10)
        fig.suptitle(
            f"Distance vs. p{pct} Time Between Samples",
            fontsize=12,
        )
        plt.savefig(FIG_DIR / f"ribbon_p{pct}_dist_vs_days_between_{valid_str}.png")
        plt.close(fig)
