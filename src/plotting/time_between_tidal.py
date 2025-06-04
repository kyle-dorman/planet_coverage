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
FIG_DIR = BASE.parent / "figs" / BASE.name / "tide"
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
# Filter out grids where there is no tidal range
query_df = query_df[~query_df["tide_range"].isna()]

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


processed = []
nbins = 10
# Use human readable log scale edges
bin_edges = np.ceil(np.logspace(np.log10(1.0), np.log10(366.0), 10 + 1)).astype(np.int32)
bin_widths = np.diff(bin_edges)
bin_left = bin_edges[:-1]  # left edge of each bar

start_year = 2023
end_year = 2024
pct = 95

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
            hex_df = grids_df[["hex_id"]].join(df, how="left").fillna({f"p{pct}_days_between": 365})

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

            title = f"Max p{pct} Time Between Samples {tide.capitalize()} Tide {disp_year} ({valid_str.capitalize()})"
            plot_gdf_column(
                gdf,
                "max_days_between",
                title=title,
                show_coastlines=True,
                save_path=FIG_DIR / f"max_p{pct}_time_between_samples_{tide}_{valid_str}_{disp_year}.png",
                vmax=365,
                vmin=1,
                scale="log",
                show=False,
            )
            title = f"Min p{pct} Time Between Samples {tide.capitalize()} Tide {disp_year} ({valid_str.capitalize()})"
            plot_gdf_column(
                gdf,
                "min_days_between",
                title=title,
                show_coastlines=True,
                save_path=FIG_DIR / f"min_p{pct}_time_between_samples_{tide}_{valid_str}_{disp_year}.png",
                vmax=365,
                vmin=1,
                scale="log",
                show=False,
            )

            title = (
                f"Median p{pct} Time Between Samples {tide.capitalize()} Tide {disp_year} ({valid_str.capitalize()})"
            )
            plot_gdf_column(
                gdf,
                "median_days_between",
                title=title,
                show_coastlines=True,
                save_path=FIG_DIR / f"median_p{pct}_time_between_samples_{tide}_{valid_str}_{disp_year}.png",
                vmax=365,
                vmin=1,
                scale="log",
                show=False,
            )

df = pd.DataFrame(processed)
for valid_flag, group in df.groupby("valid"):
    valid_str = "valid" if valid_flag else "all"
    # sort rows by tide so the sub-plots are in order
    group = group.sort_values("tide")

    # create a 3 × 1 grid of axes (2 panels)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6, 6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = axes.ravel()

    # loop over the ten rows in this validity group
    for ax, (_, row) in zip(axes, group.iterrows()):
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
        ax.set_title(f"{tide.capitalize()} Tide {valid_str.capitalize()} {row['year']}")

        # thin tick labels for space
        ax.tick_params(axis="both", labelsize=8)

    # add a single y-label on the figure border
    fig.supylabel("Frequency (grid cells)", fontsize=12)
    fig.supxlabel(f"p{pct} Days Between Samples", fontsize=12)

    # overall figure title
    fig.suptitle(
        f"Histogram of p{pct} Days Between Tidal Samples per Grid – " f"{valid_str.capitalize()}", fontsize=14, y=1.04
    )

    plt.savefig(FIG_DIR / f"histogram_p{pct}_time_between_samples_by_year_{valid_str}.png")

    # Cumulative sum plot
    fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True, constrained_layout=True)
    axes = axes.ravel()

    for ax, (_, row) in zip(axes, group.iterrows()):
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
        ax.set_title(f"{tide.capitalize()} Tide {valid_str.capitalize()} {row['year']}")
        ax.tick_params(axis="both", labelsize=8)

    fig.supylabel("Cumulative Grid Cell Count", fontsize=12)
    fig.supxlabel(f"p{pct} Days Between Samples", fontsize=12)
    fig.suptitle(
        f"Cumulative Distribution of p{pct} Days Between Tidal Samples per Grid – {valid_str.capitalize()}",
        fontsize=14,
        y=1.04,
    )

    plt.savefig(FIG_DIR / f"cumsum_p{pct}_time_between_samples_by_year_{valid_str}.png")

    # plt.show()
