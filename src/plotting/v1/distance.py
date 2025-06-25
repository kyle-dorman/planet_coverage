import logging
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.plotting.util import load_grids

warnings.filterwarnings("ignore")  # hide every warning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


BASE = Path("/Users/kyledorman/data/planet_coverage/points_30km/")  # <-- update this
FIG_DIR = BASE.parent / "figs" / BASE.name / "distance"
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
MIN_DIST = 35.0
valid = ~grids_df.is_land & ~grids_df.dist_km.isna() & (grids_df.dist_km < MIN_DIST)
grids_df = grids_df[valid]

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
        MIN(acquired) as first_sample,
        COUNT(*) AS sample_count,
    FROM samples_all
    WHERE
        item_type = 'PSScene'
        AND coverage_pct > 0.5
        AND acquired >= TIMESTAMP '2023-12-01'
        AND acquired < TIMESTAMP '2024-12-01'
        AND publishing_stage = 'finalized'
        AND quality_category = 'standard'
        AND clear_percent    > 75.0
        AND has_sr_asset
        AND ground_control
    GROUP BY grid_id
"""

df = con.execute(query).fetchdf().set_index("grid_id")
hex_df = grids_df.join(df, how="left").fillna({"sample_count": 0})

# # ------------------------------------------------------------
# # Scatter: shoreline distance vs sample count
# # ------------------------------------------------------------
# fig, ax = plt.subplots(figsize=(6, 4))
# ax.scatter(
#     hex_df["dist_km"],
#     hex_df["sample_count"],
#     s=1,
#     alpha=0.6,
# )
# ax.set_xlabel("Distance from Shore (km)")
# ax.set_ylabel("Sample Count")
# ax.set_yscale("log")
# ax.set_title("Distance vs Sample Count")
# ax.grid(True, which="both", linestyle=":", linewidth=0.3)

# plt.tight_layout()
# plt.savefig(FIG_DIR / "scatter_dist_from_shore_vs_sample_count.png")
# plt.close(fig)

# ------------------------------------------------------------
# Hexbin: shoreline distance vs sample count
# ------------------------------------------------------------

# fig, ax = plt.subplots(figsize=(6, 4))
# plt_df = hex_df[hex_df.sample_count > 0]
# hb = ax.hexbin(
#     plt_df["dist_km"],
#     plt_df["sample_count"],
#     gridsize=10,  # smaller → larger hexes
#     bins="log",  # colour scale is log-counts
#     mincnt=1,  # hide empty bins,
#     yscale='log',
# )
# ax.set_xlabel("Distance from Shore (km)")
# ax.set_ylabel("Sample Count (log scale)")
# cb = fig.colorbar(hb, ax=ax, label="log₁₀(# points)")
# ax.set_title("Distance vs Sample Count (hex-binned)")
# plt.tight_layout()
# plt.savefig(FIG_DIR / "hexbin_dist_vs_sample_count.png")
# plt.close(fig)

# ------------------------------------------------------------
# ribbon plot: shoreline distance vs sample count
# ------------------------------------------------------------

# 1-km bins (adjust width as needed)
bins = pd.cut(hex_df["dist_km"], np.arange(0, int(MIN_DIST) + 1, 1))  # type: ignore

summary = (
    hex_df.groupby(bins)["sample_count"]
    .agg(
        q25=lambda s: s.quantile(0.25),
        median="median",
        q75=lambda s: s.quantile(0.75),
        count="count",
    )
    .reset_index()
)
summary["centre_km"] = summary["dist_km"].apply(lambda i: i.left + 0.5)

fig, ax = plt.subplots(figsize=(6, 4))
ax.fill_between(summary["centre_km"], summary["q25"], summary["q75"], alpha=0.3, label="25-75 % IQR")
ax.plot(summary["centre_km"], summary["median"], marker="o", ms=3, label="Median")
ax.set_yscale("log")
ax.set_xlabel("Distance from Shore (km)")
ax.set_ylabel("Sample Count")
ax.set_title("Distribution of Sample Count per km bin")
ax.legend()
ax.grid(True, which="both", linestyle=":", linewidth=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "ribbon_dist_vs_sample_count.png")
plt.close(fig)

# # ------------------------------------------------------------
# # box plot: shoreline distance vs sample count
# # ------------------------------------------------------------

# hex_df["dist_bin"] = pd.cut(hex_df["dist_km"], bins=list(range(-1, int(MIN_DIST) + 6, 5)))
# fig, ax = plt.subplots(figsize=(6, 4))
# sns.boxplot(
#     data=hex_df,
#     x="dist_bin",
#     y="sample_count",
#     ax=ax,
#     showfliers=False,
# )
# ax.set_yscale("log")
# ax.set_xlabel("Distance bin (km)")
# ax.set_ylabel("Sample Count")
# ax.set_title("Sample Count distribution by distance bucket")
# plt.tight_layout()
# plt.savefig(FIG_DIR / "boxplot_dist_vs_sample_count.png")
# plt.close(fig)

# ------------------------------------------------------------
# cumulative distribution: cumulative sample count (%) vs distance
# ------------------------------------------------------------

total_samples = hex_df["sample_count"].sum()

cdf_df = hex_df.sort_values("dist_km").copy()
cdf_df["cum_pct"] = 100.0 * cdf_df["sample_count"].cumsum() / total_samples

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(
    cdf_df["dist_km"],
    cdf_df["cum_pct"],
    drawstyle="steps-post",
)

# Highlight p50 and p95
ax.axhline(50, linestyle="--", linewidth=0.8, label="50 %", color="red")
ax.axhline(95, linestyle=":", linewidth=0.8, label="95 %", color="green")

ax.set_xlabel("Distance from Shore (km)")
ax.set_ylabel("Cumulative Sample Count (%)")
ax.set_ylim(0, 100)
ax.set_title("Cumulative Sample Count vs Distance")
ax.grid(True, which="both", linestyle=":", linewidth=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "cdf_dist_vs_sample_count.png")
plt.close(fig)
