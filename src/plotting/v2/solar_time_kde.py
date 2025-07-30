import logging
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def wherestr(fiscal_year: int) -> str:
    return f"""
item_type               = 'PSScene'
AND coverage_pct        > 0.5
AND acquired            >=  TIMESTAMP '{fiscal_year - 1}-12-01'
AND acquired            <   TIMESTAMP '{fiscal_year}-12-01'
AND publishing_stage    = 'finalized'
AND quality_category    = 'standard'
AND clear_percent       > 75.0
AND has_sr_asset
AND ground_control
        """


def duckdb_smoothed_local_time_density(
    con,
    table: str,
    where: str,
    x_min=6.0,
    x_max=16.0,
    n_bins=300,  # resolution of the x-grid (increase for smoother look)
    bandwidth_hours=0.15,  # Gaussian bandwidth in HOURS (≈ 9 minutes)
):
    bins = np.linspace(x_min * 3600, x_max * 3600, n_bins + 1, dtype=np.int32)
    edges_sql = "LIST_VALUE(" + ", ".join(f"{b}" for b in bins) + ")"

    query = f"""
    SELECT
        instrument,
        histogram(
            solar_time_offset_seconds,
            {edges_sql}
        ) AS bucket
    FROM {table}
    WHERE {where}
    GROUP BY instrument
    """
    return con.execute(query).fetchdf()


def plot_local_time_distributions(
    df: pd.DataFrame,
    ax,
    fiscal_year: int,
    x_min=6.0,
    x_max=16.0,
):
    """
    Make a 1D KDE distribution plot per instrument for local time-of-day (hours).
    Y-axis is scaled to percentage so curves are comparable.
    """
    if df.empty:
        raise RuntimeError("No rows returned after filtering — adjust WHERE clause or table name.")

    colors = {"PS2": "tab:blue", "PS2.SD": "tab:green", "PSB.SD": "tab:red"}
    x_grid = np.linspace(x_min, x_max, 600)

    for inst, color in colors.items():
        idf = df[df["instrument"] == inst]
        if idf.empty:
            continue
        buckets = df[df["instrument"] == inst].iloc[0].bucket

        counts = list(buckets.values())[1:-1]
        edges = np.array(list(buckets.keys()))[:-1] / 3600.0
        centers = 0.5 * (edges[:-1] + edges[1:])
        # simple smoothing with a small Gaussian kernel
        kx = np.linspace(-3, 3, 31)
        ker = np.exp(-0.5 * kx**2)
        ker /= ker.sum()
        dens = np.convolve(counts, ker, mode="same")
        # resample to x_grid
        dens = np.interp(x_grid, centers, dens)
        # n = sum(counts)

        # scale to percentage so areas are comparable within [x_min, x_max]
        area = np.trapz(dens, x_grid)
        pct = dens / area if area > 0 else dens
        ax.plot(x_grid, pct, color=color, lw=2, label=inst)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1.2)
    ax.set_title(f"{fiscal_year}")
    ax.grid(True, which="both", ls=":", lw=0.3)


BASE = Path("/Users/kyledorman/data/planet_coverage/points_30km/")
con = duckdb.connect()  # or your existing connection
f_pat = "*/coastal_results_solar/*/*/*/coastal_points.parquet"
pat = str(BASE / f_pat)

all_files = list(BASE.glob(f_pat))
if not all_files:
    logger.error("No solar parquet files found in coastal_results_solar paths.")
    raise FileNotFoundError("No coastal_results_solar parquet files found")

con.execute(
    f"""
    CREATE OR REPLACE VIEW samples_all AS
    SELECT * FROM read_parquet('{pat}')
    """
)
fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, figsize=(4 * 3, 3 * 3), constrained_layout=True)
x_min = 6.0
x_max = 13.0
for year, ax in tqdm(zip(range(2016, 2025), axes.flatten()), total=9):
    ddf = duckdb_smoothed_local_time_density(con, table="samples_all", where=wherestr(year), x_min=x_min, x_max=x_max)
    plot_local_time_distributions(ddf, ax, year, x_min=x_min, x_max=x_max)

for i in range(3):
    axes[i, 0].set_ylabel("Frequency (%)")
    axes[-1, i].set_xlabel("Local time (hours)")
axes[1, 1].legend()


FIG_DIR = BASE.parent / "figs_v2" / "solar_time"
FIG_DIR.mkdir(exist_ok=True, parents=True)

fig.suptitle("Local-time distribution by instrument", fontsize=14)
fig.savefig(FIG_DIR / "local_time_distribution_by_instrument.png", dpi=150)
