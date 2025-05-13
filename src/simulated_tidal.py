import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, Optional, Union

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import pvlib
from pyproj import Transformer
from shapely import Point
from tqdm.auto import tqdm

from src.tides import tide_model

logger = logging.getLogger(__name__)


#
# --------------------------------------------------------------------- #
# compute_tide_info
#
# Given a single grid cell id and its centroid geometry, simulate one full
# year of 1‑minute tide elevations; bin heights, compute satellite offsets,
# subsample by satellite‑specific stride, and return summary statistics
# (median / 95th‑percentile days‑between passes, counts).
#
# The function relies on globals (minutes, NBINS, SENTINEL_TIME, …) that are
# initialised in `main()` so it can be pickled/executed inside a Pool worker.
# --------------------------------------------------------------------- #
def compute_tide_info(
    cell_id: Union[int, str],
    grid_point: Point,
    start_ts: np.datetime64,
    end_ts: np.datetime64,
    nbins: int,
    tide_data_dir: str,
    sentinel_time: pd.Timedelta,
    landsat_time: pd.Timedelta,
    planet_time: pd.Timedelta,
    sentinel_stride: int,
    landsat_stride: int,
) -> Dict[str, Any] | None:
    ts_range = end_ts - start_ts
    minutes = np.arange(start_ts, end_ts, np.timedelta64(1, "m")).astype("datetime64[ns]")
    tm = tide_model(Path(tide_data_dir), "GOT4.10", "GOT")

    latlons = np.array([grid_point.y, grid_point.x])

    try:
        tide_elevations = tm.tide_elevations(latlons, times=[minutes])[0]  # type: ignore
    except IndexError as e:
        logger.error(f"Invalid cell_id {cell_id}")
        logger.exception(e)
        return None

    tides_df = pd.DataFrame(
        {
            "acquired": minutes,
            "tide_height": tide_elevations,
            "lat": latlons[0],
            "lon": latlons[1],
        }
    )

    # ------------------------------------------------------------------
    # Build quantile‑based bins so each bin holds ≈ 1/nbins of the samples
    # (equal‑frequency bins instead of equal‑width across the numeric range)
    # ------------------------------------------------------------------
    q_edges = np.linspace(0.0, 1.0, nbins + 1)
    height_edges = np.quantile(tide_elevations, q_edges)

    # Make the first/last edges a hair wider so everything is included
    eps = 1e-6
    height_edges[0] -= eps
    height_edges[-1] += eps
    # this gives integer bins 0–nbins-1
    tides_df["height_bin"] = pd.cut(
        tides_df["tide_height"], bins=height_edges, labels=False, include_lowest=True  # type: ignore
    )  # type: ignore
    assert not tides_df["height_bin"].isna().any()

    solpos = pvlib.solarposition.get_solarposition(tides_df.acquired, tides_df.lat, tides_df.lon)
    tides_df["eot"] = pd.to_timedelta(solpos["equation_of_time"], unit="m").to_numpy()
    tides_df["lon_term"] = pd.to_timedelta(tides_df.lon / 15.0, unit="h").to_numpy()
    tides_df["solar_time"] = tides_df.acquired + tides_df.eot + tides_df.lon_term
    tides_df["solar_time_offset"] = tides_df.solar_time - tides_df.solar_time.dt.normalize()

    #  / 3.6e+12 nanoseconds to hours
    tides_df["sentinel_offset"] = (
        np.abs(np.float32((sentinel_time - tides_df.solar_time_offset).values)) / 3.6e12  # type: ignore
    )  # type: ignore
    tides_df["landsat_offset"] = (
        np.abs(np.float32((landsat_time - tides_df.solar_time_offset).values)) / 3.6e12  # type: ignore
    )  # type: ignore
    tides_df["planet_offset"] = (
        np.abs(np.float32((planet_time - tides_df.solar_time_offset).values)) / 3.6e12  # type: ignore
    )  # type: ignore

    # === group by calendar date and pick the closest-overpass per sensor ===
    tides_df["solar_date"] = tides_df["solar_time"].dt.date

    groups = [
        ("planet", "planet_offset", 1),
        ("sentinel", "sentinel_offset", sentinel_stride),
        ("landsat", "landsat_offset", landsat_stride),
    ]

    out = {"cell_id": int(cell_id)}
    for satname, offset_name, kstride in groups:
        # Due to date rounding we will get 1-2 days that are bad solar windows and should not be sampled
        key_df = tides_df[tides_df[offset_name] < 1]

        # Subsample every N days
        df = (
            key_df.loc[
                key_df.groupby("solar_date")[offset_name].idxmin(),
                ["solar_date", "acquired", "height_bin", offset_name],
            ]
            .iloc[::kstride]
            .reset_index(drop=True)
        )

        logger.info(f"{satname} {kstride}-day stride count: {len(df)}")

        search_ranges = [
            (0, "low"),
            (nbins - 1, "high"),
        ]
        for height, height_name in search_ranges:
            is_height = df.height_bin == height
            # reset to a fresh 0…N‑1 integer index; the original DatetimeIndex
            # is no longer needed and there is only one level
            tide_match = df[is_height].copy().reset_index(drop=True)

            # Pad so we get a wrapped result for time diff
            if is_height.sum() > 0:
                first_row = tide_match.iloc[0].copy()
                first_row.acquired = first_row.acquired + ts_range
                tide_match.loc[len(tide_match)] = first_row

            full_diffs = tide_match.acquired.diff().dt.total_seconds().dropna() / 3600.0 / 24.0

            out[f"{satname}_{height_name}_days_between_p50"] = float(full_diffs.quantile(0.5))  # type: ignore
            out[f"{satname}_{height_name}_days_between_p95"] = float(full_diffs.quantile(0.95))  # type: ignore
            out[f"{satname}_{height_name}_count"] = int(is_height.sum())

    return out


# helper so we can use imap_unordered and keep tqdm progress
def _star_compute(args):
    return compute_tide_info(*args)


@click.command()
@click.option(
    "--base-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing ocean_grids.gpkg and shoreline layers",
)
@click.option(
    "--tide-data-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Directory with GOT/FES tide constituents",
)
@click.option(
    "--start-date", type=str, default="2023-12-01T00:00", show_default=True, help="Start of simulation (UTC, ISO‑8601)"
)
@click.option(
    "--end-date", type=str, default="2024-12-01T00:00", show_default=True, help="End of simulation (UTC, ISO‑8601)"
)
@click.option("--nbins", type=int, default=10, show_default=True, help="Number of equal‑width tide‑height bins")
@click.option("--sentinel-stride", type=int, default=5, show_default=True, help="Stride (days) for Sentinel sampling")
@click.option("--landsat-stride", type=int, default=8, show_default=True, help="Stride (days) for Landsat sampling")
@click.option("--processes", type=int, default=None, help="Number of worker processes (defaults to CPU count)")
@click.option(
    "--out-path",
    type=click.Path(dir_okay=True),
    required=True,
    help="Output save path (simulated.csv)",
)
def main(
    base_dir: str,
    tide_data_dir: str,
    start_date: str,
    end_date: str,
    nbins: int,
    sentinel_stride: int,
    landsat_stride: int,
    processes: Optional[int],
    out_path: str,
) -> None:
    """
    Simulate tidal sampling for each coastal grid cell in parallel and
    write a summary table to *out_dir/out.csv*.
    """
    # ------------------------------------------------------------------ #
    # 0.  Global configuration shared with worker processes
    # ------------------------------------------------------------------ #
    sentinel_time = pd.Timedelta(hours=10, minutes=30)
    planet_time = pd.Timedelta(hours=11)
    landsat_time = pd.Timedelta(hours=10)

    start_ts = np.datetime64(start_date)
    end_ts = np.datetime64(end_date)

    # ------------------------------------------------------------------ #
    # 1.  Load and pre‑filter geometry layers
    # ------------------------------------------------------------------ #
    logger.info("Loading ocean grids and coastline layers")
    ocean_grids = gpd.read_file(Path(base_dir) / "ocean_grids.gpkg")
    # ca_ocean = gpd.read_file(Path(base_dir) / "ca_ocean.geojson")
    ocean_grids["centroid"] = ocean_grids.geometry.centroid
    all_grids = ocean_grids.to_crs("EPSG:4326")
    # all_grids = all_grids[all_grids.geometry.intersects(ca_ocean.union_all())]

    mainlands = gpd.read_file(Path(base_dir).parent / "shorelines" / "mainlands.gpkg")
    big_islands = gpd.read_file(Path(base_dir).parent / "shorelines" / "big_islands.gpkg")
    small_islands = gpd.read_file(Path(base_dir).parent / "shorelines" / "small_islands.gpkg")

    assert mainlands.crs == all_grids.crs

    logger.info("Spatial join to find coastal cells")
    coastal_ids = np.unique(
        np.concatenate(
            [
                gpd.sjoin(all_grids[["cell_id", "geometry"]], lyr[["geometry"]]).cell_id.unique()
                for lyr in (mainlands, big_islands, small_islands)
            ]
        )
    )
    grid_df = ocean_grids.loc[ocean_grids.cell_id.isin(coastal_ids), ["cell_id", "centroid"]]
    transformer = Transformer.from_crs(ocean_grids.crs, all_grids.crs, always_xy=True)

    # ------------------------------------------------------------------ #
    # 2.  Run each cell in parallel
    # ------------------------------------------------------------------ #
    logger.info(f"Computing tides in parallel (~{len(grid_df)} cells)")
    nproc = processes or cpu_count()

    # build task list
    tasks = []
    for _, row in grid_df.iterrows():
        grid_point = Point(*transformer.transform(row.centroid.x, row.centroid.y))
        tasks.append(
            (
                row.cell_id,
                grid_point,
                start_ts,
                end_ts,
                nbins,
                tide_data_dir,
                sentinel_time,
                landsat_time,
                planet_time,
                sentinel_stride,
                landsat_stride,
            )
        )

    # run in parallel with a live progress bar
    logger.info("Launching worker pool …")
    with Pool(processes=nproc) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_star_compute, tasks, chunksize=32),
                total=len(tasks),
                desc="Cells",
            )
        )

    # ------------------------------------------------------------------ #
    # 3.  Collect + save
    # ------------------------------------------------------------------ #
    results_df = pd.DataFrame([r for r in results if r is not None])
    results_df.to_csv(out_path, index=False)
    logger.info(f"Wrote results to {out_path}")


# Entrypoint for CLI
if __name__ == "__main__":
    # configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
    main()
