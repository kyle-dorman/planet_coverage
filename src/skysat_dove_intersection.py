from functools import lru_cache
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Tuple

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
from shapely import wkb
from tqdm.auto import tqdm

from src.tides import tide_model

SCHEMA = {
    "dove_id": pl.Utf8,
    "skysat_id": pl.Utf8,
    "grid_id": pl.UInt32,
    "cell_id": pl.UInt32,
    "overlap_area": pl.Float32,
    "geometry_wkb": bytes,
    "dove_acquired": pl.Datetime,
    "skysat_acquired": pl.Datetime,
    "time_of_day_delta_sec": pl.Float32,
    "acquired_delta_sec": pl.Float32,
    "dove_tide_height": pl.Float32,
    "skysat_tide_height": pl.Float32,
    "tide_height_delta": pl.Float32,
    "has_8_channel": pl.Boolean,
}


def load_gdf(pth: Path | str, crs: str) -> gpd.GeoDataFrame:
    df_pd = pd.read_parquet(pth)
    df_pd["geometry"] = df_pd["geometry_wkb"].apply(wkb.loads)  # type: ignore
    df_pd = df_pd.drop(columns=["geometry_wkb"])
    satellite_gdf = gpd.GeoDataFrame(df_pd, geometry="geometry", crs="EPSG:4326").to_crs(crs)
    return satellite_gdf


@lru_cache(maxsize=1)
def _load_grids(path: str) -> gpd.GeoDataFrame:
    """Cache coastal grids in-process to avoid re-reading on each call."""
    return gpd.read_file(path)


def get_save_path(base: Path, year: int, index: int) -> Path:
    hex_id = f"{index:06x}"  # unique 6-digit hex, e.g. '0f1a2b'
    d1, d2, d3 = hex_id[:2], hex_id[2:4], hex_id[4:6]
    save_path = base / str(year) / d1 / d2 / d3
    return save_path


# helper so we can use imap_unordered and keep tqdm progress
def _star_compute(args: Tuple[Any, ...]) -> pd.DataFrame | None:
    return process_pair(*args)


def process_pair(
    save_dir: Path,
    dove_path: Path,
    skysat_path: Path,
    grid_path: Path,
    query_path: Path,
    tide_data_dir: Path,
    clear_thresh: float,
    time_window: float,  # days
    time_of_day_window: float,  # hours  <-- NEW
    overlap_area: float,
    max_tide_difference: float,
    filter_coastal_area: bool,
) -> pd.DataFrame | None:
    """
    Process a single Dove / SkySat file pair.

    Parameters
    ----------
    save_dir : Path
        The save directory
    dove_path : Path
        Parquet file for the Dove cell.
    skysat_path : Path
        Parquet file for the matching SkySat cell.
    grid_path : Path
        Path to the GPKG file containing coastal grid polygons.
    overlap_area : float
        Minimum overlap area in square meters.
    max_tide_difference : float
        Maximum allowable difference in tide height (m).
    filter_coastal_area : bool
        If True, restrict SkySat frames to those intersecting coastal land areas (mainlands and islands).
    time_window : float
        Maximum allowable difference in acquisition time (days) between Dove and SkySat.
    time_of_day_window : float
        Maximum allowable difference in clock time (hours) between the Dove and SkySat acquisitions.
    """
    grids_df = _load_grids(str(grid_path)).rename(columns={"cell_id": "grid_id"})
    query_df = _load_grids(str(query_path))
    query_df = query_df.set_index("cell_id")

    tm = tide_model(tide_data_dir, "GOT4.10", "GOT")
    assert tm is not None

    d_df = load_gdf(dove_path, "EPSG:4326")  # type: ignore
    ss_df = load_gdf(skysat_path, "EPSG:4326")  # type: ignore
    cell_ids = d_df.cell_id.unique()
    assert len(cell_ids) == 1
    cell_id = cell_ids[0]
    assert (ss_df.cell_id == cell_id).all()

    year = int(dove_path.parent.parent.parent.parent.name)
    save_dest = get_save_path(save_dir, year, cell_id) / "data.parquet"

    grids_df = grids_df[grids_df.intersects(query_df.loc[cell_id].geometry)]
    grid_geom = query_df.to_crs("EPSG:4326").loc[cell_id].geometry

    d_df = d_df[
        (d_df.clear_percent > clear_thresh)
        & (d_df.publishing_stage == "finalized")
        & (d_df.quality_category == "standard")
        & (d_df.has_sr_asset)
        & (d_df.ground_control)
    ].copy()
    ss_df = ss_df[
        (ss_df.clear_percent > clear_thresh)
        & (ss_df.publishing_stage == "finalized")
        & (ss_df.quality_category == "standard")
        & (ss_df.has_sr_asset)
        & (ss_df.ground_control)
    ].copy()
    if not len(ss_df) or not len(d_df):
        return

    if filter_coastal_area:
        keep_grids = grids_df[~(grids_df["dist_km"].isna()) & (grids_df["dist_km"] < 5)].to_crs("EPSG:4326")

        # Filter to just coastal geoms
        ss_df = ss_df.set_index("id")
        coastal_ss_ids = ss_df[["geometry"]].sjoin(keep_grids[["geometry"]]).index
        ss_df = ss_df.loc[coastal_ss_ids].copy()
        ss_df.reset_index(inplace=True)

    if not len(ss_df):
        return

    grid_centroid = grid_geom.centroid
    latlon = np.array([grid_centroid.y, grid_centroid.x])

    heights = tm.tide_elevations(latlon, d_df.acquired.to_numpy()[None])[0]  # type: ignore
    d_df["tide_height"] = heights
    heights = tm.tide_elevations(latlon, ss_df.acquired.to_numpy()[None])[0]  # type: ignore
    ss_df["tide_height"] = heights

    d_df = d_df.to_crs(grids_df.crs)  # type: ignore
    ss_df = ss_df.to_crs(grids_df.crs)  # type: ignore

    dss_joined = gpd.overlay(
        d_df[["id", "geometry", "acquired", "tide_height", "has_8_channel"]],
        ss_df[["id", "geometry", "acquired", "tide_height"]],
        how="intersection",
    )
    dss_joined = dss_joined.rename(
        columns={
            "id_1": "dove_id",
            "id_2": "skysat_id",
            "acquired_1": "dove_acquired",
            "acquired_2": "skysat_acquired",
            "tide_height_1": "dove_tide_height",
            "tide_height_2": "skysat_tide_height",
        }
    )

    if not len(dss_joined):
        return None

    dss_joined["tide_height_delta"] = (dss_joined.dove_tide_height - dss_joined.skysat_tide_height).abs()
    dss_joined["acquired_delta"] = dss_joined.dove_acquired - dss_joined.skysat_acquired
    dss_joined["overlap_area"] = dss_joined.geometry.area

    # ── clock-time difference (mod 24 h) ────────────────────────────────────
    dove_tod_sec = (
        dss_joined.dove_acquired.dt.hour * 3600
        + dss_joined.dove_acquired.dt.minute * 60
        + dss_joined.dove_acquired.dt.second
    )
    skysat_tod_sec = (
        dss_joined.skysat_acquired.dt.hour * 3600
        + dss_joined.skysat_acquired.dt.minute * 60
        + dss_joined.skysat_acquired.dt.second
    )
    tod_delta = (dove_tod_sec - skysat_tod_sec).abs()
    dss_joined["time_of_day_delta_sec"] = np.minimum(tod_delta, 86400 - tod_delta)

    overlapping = dss_joined[
        (dss_joined.acquired_delta < pd.Timedelta(days=time_window))
        & (dss_joined.acquired_delta > pd.Timedelta(days=-time_window))
        & (dss_joined.time_of_day_delta_sec < time_of_day_window * 3600)
        & (dss_joined.overlap_area > overlap_area)
        & (dss_joined.tide_height_delta < max_tide_difference)
    ].copy()

    if not len(overlapping):
        return None

    # initial spatial join: match coastal grids to grid cells by containment
    joined = gpd.overlay(
        overlapping[["dove_id", "skysat_id", "geometry"]],
        grids_df[["grid_id", "geometry"]],
    )
    joined["area"] = joined.geometry.area
    joined = joined.sort_values(by="area", ascending=False).drop_duplicates(subset=["dove_id", "skysat_id"])

    overlap_with_grid_id = overlapping.merge(
        joined[["dove_id", "skysat_id", "grid_id"]], on=["dove_id", "skysat_id"], how="inner"  # "left"
    )
    assert not overlap_with_grid_id.grid_id.isna().any()

    # ── normalise dtypes so they can be written to GPKG ────────────────────────
    # GeoPackage cannot handle micro-second timedeltas, so turn them into seconds
    overlap_with_grid_id["acquired_delta_sec"] = overlap_with_grid_id["acquired_delta"].dt.total_seconds()
    overlap_with_grid_id.drop(columns="acquired_delta", inplace=True)

    # ensure datetimes use nanosecond resolution (default pandas) instead of µs
    overlap_with_grid_id["dove_acquired"] = pd.to_datetime(overlap_with_grid_id["dove_acquired"]).astype(
        "datetime64[ns]"
    )
    overlap_with_grid_id["skysat_acquired"] = pd.to_datetime(overlap_with_grid_id["skysat_acquired"]).astype(
        "datetime64[ns]"
    )
    overlap_with_grid_id["cell_id"] = cell_id

    if not len(overlap_with_grid_id):
        return

    overlap_with_grid_id["geometry_wkb"] = overlap_with_grid_id["geometry"].apply(wkb.dumps)
    overlap_with_grid_id.drop(columns=["geometry"], inplace=True)

    # Convert to Polars (or pandas) to write out
    pl_out = pl.from_pandas(overlap_with_grid_id, schema_overrides=SCHEMA, include_index=False)

    save_dest.parent.mkdir(exist_ok=True, parents=True)
    pl_out.write_parquet(save_dest)


@click.command()
@click.option(
    "--base-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Root directory that contains dove/ and skysat result trees.",
)
@click.option(
    "--save-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Where to save the results.",
)
@click.option(
    "--query-grids-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="GeoPackage of polygons use for planet querying (index must match cell_id)",
)
@click.option(
    "--grid-file", required=True, type=click.Path(exists=True, dir_okay=False), help="Coastal grids GPKG file."
)
@click.option(
    "--tide-data-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing tidal constituent files for tide_model",
)
@click.option("--clear-thresh", default=75.0, show_default=True, help="Minimum SkySat clear_percent.")
@click.option("--time-window", default=15.0, show_default=True, help="Max |Δacquired| in DAYS between Dove and SkySat.")
@click.option(
    "--time-of-day-window",
    default=2.0,
    show_default=True,
    help="Max time-of-day difference (in HOURS) between Dove and SkySat.",
)
@click.option("--overlap-area", default=300**2, show_default=True, help="Minimum overlap area in square meters.")
@click.option(
    "--max-tide-difference",
    default=0.5,
    show_default=True,
    help="Maximum allowable difference in tide height (m).",
)
@click.option(
    "--filter-coastal-area",
    is_flag=True,
    default=False,
    help="If given, restrict SkySat frames to those intersecting coastal land areas (mainlands and islands).",
)
@click.option("--nproc", type=int, default=cpu_count() - 1, show_default=True, help="Number of worker processes.")
@click.option("--chunksize", default=16, show_default=True, help="Chunk size passed to imap_unordered.")
def main(
    base_dir: str,
    save_dir: Path,
    query_grids_path: Path,
    grid_file: str,
    tide_data_dir: Path,
    clear_thresh: float,
    time_window: float,
    time_of_day_window: float,
    overlap_area: float,
    max_tide_difference: float,
    filter_coastal_area: bool,
    nproc: int,
    chunksize: int,
) -> None:
    """
    Find intersecting Dove/SkySat footprints for every matching pair of Parquet
    files under *base_dir* and write a combined GPKG.
    """
    base = Path(base_dir)
    grid_path = Path(grid_file)

    tasks = []
    for dove_path in base.glob("dove/results/*/*/*/*/data.parquet"):
        skysat_path = Path(str(dove_path).replace("/dove/", "/skysat/"))
        if not skysat_path.exists():
            continue
        tasks.append(
            (
                Path(save_dir),
                dove_path,
                skysat_path,
                grid_path,
                query_grids_path,
                tide_data_dir,
                clear_thresh,
                time_window,
                time_of_day_window,
                overlap_area,
                max_tide_difference,
                filter_coastal_area,
            )
        )

    if not tasks:
        click.echo("No valid SkySat–Dove pairs found.")
        return

    nproc = min(cpu_count(), nproc, len(tasks))
    click.echo(f"Processing {len(tasks)} pairs on {nproc} processes …")

    with Pool(processes=nproc) as pool:
        for _ in tqdm(
            pool.imap_unordered(_star_compute, tasks, chunksize=chunksize),
            total=len(tasks),
            desc="Intersecting Year/Grids",
        ):
            pass


if __name__ == "__main__":
    main()
