from functools import lru_cache
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Tuple

import click
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkb
from tqdm.auto import tqdm

from src.tides import tide_model


def load_gdf(pth: Path | str, crs: str) -> gpd.GeoDataFrame:
    df_pd = pd.read_parquet(pth)
    df_pd["geometry"] = df_pd["geometry_wkb"].apply(wkb.loads)  # type: ignore
    df_pd = df_pd.drop(columns=["geometry_wkb"])
    satellite_gdf = gpd.GeoDataFrame(df_pd, geometry="geometry", crs="EPSG:4326").to_crs(crs)
    return satellite_gdf


@lru_cache(maxsize=1)
def _load_grids(path: str) -> gpd.GeoDataFrame:
    """Cache coastal grids in‑process to avoid re‑reading on each call."""
    return gpd.read_file(path)


# helper so we can use imap_unordered and keep tqdm progress
def _star_compute(args: Tuple[Any, ...]) -> pd.DataFrame | None:
    return process_pair(*args)


def process_pair(
    base: Path,
    dove_path: Path,
    skysat_path: Path,
    grid_path: Path,
    query_path: Path,
    tide_data_dir: Path,
    clear_thresh: float,
    time_window: float,  # days
    overlap_area: float,
    max_tide_difference: float,
    filter_coastal_area: bool,
) -> pd.DataFrame | None:
    """
    Process a single Dove / SkySat file pair.

    Parameters
    ----------
    base : Path
        The root run directory
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

    Returns
    -------
    pd.DataFrame
        Intersection rows with cell_id and WKB geometry.
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

    grid_geom = query_df.centroid.to_crs("EPSG:4326").loc[cell_id]

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
        mainlands = gpd.read_file(base.parent / "shorelines" / "mainlands.gpkg")
        mainlands = mainlands[mainlands.intersects(grid_geom)]
        big_islands = gpd.read_file(base.parent / "shorelines" / "big_islands.gpkg")
        big_islands = big_islands[big_islands.intersects(grid_geom)]
        small_islands = gpd.read_file(base.parent / "shorelines" / "small_islands.gpkg")
        small_islands = small_islands[small_islands.intersects(grid_geom)]

        # Filter to just coastal geoms
        ss_df = ss_df.set_index("id")
        coastal_ss_ids = np.unique(
            np.concatenate(
                [
                    gpd.sjoin(ss_df[["geometry"]], lyr[["geometry"]]).index
                    for lyr in (mainlands, big_islands, small_islands)
                ]
            )
        )
        ss_df = ss_df.loc[coastal_ss_ids].copy()
        ss_df.reset_index(inplace=True)

    if not len(ss_df):
        return

    latlon = np.array([grid_geom.y, grid_geom.x])

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

    overlapping = dss_joined[
        (dss_joined.acquired_delta < pd.Timedelta(days=time_window))
        & (dss_joined.acquired_delta > pd.Timedelta(days=-time_window))
        & (dss_joined.overlap_area > overlap_area)
        & (dss_joined.tide_height_delta < max_tide_difference)
    ].copy()

    if not len(overlapping):
        return None

    # initial spatial join: match coastal grids to query cells by containment
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
    # GeoPackage cannot handle micro‑second timedeltas, so turn them into seconds
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

    return overlap_with_grid_id


@click.command()
@click.option(
    "--base-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Root directory that contains dove/ and skysat result trees.",
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
@click.option(
    "--out",
    type=click.Path(dir_okay=False),
    default="skysat_dove_intersections.gpkg",
    show_default=True,
    help="Output GPKG path.",
)
def main(
    base_dir: str,
    query_grids_path: Path,
    grid_file: str,
    tide_data_dir: Path,
    clear_thresh: float,
    time_window: float,
    overlap_area: float,
    max_tide_difference: float,
    filter_coastal_area: bool,
    nproc: int,
    chunksize: int,
    out: str,
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
                base,
                dove_path,
                skysat_path,
                grid_path,
                query_grids_path,
                tide_data_dir,
                clear_thresh,
                time_window,
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
        results = list(
            tqdm(
                pool.imap_unordered(_star_compute, tasks, chunksize=chunksize),
                total=len(tasks),
                desc="Intersecting Year/Grids",
            )
        )

    results = [r for r in results if r is not None]
    if results:
        all_results = pd.concat(results, ignore_index=True)
        gdf = gpd.GeoDataFrame(all_results, geometry="geometry", crs=_load_grids(grid_path).crs)
        gdf.to_file(base / out, driver="GPKG")
        click.echo(f"✅ Saved {len(all_results)} intersections → {base / out}")
    else:
        click.echo("No valid intersections produced.")


if __name__ == "__main__":
    main()
