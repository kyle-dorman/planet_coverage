from functools import lru_cache
from multiprocessing import Pool, cpu_count
from pathlib import Path

import click
import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
from shapely import wkb
from tqdm.auto import tqdm


def load_gdf(pth, crs):
    df_pd = pd.read_parquet(pth)
    df_pd["geometry"] = df_pd["geometry_wkb"].apply(wkb.loads)  # type: ignore
    df_pd = df_pd.drop(columns=["geometry_wkb"])
    satellite_gdf = gpd.GeoDataFrame(df_pd, geometry="geometry", crs="EPSG:4326").to_crs(crs)
    return satellite_gdf


@lru_cache(maxsize=1)
def _load_grids(path: str) -> gpd.GeoDataFrame:
    """Cache coastal grids in‑process to avoid re‑reading on each call."""
    return gpd.read_file(path)


def _parquet_empty(pth: Path) -> bool:
    """
    Return True if a Parquet file has zero rows without loading full data.
    """
    meta = pq.ParquetFile(pth).metadata
    return meta.num_rows == 0


def process_pair(
    dove_path: Path,
    skysat_path: Path,
    grid_path: Path,
    clear_thresh: float,
    time_window: float,  # hours
    overlap_thresh: float,
) -> pd.DataFrame:
    """
    Process a single Dove / SkySat file pair.

    Parameters
    ----------
    dove_path : Path
        Parquet file for the Dove cell.
    skysat_path : Path
        Parquet file for the matching SkySat cell.
    grid_path : Path
        Path to the GPKG file containing coastal grid polygons.

    Returns
    -------
    pd.DataFrame
        Intersection rows with cell_id and WKB geometry.
    """
    grids_df = _load_grids(str(grid_path))
    d_df = load_gdf(dove_path, grids_df.crs)
    ss_df = load_gdf(skysat_path, grids_df.crs)

    filtered_d_df = d_df[
        (d_df.publishing_stage == "finalized")
        & (d_df.quality_category == "standard")
        & (d_df.has_sr_asset)
        & (d_df.ground_control)
    ]
    filtered_ss_df = ss_df[
        (ss_df.clear_percent > clear_thresh)
        & (ss_df.publishing_stage == "finalized")
        & (ss_df.quality_category == "standard")
        & (ss_df.has_sr_asset)
        & (ss_df.ground_control)
    ]

    dss_joined = gpd.overlay(
        filtered_d_df[["id", "geometry", "acquired", "has_8_channel"]],
        filtered_ss_df[["id", "geometry", "acquired"]],
        how="intersection",
    ).rename(
        columns={
            "id_1": "dove_id",
            "id_2": "skysat_id",
            "acquired_1": "dove_acquired",
            "acquired_2": "skysat_acquired",
        }
    )

    dss_joined["acquired_delta"] = dss_joined.dove_acquired - dss_joined.skysat_acquired
    dss_joined["overlap_pct_crop"] = dss_joined.geometry.area / (300**2)

    overlapping = dss_joined[
        (dss_joined.acquired_delta < pd.Timedelta(hours=time_window))
        & (dss_joined.acquired_delta > pd.Timedelta(hours=-time_window))
        & (dss_joined.overlap_pct_crop > overlap_thresh)
    ].copy()

    overlapping.drop(columns="overlap_pct_crop", inplace=True)

    # initial spatial join: match coastal grids to query cells by containment
    joined = gpd.overlay(
        overlapping[["dove_id", "skysat_id", "geometry"]],
        grids_df[["cell_id", "geometry"]],
    )

    # Faster for large DataFrames
    overlapping["_key"] = list(zip(overlapping["dove_id"], overlapping["skysat_id"]))
    joined["_key"] = list(zip(joined["dove_id"], joined["skysat_id"]))

    mask = overlapping["_key"].isin(joined["_key"])
    overlapping.drop(columns="_key", inplace=True)
    joined.drop(columns="_key", inplace=True)

    # separate those with overlap from those without
    misses = ~mask
    joined_missing = overlapping[misses]

    joined["area"] = joined.geometry.area
    joined = joined.sort_values(by="area", ascending=False).drop_duplicates(subset=["dove_id", "skysat_id"])

    # for any missing, assign nearest cell
    if not joined_missing.empty:
        print(f"{len(joined_missing)} SkyDat-Dove grids lack overlap; assigning nearest cell")
        nearest = gpd.sjoin_nearest(
            left_df=joined_missing[["dove_id", "skysat_id", "geometry"]],
            right_df=grids_df[["cell_id", "geometry"]],
            how="left",
        )
        # combine overlap and nearest
        joined_final = pd.concat([joined, nearest], ignore_index=True)
    else:
        joined_final = joined

    overlap_with_cell_id = overlapping.merge(
        joined_final[["dove_id", "skysat_id", "cell_id"]], on=["dove_id", "skysat_id"], how="left"
    )
    assert not overlap_with_cell_id.cell_id.isna().any()
    assert len(overlap_with_cell_id) == len(overlapping)
    overlap_with_cell_id["geometry_wkb"] = overlap_with_cell_id.geometry.map(lambda geom: geom.wkb)
    return overlap_with_cell_id


@click.command()
@click.option(
    "--base-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Root directory that contains dove/ and skysat result trees.",
)
@click.option(
    "--grid-file", required=True, type=click.Path(exists=True, dir_okay=False), help="Coastal grids GPKG file."
)
@click.option("--clear-thresh", default=75.0, show_default=True, help="Minimum SkySat clear_percent.")
@click.option("--time-window", default=1.0, show_default=True, help="Max |Δacquired| in HOURS between Dove and SkySat.")
@click.option("--overlap-thresh", default=1.0, show_default=True, help="Minimum overlap percentage (1.0 ≙ 100 %).")
@click.option("--nproc", default=cpu_count() - 1, show_default=True, help="Number of worker processes.")
@click.option("--chunksize", default=16, show_default=True, help="Chunk size passed to imap_unordered.")
@click.option(
    "--out",
    type=click.Path(dir_okay=False),
    default="skysat_dove_intersections.gpkg",
    show_default=True,
    help="Output GPKG path.",
)
def main(base_dir, grid_file, clear_thresh, time_window, overlap_thresh, nproc, chunksize, out):
    """
    Find intersecting Dove/SkySat footprints for every matching pair of Parquet
    files under *base_dir* and write a combined GPKG.
    """
    base = Path(base_dir)
    grid_path = Path(grid_file)

    tasks = []
    for dove_path in base.glob("dove/results/2017/*/*/*/data.parquet"):
        skysat_path = Path(str(dove_path).replace("/dove/", "/skysat/"))
        if not skysat_path.exists() or _parquet_empty(dove_path) or _parquet_empty(skysat_path):
            continue
        tasks.append(
            (
                dove_path,
                skysat_path,
                grid_path,
                clear_thresh,
                time_window,
                overlap_thresh,
            )
        )

    if not tasks:
        click.echo("No valid SkySat–Dove pairs found.")
        return

    nproc = min(cpu_count(), nproc)
    click.echo(f"Processing {len(tasks)} pairs on {nproc} processes …")

    def _star(args):
        try:
            return process_pair(*args)
        except Exception as exc:
            click.echo(f"⚠️  Skipping {args[0].parent.name}: {exc}")
            return None

    with Pool(processes=nproc) as pool:
        results = list(
            tqdm(
                filter(None, pool.imap_unordered(_star, tasks, chunksize=chunksize)),
                total=len(tasks),
                desc="Pairs",
            )
        )

    if results:
        all_results = pd.concat(results, ignore_index=True)
        gdf = gpd.GeoDataFrame(all_results, geometry="geometry", crs=_load_grids(grid_path).crs)
        out_path = Path(out)
        gdf.to_file(out_path, driver="GPKG")
        click.echo(f"✅ Saved {len(all_results)} intersections → {out_path}")
    else:
        click.echo("No valid intersections produced.")


if __name__ == "__main__":
    main()
