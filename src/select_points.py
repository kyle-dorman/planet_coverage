#!/usr/bin/env python3
"""
select_points.py

Generate a regular Sinusoidal grid, buffer points and retain only those
that intersect coastal strips.  Uses a prepared union for lightning-fast
intersection tests, Dask parallelism, and a live progress bar.
"""
import logging
import multiprocessing as mp

import click
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
from shapely.prepared import prep

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_coastal(path: str, proj_crs: str) -> gpd.GeoDataFrame:
    """
    Load coastal strips GeoPackage and verify CRS.
    """
    logger.info("Loading coastal strips from %s", path)
    coastal = gpd.read_file(path)
    assert coastal.crs is not None
    assert coastal.crs.to_string() == proj_crs, f"Expected CRS {proj_crs}, got {coastal.crs}"
    assert len(coastal) == 1, "No coastal features found"
    return coastal


def generate_grid(coastal: gpd.GeoDataFrame, sinus_crs: str, step: float) -> gpd.GeoDataFrame:
    """
    Generate a regular grid in sinusoidal projection covering coastal bounds.
    """
    bounds = coastal.to_crs(sinus_crs).total_bounds
    minx, miny, maxx, maxy = bounds
    xs = np.arange(minx, maxx + step, step)
    ys = np.arange(miny, maxy + step, step)
    logger.info("Generating grid: %d cols × %d rows = %d points", len(xs), len(ys), xs.size * ys.size)
    xx, yy = np.meshgrid(xs, ys)
    grid = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()), crs=sinus_crs)
    return grid


def filter_partition(
    df: gpd.GeoDataFrame, coastal: gpd.GeoDataFrame, buffer_radius: float, proj_crs: str
) -> gpd.GeoDataFrame:
    """
    Buffer points and keep those intersecting the coastal union.
    """
    # Buffer points in projected CRS
    buffered = df.to_crs(proj_crs).geometry.buffer(buffer_radius)
    # Prepare coastal union for fast intersects
    coastal_union = coastal.union_all(method="coverage")
    prep_coastal = prep(coastal_union)
    mask = buffered.apply(lambda geom: prep_coastal.intersects(geom))
    return df.loc[mask]


def save_points(df: gpd.GeoDataFrame, output_path: str) -> None:
    """
    Save filtered points to GeoPackage.
    """
    logger.info("Saving %d points to %s", len(df), output_path)
    df.to_file(output_path, driver="GPKG")


@click.command()
@click.option(
    "--coastal-path", "-c", type=click.Path(exists=True), required=True, help="Path to coastal strips GeoPackage"
)
@click.option("--output-path", "-o", type=click.Path(), required=True, help="Output GeoPackage for filtered points")
@click.option("--sinus-crs", default="ESRI:54008", show_default=True, help="Sinusoidal CRS code for grid")
@click.option("--proj-crs", default="EPSG:6933", show_default=True, help="Projected CRS for buffering")
@click.option("--step", type=float, default=5559.7522, show_default=True, help="Grid spacing in meters")
@click.option("--buffer-factor", type=float, default=0.5, show_default=True, help="Fraction of step for buffer radius")
@click.option(
    "--partitions",
    type=int,
    default=mp.cpu_count() - 1,
    help="Number of Dask partitions; defaults to CPU count minus one",
)
def main(
    coastal_path: str,
    output_path: str,
    sinus_crs: str,
    proj_crs: str,
    step: float,
    buffer_factor: float,
    partitions: int,
) -> None:
    # Determine partitions
    partitions = min(partitions, mp.cpu_count() - 1)

    # Load and prepare data
    coastal = load_coastal(coastal_path, proj_crs)
    grid = generate_grid(coastal, sinus_crs, step)

    # Convert to Dask GeoDataFrame
    logger.info("Converting grid to Dask GeoDataFrame with %d partitions", partitions)
    dask_grid: dgpd.GeoDataFrame = dgpd.from_geopandas(grid, npartitions=partitions)

    # Prepare buffer and coastal union
    buffer_radius = step * buffer_factor

    # Filter partitions
    logger.info("Filtering grid in parallel using prepared geometry…")
    filtered = dask_grid.map_partitions(
        filter_partition,
        coastal,
        buffer_radius,
        proj_crs,
        meta=dask_grid._meta,
    )
    logger.info("Computing filtered points")
    result = filtered.compute()
    logger.info("Filtered point count: %d", len(result))

    # Save output
    save_points(result, output_path)


if __name__ == "__main__":
    main()
