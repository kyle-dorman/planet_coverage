#!/usr/bin/env python3
"""
Generate a regular Sinusoidal grid of polygons, retain only those
that intersect coastal strips.
"""

import logging
import multiprocessing as mp

import click
import dask_geopandas as dgpd
import geopandas as gpd
from shapely.prepared import prep

from src.gen_points_map import compute_step, make_equal_area_grid
from src.geo_util import load_coastal

logger = logging.getLogger(__name__)


def filter_partition(box_grid: gpd.GeoDataFrame, coastal: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Keep those intersecting the coastal union.
    """
    assert box_grid.crs == coastal.crs

    # Prepare coastal union for fast intersects
    coastal_union = coastal.union_all(method="coverage")
    prep_coastal = prep(coastal_union)
    mask = box_grid.geometry.apply(lambda geom: prep_coastal.intersects(geom))
    return box_grid.loc[mask]


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
@click.option(
    "--degrees",
    type=float,
    default=0.05,
    show_default=True,
    help="Grid spacing in degrees (default 0.05° at the equator)",
)
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
    degrees: float,
    partitions: int,
) -> None:
    # Determine partitions
    partitions = min(partitions, mp.cpu_count() - 1)

    # Load and prepare data
    coastal = load_coastal(coastal_path, sinus_crs)
    cell_size_m = compute_step(degrees)
    _, box_grid = make_equal_area_grid(cell_size_m, sinus_crs)

    # Convert to Dask GeoDataFrame
    logger.info("Converting grid to Dask GeoDataFrame with %d partitions", partitions)
    dask_grid: dgpd.GeoDataFrame = dgpd.from_geopandas(box_grid, npartitions=partitions)

    # Filter partitions
    logger.info("Filtering grid in parallel using prepared geometry…")
    filtered = dask_grid.map_partitions(
        filter_partition,
        coastal,
        meta=dask_grid._meta,
    )
    logger.info("Computing filtered points")
    result = filtered.compute()
    logger.info("Filtered point count: %d", len(result))

    # Save output
    save_points(result, output_path)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)

    main()
