#!/usr/bin/env python3
"""
select_points.py

Generate a regular Sinusoidal grid, buffer points and retain only those
that intersect coastal strips.  Uses a prepared union for lightning-fast
intersection tests, Dask parallelism, and a live progress bar.
"""
import logging
import multiprocessing as mp
from functools import lru_cache

import click
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
from pyproj import Geod, Transformer
from shapely.prepared import prep

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def compute_step(degrees: float = 0.05) -> float:
    """
    Compute the east–west distance (in meters) at the equator
    corresponding to `degrees` of longitude on the WGS84 ellipsoid.
    Caches the result so repeated calls are free.
    """
    # Initialize a geodetic calculator for the WGS84 ellipsoid
    geod = Geod(ellps="WGS84")
    # Inverse geodetic between (lon0, lat0) and (lon0+degrees, lat0):
    # we pick lat0 = 0 to get equatorial distance
    lon0, lat0 = 0.0, 0.0
    lon1, lat1 = degrees, 0.0
    # https://pyproj4.github.io/pyproj/stable/api/geod.html
    _, _, distance_m = geod.inv(lon0, lat0, lon1, lat1)  # meters
    return distance_m


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


def generate_grid(sinus_crs: str, step: float) -> gpd.GeoDataFrame:
    """
    Generate a regular global grid in sinusoidal projection (ESRI:54008).
    """
    # Compute global bounds via pyproj at key latitudes/longitudes
    transformer = Transformer.from_crs("EPSG:4326", sinus_crs, always_xy=True)
    # Equator extremes for longitude ±180
    x_min, _ = transformer.transform(-180.0, 0.0)
    x_max, _ = transformer.transform(180.0, 0.0)
    # Pole extremes for latitude ±90 at central meridian
    _, y_min = transformer.transform(0.0, -90.0)
    _, y_max = transformer.transform(0.0, 90.0)
    # Ensure proper ordering
    minx, maxx = min(x_min, x_max), max(x_min, x_max)
    miny, maxy = min(y_min, y_max), max(y_min, y_max)

    xs = np.arange(minx, maxx + step, step)
    ys = np.arange(miny, maxy + step, step)
    logger.info("Generating global grid: %d cols × %d rows = %d points", len(xs), len(ys), xs.size * ys.size)
    xx, yy = np.meshgrid(xs, ys)
    grid = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()), crs=sinus_crs)
    return grid


def filter_partition(df: gpd.GeoDataFrame, coastal: gpd.GeoDataFrame, buffer_radius: float) -> gpd.GeoDataFrame:
    """
    Buffer points and keep those intersecting the coastal union.
    """
    assert df.crs == coastal.crs

    # Buffer points in projected CRS
    buffered = df.geometry.buffer(buffer_radius)
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
@click.option(
    "--step",
    type=float,
    default=compute_step(),
    show_default=True,
    help="Grid spacing in meters (computed from 0.05° at the equator)",
)
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
    step: float,
    buffer_factor: float,
    partitions: int,
) -> None:
    # Determine partitions
    partitions = min(partitions, mp.cpu_count() - 1)

    # Load and prepare data
    coastal = load_coastal(coastal_path, sinus_crs)
    grid = generate_grid(sinus_crs, step)

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
        meta=dask_grid._meta,
    )
    logger.info("Computing filtered points")
    result = filtered.compute()
    logger.info("Filtered point count: %d", len(result))

    # Save output
    save_points(result, output_path)


if __name__ == "__main__":
    main()
