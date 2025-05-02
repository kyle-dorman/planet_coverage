#!/usr/bin/env python3
"""
select_ocean_grids.py

Generate a regular polygon grid, retain only grids that are over the coastline or open ocean.
"""
import logging
import multiprocessing as mp

import click
import dask_geopandas as dgpd
import geopandas as gpd
import pandas as pd
from shapely.prepared import prep

from src.gen_points_map import compute_step, make_equal_area_grid
from src.geo_util import clean_invalid, preprocess_geometry

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_coastal(path: str, proj_crs: str) -> gpd.GeoDataFrame:
    """
    Load coastal strips GeoPackage and verify CRS.
    """
    logger.info("Loading coastal strips from %s", path)
    coastal = gpd.read_file(path).to_crs(proj_crs)
    assert coastal.crs is not None and coastal.crs == proj_crs
    assert coastal.geometry.is_valid.all()
    assert len(coastal) == 1, "No coastal features found"
    return coastal


def prepare_land(df: gpd.GeoDataFrame, simplify_tol) -> gpd.GeoDataFrame:
    assert df.crs is not None
    df = preprocess_geometry(df, df.crs, simplify_tol)  # type: ignore
    clean_invalid(df)

    return df


def filter_partition(
    grid_df: gpd.GeoDataFrame,
    coastal: gpd.GeoDataFrame,
    land: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Buffer point geometries by buffer_radius and keep those that intersect
    the coastal union or are outside land.
    """
    assert grid_df.crs == coastal.crs

    assert land.crs is not None
    wgs = grid_df.to_crs(land.crs)
    land_union = land.union_all()
    prep_land = prep(land_union)
    land_mask = wgs.geometry.apply(lambda geom: prep_land.intersects(geom))

    # Check where the polygons intersect the coast
    coastal_union = coastal.union_all("coverage")
    prep_coastal = prep(coastal_union)
    coastal_mask = grid_df.geometry.apply(lambda geom: prep_coastal.intersects(geom))

    mask = ~land_mask | coastal_mask
    return grid_df.loc[mask]


@click.command()
@click.option(
    "--mainlands", "mainlands_gpkg", type=click.Path(exists=True), required=True, help="Path to mainlands GeoPackage"
)
@click.option(
    "--big-islands",
    "big_islands_gpkg",
    type=click.Path(exists=True),
    required=True,
    help="Path to big islands GeoPackage",
)
@click.option(
    "--coastal-path", "-c", type=click.Path(exists=True), required=True, help="Path to coastal strips GeoPackage"
)
@click.option("--output-path", "-o", type=click.Path(), required=True, help="Output GeoPackage for filtered points")
@click.option("--crs", default="ESRI:54008", show_default=True, help="Sinusoidal CRS code for grid")
@click.option("--degree", type=float, default=1.0, show_default=True, help="Grid size in degrees")
@click.option(
    "--simplify-tol",
    type=float,
    default=50,
    show_default=True,
    help="Tolerance for shoreline geometry simplification (m)",
)
@click.option(
    "--partitions",
    type=int,
    default=mp.cpu_count() - 1,
    help="Number of Dask partitions; defaults to CPU count minus one",
)
def main(
    mainlands_gpkg: str,
    big_islands_gpkg: str,
    coastal_path: str,
    output_path: str,
    crs: str,
    degree: float,
    simplify_tol: float,
    partitions: int,
) -> None:
    # Determine partitions
    partitions = min(partitions, mp.cpu_count() - 1)

    # Load and prepare data
    logger.info("Loading mainlands")
    mldf = gpd.read_file(mainlands_gpkg)

    logger.info("Loading big islands")
    bidf = gpd.read_file(big_islands_gpkg)

    logger.info("Creating Dask GeoDataFrame")
    # Concatenate mainland and big-islands GeoDataFrames before parallel processing
    combined_gdf = pd.concat([mldf, bidf], ignore_index=True)
    dask_grid: dgpd.GeoDataFrame = dgpd.from_geopandas(combined_gdf, npartitions=partitions)
    dask_land = dask_grid.map_partitions(
        prepare_land,
        simplify_tol,
        meta=dask_grid._meta,
    )
    logger.info("Preparing land in parallel…")
    land = dask_land.compute()

    logger.info("Loading coastal strings")
    coastal = load_coastal(coastal_path, crs)

    logger.info("Creating Grid")
    cell_size_m = compute_step(degree)
    _, poly_grid = make_equal_area_grid(cell_size_m, crs)
    logger.info(f"Created grid with {len(poly_grid)} points")

    # Convert to Dask GeoDataFrame
    logger.info("Converting grid to Dask GeoDataFrame with %d partitions", partitions)
    dask_grid: dgpd.GeoDataFrame = dgpd.from_geopandas(poly_grid, npartitions=partitions)

    # Filter partitions
    logger.info("Filtering grid in parallel using prepared geometry…")
    filtered = dask_grid.map_partitions(
        filter_partition,
        coastal,
        land,
        meta=dask_grid._meta,
    )
    logger.info("Computing filtered cells")
    result = filtered.compute()
    logger.info("Filtered grid count: %d", len(result))

    # Save output
    result.to_file(output_path, driver="GPKG")


if __name__ == "__main__":
    main()
