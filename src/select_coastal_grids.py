#!/usr/bin/env python3
"""
Generate a regular Sinusoidal grid of polygons, retain only those
that intersect coastal strips.
"""

import logging
import multiprocessing as mp
import time

import click
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree  # type: ignore
from shapely import MultiPolygon, get_coordinates
from shapely.prepared import prep

from src.gen_points_map import compute_step, make_equal_area_grid
from src.geo_util import load_coastal

logger = logging.getLogger(__name__)


SEGMENIZE_M = 1000
SIMPLIFY_2 = 100


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
    "--small-islands",
    "small_islands_gpkg",
    type=click.Path(exists=True),
    required=True,
    help="Path to small islands GeoPackage",
)
@click.option(
    "--ignore-region",
    "ignore_region_geojson",
    type=click.Path(exists=True),
    required=True,
    help="Path to the region to ignore.",
)
@click.option("--sinus-crs", default="ESRI:54008", show_default=True, help="Sinusoidal CRS code for grid")
@click.option(
    "--degrees",
    type=float,
    default=0.05,
    show_default=True,
    help="Grid spacing in degrees (default 0.05° at the equator)",
)
@click.option(
    "--pre-simplify-tol",
    type=float,
    default=50,
    show_default=True,
    help="Tolerance for pre-buffer geometry simplification (m)",
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
    mainlands_gpkg: str,
    big_islands_gpkg: str,
    small_islands_gpkg: str,
    ignore_region_geojson: str,
    sinus_crs: str,
    degrees: float,
    pre_simplify_tol: float,
    partitions: int,
) -> None:
    # Determine partitions
    partitions = min(partitions, mp.cpu_count() - 1)

    t0 = time.perf_counter()
    logger.info("Building Grid Cells …")
    cell_size_m = compute_step(degrees)
    box_grid = make_equal_area_grid(cell_size_m, sinus_crs)
    ignore_region = gpd.read_file(ignore_region_geojson)
    # Filter out the North East Russia area.
    box_grid = box_grid[~box_grid.to_crs(ignore_region.crs).within(ignore_region.geometry.iloc[0])]  # type: ignore
    # Convert to Dask GeoDataFrame
    dask_grid: dgpd.GeoDataFrame = dgpd.from_geopandas(box_grid, npartitions=partitions)
    logger.info("Grid Cells built in %.2f s", time.perf_counter() - t0)

    # Filter partitions
    t0 = time.perf_counter()
    logger.info("Filtering grid in parallel using prepared geometry…")
    # Load and prepare data
    coastal = load_coastal(coastal_path, sinus_crs)
    filtered = dask_grid.map_partitions(
        filter_partition,
        coastal,
        meta=dask_grid._meta,
    )
    logger.info("Computing filtered points")
    coastal_grids: gpd.GeoDataFrame = filtered.compute()
    logger.info("Partition filtering + compute finished in %.2f s", time.perf_counter() - t0)
    logger.info("Filtered point count: %d", len(coastal_grids))

    t0 = time.perf_counter()
    logger.info("Reading mainlands GeoPackage %s", mainlands_gpkg)
    mldf = gpd.read_file(mainlands_gpkg)
    mldf.geometry = mldf.buffer(0)
    mldf = mldf.to_crs(sinus_crs)
    mldf.geometry = mldf.geometry.simplify(pre_simplify_tol, preserve_topology=True)
    invalid = ~mldf.is_valid
    mldf.loc[invalid, "geometry"] = mldf.geometry[invalid].buffer(0)
    logger.info(
        "Loaded & preprocessed %d mainland polygons in %.2f s",
        len(mldf),
        time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    logger.info("Reading big-islands GeoPackage %s", big_islands_gpkg)
    bidf = gpd.read_file(big_islands_gpkg)
    bidf.geometry = bidf.buffer(0)
    bidf = bidf.to_crs(sinus_crs)
    bidf.geometry = bidf.geometry.simplify(pre_simplify_tol, preserve_topology=True)
    invalid = ~bidf.is_valid
    bidf.loc[invalid, "geometry"] = bidf.geometry[invalid].buffer(0)
    logger.info(
        "Loaded & preprocessed %d big-island polygons in %.2f s",
        len(bidf),
        time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    logger.info("Reading small-islands GeoPackage %s", big_islands_gpkg)
    sidf = gpd.read_file(small_islands_gpkg)
    sidf.geometry = sidf.buffer(0)
    sidf = sidf.to_crs(sinus_crs)
    sidf.geometry = sidf.geometry.simplify(pre_simplify_tol, preserve_topology=True)
    invalid = ~sidf.is_valid
    sidf.loc[invalid, "geometry"] = sidf.geometry[invalid].buffer(0)
    logger.info(
        "Loaded & preprocessed %d small-island polygons in %.2f s",
        len(sidf),
        time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    logger.info("Building unified land geometry …")
    land_union = gpd.GeoSeries(
        pd.concat([mldf.geometry, bidf.geometry], ignore_index=True),
        crs=sinus_crs,
    ).union_all()
    simp_land_union: MultiPolygon = land_union.simplify(SIMPLIFY_2, preserve_topology=True)  # type: ignore
    invalids = []
    valids = []
    for poly in simp_land_union.geoms:
        if not poly.is_valid:
            invalids.append(poly)
        else:
            valids.append(poly)
    invalid_df = gpd.GeoDataFrame(geometry=invalids, crs=sinus_crs)
    invalid_df.geometry = invalid_df.geometry.buffer(0)
    simp_land_union: MultiPolygon = MultiPolygon(valids + invalid_df.geometry.tolist()).buffer(0)  # type: ignore
    logger.info("Land union built in %.2f s", time.perf_counter() - t0)

    t0 = time.perf_counter()
    logger.info("Building Coastal Points Tree …")
    coast_pts = []
    for poly in simp_land_union.geoms:
        coast_pts.append(get_coordinates(poly.boundary.segmentize(SEGMENIZE_M)))
    coast_xy = np.concatenate(coast_pts)
    tree = cKDTree(coast_xy)
    logger.info("Coastal Points Tree built in %.2f s", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    #  Compute land flag + distance in parallel with Dask
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("Computing Coastal Grid Distances …")
    # pre-compute centroids array
    centers = coastal_grids.geometry.centroid
    cent_xy = np.column_stack((centers.x.values, centers.y.values))  # type: ignore

    # nearest neighbour query  (returns (distance, index))
    dists, _ = tree.query(cent_xy, workers=-1)  # parallel threads, SciPy ≥1.9
    logger.info("Coastal Grid Distances calculated in %.2f s", time.perf_counter() - t0)

    t0 = time.perf_counter()
    logger.info("Computing Land Grid Overlaps …")
    pts_df = coastal_grids.copy()
    pts_df.geometry = pts_df.geometry.centroid

    sinter = coastal_grids.sjoin(sidf, how="inner", predicate="within").cell_id.unique()
    binter = coastal_grids.sjoin(bidf, how="inner", predicate="within").cell_id.unique()
    minter = coastal_grids.sjoin(mldf, how="inner", predicate="within").cell_id.unique()
    land_ids = np.unique(np.concatenate([sinter, binter, minter]))
    logger.info("Coastal Grid - Land Overlap calculated in %.2f s", time.perf_counter() - t0)

    t0 = time.perf_counter()
    logger.info("Computing Coastal Grid Overlaps …")
    sinter = pts_df.sjoin(sidf, how="inner", predicate="within").cell_id.unique()
    binter = pts_df.sjoin(bidf, how="inner", predicate="within").cell_id.unique()
    minter = pts_df.sjoin(mldf, how="inner", predicate="within").cell_id.unique()
    coastal_ids = np.unique(np.concatenate([sinter, binter, minter]))
    logger.info("Coastal Grid - CoastLine Overlap calculated in %.2f s", time.perf_counter() - t0)

    t0 = time.perf_counter()
    logger.info("Computing Coastal Grid - Small Island Distances …")
    snearest = (
        gpd.sjoin_nearest(
            pts_df[["geometry", "cell_id"]],
            sidf[["geometry"]],
            how="inner",
            max_distance=60 * 1000,
            distance_col="dist",
        )
        .sort_values(by=["cell_id", "dist"])
        .drop_duplicates(subset=["cell_id"])
    )
    logger.info("Coastal Grid - Small Island Distances calculated in %.2f s", time.perf_counter() - t0)

    pts_df["dist_km"] = dists / 1000
    pts_df["sdist_km"] = dists.max() / 1000

    pts_df.loc[snearest.index, "sdist_km"] = snearest.dist / 1000

    coastal_grids["dist_km"] = pts_df[["dist_km", "sdist_km"]].min(axis=1)
    coastal_grids["is_land"] = coastal_grids.cell_id.isin(land_ids)
    coastal_grids["is_coast"] = coastal_grids.cell_id.isin(coastal_ids) & ~coastal_grids.is_land
    coastal_grids.loc[coastal_grids.is_land, "dist_km"] = 0.0
    coastal_grids.loc[coastal_grids.is_coast, "dist_km"] = 0.0
    coastal_grids.loc[coastal_grids.dist_km > 50, "dist_km"] = 50

    logger.info(
        "Land flag + distance computed for %d rows in %.2f s",
        len(coastal_grids),
        time.perf_counter() - t0,
    )

    # Save output
    save_points(coastal_grids, output_path)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)

    main()
