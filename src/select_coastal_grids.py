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
from scipy.spatial import cKDTree
from shapely import MultiLineString
from shapely.ops import linemerge
from shapely.prepared import prep

from src.gen_points_map import compute_step, make_equal_area_grid
from src.geo_util import load_coastal, preprocess_geometry

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
    "--antarctica",
    "antarctica_geojson",
    type=click.Path(exists=True),
    required=True,
    help="Path to antarctica GeoJson",
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
    antarctica_geojson: str,
    sinus_crs: str,
    degrees: float,
    pre_simplify_tol: float,
    partitions: int,
) -> None:
    # Determine partitions
    partitions = min(partitions, mp.cpu_count() - 1)

    antarctica = gpd.read_file(antarctica_geojson)
    assert antarctica.crs is not None

    t0 = time.perf_counter()
    logger.info("Reading mainlands GeoPackage %s", mainlands_gpkg)
    mldf = gpd.read_file(mainlands_gpkg)
    mldf = preprocess_geometry(mldf, proj_crs=sinus_crs, tol=pre_simplify_tol)
    logger.info(
        "Loaded & preprocessed %d mainland polygons in %.2f s",
        len(mldf),
        time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    logger.info("Reading big‑islands GeoPackage %s", big_islands_gpkg)
    bidf = gpd.read_file(big_islands_gpkg)
    bidf = preprocess_geometry(bidf, proj_crs=sinus_crs, tol=pre_simplify_tol)
    logger.info(
        "Loaded & preprocessed %d big‑island polygons in %.2f s",
        len(bidf),
        time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    logger.info("Reading big‑islands GeoPackage %s", big_islands_gpkg)
    sidf = gpd.read_file(small_islands_gpkg)
    sidf = preprocess_geometry(sidf, proj_crs=sinus_crs, tol=pre_simplify_tol)
    logger.info(
        "Loaded & preprocessed %d small‑island polygons in %.2f s",
        len(bidf),
        time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    logger.info("Building unified land geometry …")
    land_union = gpd.GeoSeries(
        pd.concat([mldf.geometry, bidf.geometry], ignore_index=True),
        crs=sinus_crs,
    ).union_all()
    simp_land_union = land_union.simplify(500, preserve_topology=True)
    coast_lines: MultiLineString = linemerge(simp_land_union.boundary)  # type: ignore
    coast_pts = []
    for coast_line in coast_lines.geoms:
        coast_pts.extend(coast_line.coords)
    coast_xy = np.array([(p[0], p[1]) for p in coast_pts])
    tree = cKDTree(coast_xy)
    logger.info("Land union built in %.2f s", time.perf_counter() - t0)

    # Load and prepare data
    coastal = load_coastal(coastal_path, sinus_crs)
    cell_size_m = compute_step(degrees)
    _, box_grid = make_equal_area_grid(cell_size_m, sinus_crs)

    # Convert to Dask GeoDataFrame
    t0 = time.perf_counter()
    logger.info("Converting grid to Dask GeoDataFrame with %d partitions", partitions)
    dask_grid: dgpd.GeoDataFrame = dgpd.from_geopandas(box_grid, npartitions=partitions)
    logger.info("Converted to Dask GeoDataFrame in %.2f s", time.perf_counter() - t0)

    # Filter partitions
    t0 = time.perf_counter()
    logger.info("Filtering grid in parallel using prepared geometry…")
    filtered = dask_grid.map_partitions(
        filter_partition,
        coastal,
        meta=dask_grid._meta,
    )
    logger.info("Computing filtered points")
    coastal_grids: gpd.GeoDataFrame = filtered.compute()
    logger.info("Partition filtering + compute finished in %.2f s", time.perf_counter() - t0)
    logger.info("Filtered point count: %d", len(coastal_grids))

    logger.info("Finding Antartic Grids")
    antarctica_grids = box_grid.loc[
        box_grid.to_crs(antarctica.crs).intersects(antarctica.geometry.union_all(method="coverage"))
        & ~box_grid.index.isin(coastal_grids.index)
    ].copy()
    logger.info(f"Found {len(antarctica_grids)} Antartic Grids")

    # ------------------------------------------------------------------
    #  Compute land flag + distance in parallel with Dask
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    # pre-compute centroids array
    centers = coastal_grids.geometry.centroid
    cent_xy = np.column_stack((centers.x.values, centers.y.values))  # type: ignore

    # nearest neighbour query  (returns (distance, index))
    dists, _ = tree.query(cent_xy, workers=-1)  # parallel threads, SciPy ≥1.9

    sinter = coastal_grids.sjoin(sidf, how="inner", predicate="within").cell_id.unique()
    binter = coastal_grids.sjoin(bidf, how="inner", predicate="within").cell_id.unique()
    minter = coastal_grids.sjoin(mldf, how="inner", predicate="within").cell_id.unique()
    land_ids = np.unique(np.concatenate([sinter, binter, minter]))

    pts_df = coastal_grids.copy()
    pts_df.geometry = pts_df.geometry.centroid
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

    pts_df["dist_km"] = dists / 1000
    pts_df["sdist_km"] = dists.max() / 1000

    pts_df.loc[snearest.index, "sdist_km"] = snearest.dist / 1000

    coastal_grids["dist_km"] = pts_df[["dist_km", "sdist_km"]].min(axis=1)
    coastal_grids["is_land"] = coastal_grids.cell_id.isin(land_ids)
    coastal_grids.loc[coastal_grids.is_land, "dist_km"] = 0.0
    coastal_grids.loc[coastal_grids.dist_km > 50, "dist_km"] = 50

    antarctica_grids["dist_km"] = np.nan
    antarctica_grids["is_land"] = False

    coastal_grids_df = pd.concat([coastal_grids, antarctica_grids], ignore_index=False)
    coastal_grids = gpd.GeoDataFrame(coastal_grids_df, geometry="geometry", crs=sinus_crs)

    logger.info(
        "Land flag + distance computed for %d rows in %.2f s",
        len(coastal_grids),
        time.perf_counter() - t0,
    )

    # Save output
    save_points(coastal_grids, output_path)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)

    main()
