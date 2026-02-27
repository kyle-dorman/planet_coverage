#!/usr/bin/env python3
import logging
import time

import click
import geopandas as gpd
import numpy as np

from src.gen_points_map import compute_step, make_equal_area_hex_grid

logger = logging.getLogger(__name__)


def save_points(df: gpd.GeoDataFrame, output_path: str) -> None:
    """
    Save filtered points to GeoPackage.
    """
    logger.info("Saving %d points to %s", len(df), output_path)
    df.to_file(output_path, driver="GPKG")


@click.command()
@click.option("--output-path", "-o", type=click.Path(), required=True, help="Output GeoPackage for hex grid")
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
@click.option("--sinus-crs", default="ESRI:54008", show_default=True, help="Sinusoidal CRS code for grid")
@click.option(
    "--degrees",
    type=float,
    default=1.0,
    show_default=True,
    help="Grid spacing in degrees",
)
@click.option(
    "--pre-simplify-tol",
    type=float,
    default=50,
    show_default=True,
    help="Tolerance for pre-buffer geometry simplification (m)",
)
def main(
    output_path: str,
    mainlands_gpkg: str,
    big_islands_gpkg: str,
    small_islands_gpkg: str,
    sinus_crs: str,
    degrees: float,
    pre_simplify_tol: float,
) -> None:
    robinson_crs = "ESRI:54030"

    t0 = time.perf_counter()
    logger.info("Building Hex Cells …")
    cell_size_m = compute_step(degrees)
    hex_grid = make_equal_area_hex_grid(cell_size_m, robinson_crs).rename(columns={"cell_id": "hex_id"})
    hex_grid_sinus = hex_grid.to_crs(sinus_crs)
    assert hex_grid_sinus.geometry.is_valid.all()
    logger.info("Hex Cells built in %.2f s", time.perf_counter() - t0)

    t0 = time.perf_counter()
    logger.info("Reading mainlands GeoPackage %s", mainlands_gpkg)
    mldf = gpd.read_file(mainlands_gpkg)
    mldf.geometry = mldf.buffer(0)
    mldf = mldf.to_crs(sinus_crs)
    mldf.geometry = mldf.geometry.simplify(pre_simplify_tol, preserve_topology=True)
    invalid = ~mldf.is_valid
    mldf.loc[invalid, "geometry"] = mldf.geometry[invalid].buffer(0)  # type: ignore
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
    bidf.loc[invalid, "geometry"] = bidf.geometry[invalid].buffer(0)  # type: ignore
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
    sidf.loc[invalid, "geometry"] = sidf.geometry[invalid].buffer(0)  # type: ignore
    logger.info(
        "Loaded & preprocessed %d small-island polygons in %.2f s",
        len(sidf),
        time.perf_counter() - t0,
    )

    # ------------------------------------------------------------------
    #  Compute land flag
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("Computing Land Grid Overlaps …")
    sinter = hex_grid_sinus.sjoin(sidf, how="inner", predicate="within").hex_id.unique()
    binter = hex_grid_sinus.sjoin(bidf, how="inner", predicate="within").hex_id.unique()
    minter = hex_grid_sinus.sjoin(mldf, how="inner", predicate="within").hex_id.unique()
    land_ids = np.unique(np.concatenate([sinter, binter, minter]))
    logger.info("Coastal Grid - Land Overlap calculated in %.2f s", time.perf_counter() - t0)

    hex_grid["is_land"] = hex_grid.hex_id.isin(land_ids)

    logger.info(
        "Land flag computed for %d rows in %.2f s",
        len(hex_grid),
        time.perf_counter() - t0,
    )

    # Save output
    save_points(hex_grid, output_path)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)

    main()
