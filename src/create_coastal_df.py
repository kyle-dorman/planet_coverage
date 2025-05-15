#!/usr/bin/env python3
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path

import click
import geopandas as gpd
import pandas as pd
import polars as pl
from shapely import wkb
from tqdm import tqdm

from src.config import Instrument, ItemType, PublishingStage, QualityCategory
from src.query_udms import DataFrameRow

logger = logging.getLogger(__name__)

# build the glob pattern once
GLOB_PATTERN = "*/*/*/*/data.parquet"

SCHEMA = {
    "id": pl.Utf8,
    "acquired": pl.Datetime,
    "item_type": pl.Enum(ItemType),
    "satellite_id": pl.Utf8,
    "instrument": pl.Enum(Instrument),
    "cell_id": pl.UInt32,
    "grid_id": pl.UInt32,
    "has_8_channel": pl.Boolean,
    "has_sr_asset": pl.Boolean,
    "clear_percent": pl.Float32,
    "quality_category": pl.Enum(QualityCategory),
    "ground_control": pl.Boolean,
    "publishing_stage": pl.Enum(PublishingStage),
    "satellite_azimuth": pl.Float32,
    "sun_azimuth": pl.Float32,
    "sun_elevation": pl.Float32,
    "view_angle": pl.Float32,
    "geometry_wkb": pl.Binary,
    "coverage_pct": pl.Float32,
}


def get_save_path(base: Path, index: int) -> Path:
    hex_id = f"{index:06x}"  # unique 6‑digit hex, e.g. '0f1a2b'
    d1, d2, d3 = hex_id[:2], hex_id[2:4], hex_id[4:6]
    save_path = base / d1 / d2 / d3
    return save_path


# helper so we can use imap_unordered and keep tqdm progress
def _star_compute(args):
    return process_file(*args)


def process_file(grid_gdf: gpd.GeoDataFrame, lazy_df: pl.LazyFrame, out_dir: Path) -> None:
    """Process one data.parquet file for coastal points."""
    assert grid_gdf.crs is not None
    assert grid_gdf.poly_area is not None
    dest = out_dir / "coastal_points.parquet"
    if dest.exists():
        return

    count_df = lazy_df.select(pl.count().alias("n_rows")).collect()
    n_rows = count_df["n_rows"][0]
    if n_rows == 0:
        return

    # load the parquet into pandas to rebuild geometries
    df_pd: pd.DataFrame = lazy_df.collect().to_pandas()
    df_pd["geometry"] = df_pd["geometry_wkb"].apply(wkb.loads)  # type: ignore
    df_pd = df_pd.drop(columns=["geometry_wkb"])
    satellite_gdf = gpd.GeoDataFrame(df_pd, geometry="geometry", crs="EPSG:4326").to_crs(grid_gdf.crs)

    # Verify all geometries are valid
    assert satellite_gdf.geometry.is_valid.all(), "Found invalid geometries!"

    # --- find intersection of all grids and downloaded satellite captures

    # intersect image footprints with grid polygons
    joined = gpd.overlay(grid_gdf[["grid_id", "geometry", "poly_area"]], satellite_gdf, how="intersection")

    # remove any duplicate captures per grid_id/id combination
    joined = joined.drop_duplicates(subset=["grid_id", "id"])

    if joined.empty:
        return

    # Add percent of input geometry that is covered by satellite capture
    joined["coverage_pct"] = joined.geometry.area / joined["poly_area"]

    # Map back to Planet crs
    joined = joined.to_crs("EPSG:4326")

    # Convert point geometry to wkb for saving (overwriting polygon wkb)
    joined["geometry_wkb"] = joined.geometry.map(lambda geom: geom.wkb)

    # joined now has all columns from gdf + a `grid_id` and the index of the matched pt
    # Drop the extra index column that sjoin adds and the point geometry column
    joined = joined.drop(columns=["poly_area", "geometry"])

    # Convert to Polars (or pandas) to write out
    pl_out = pl.from_pandas(joined, schema_overrides=SCHEMA, include_index=False)

    out_dir.mkdir(exist_ok=True, parents=True)
    pl_out.write_parquet(dest)


@click.command()
@click.argument("base_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--query-grids-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="GeoPackage of polygons use for planet querying (index must match cell_id)",
)
@click.option(
    "--coastal-grids-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="GeoPackage of polygons to high res analysis",
)
@click.option(
    "--save-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Base directory to save results to",
)
@click.option(
    "--chunk-size",
    "-n",
    type=int,
    default=100,
    show_default=True,
    help="Number of distinct grid_ids to process per chunk",
)
def main(
    base_dir: Path,
    query_grids_path: Path,
    coastal_grids_path: Path,
    save_dir: Path,
    chunk_size: int,
):
    """
    Match coastal grid polygons to PlanetScope image footprints,
    compute their intersections, and save matched image metadata
    per coastal chunk.

    This script loads coastal grid cells and PlanetScope query cells,
    assigns each coastal cell to a valid query cell (via overlap or nearest neighbor),
    filters image data by those cells, intersects them with the coastal grids,
    filters for minimum area coverage, and saves all matched points
    (including geometry and metadata) to partitioned Parquet files.

    Run in parallel, grouped by batches of grid_ids.
    """
    files = list(base_dir.glob(GLOB_PATTERN))
    if not len(files):
        logger.error("No data.parquet files found under %s", base_dir)
        return

    # define the global scan up front (only once)
    all_lazy = pl.scan_parquet(
        f"{base_dir}/{GLOB_PATTERN}",
        schema=DataFrameRow.polars_schema(),
    )

    if save_dir.exists():
        logger.error(f"save_dir {save_dir} exists!")
        return

    logger.info("Loading query grids from %s", query_grids_path)
    gdf_cells = gpd.read_file(query_grids_path)
    assert gdf_cells.crs is not None

    logger.info("Loading coastal grids from %s", coastal_grids_path)
    gdf_coastal = gpd.read_file(coastal_grids_path).rename(columns={"cell_id": "grid_id"})
    gdf_coastal["poly_area"] = gdf_coastal.geometry.area

    logger.info("Finding query grid and coastal grid intersection")
    assert gdf_coastal.crs == gdf_cells.crs, "GDFs must be in the same CRS"

    # initial spatial join: match coastal grids to query cells by containment
    joined = gpd.sjoin(
        left_df=gdf_coastal[["grid_id", "geometry"]],
        right_df=gdf_cells[["cell_id", "geometry"]],
        how="left",
    )

    # separate those with overlap from those without
    joined_overlap = joined[joined["cell_id"].notnull()].copy()
    joined_missing = joined[joined["cell_id"].isnull()]

    # for any missing, assign nearest cell
    if not joined_missing.empty:
        logger.info(f"{len(joined_missing)} coastal grids lack overlap; assigning nearest cell")
        nearest = gpd.sjoin_nearest(
            joined_missing[["grid_id", "geometry"]],
            gdf_cells[["cell_id", "geometry"]],
            how="left",
        )
        # combine overlap and nearest
        joined = pd.concat([joined_overlap, nearest], ignore_index=True)
    else:
        joined = joined_overlap

    # filter to just cell_ids with data
    valid_cell_ids = all_lazy.select(pl.col("cell_id").unique().sort()).collect().to_series().to_list()
    joined = joined[joined.cell_id.isin(valid_cell_ids)]

    logger.info(
        f"""
        Found {len(joined)} intersections between
        {len(gdf_coastal)} coastal grids,
        {len(gdf_cells)} total query grids and
        {len(valid_cell_ids)} valid query grids."""
    )
    grid_ids = sorted(joined.grid_id.unique())

    idxes = list(range(0, len(grid_ids), chunk_size))
    logger.info(f"Creating {len(idxes)} tasks for processing")
    tasks = []
    for idx in idxes:
        stop = idx + chunk_size
        batch_grid_idxes = grid_ids[idx:stop]
        # select cell_ids via grid_id lookup
        cell_ids = joined[joined.grid_id.isin(batch_grid_idxes)].cell_id.unique()
        lazy_df = all_lazy.filter(pl.col("cell_id").is_in(cell_ids))
        # select coastal rows via index lookup, then restore grid_id as column
        coastal_batch = gdf_coastal[gdf_coastal.grid_id.isin(batch_grid_idxes)].reset_index()
        # create tasks to run
        tasks.append((coastal_batch, lazy_df, get_save_path(save_dir, idx)))

    # Run in parallel
    logger.info("Running tasks in parallel")
    num_workers = max(1, cpu_count() - 1)
    with Pool(num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(_star_compute, tasks), total=len(tasks), desc="Processing tasks"):
            pass


if __name__ == "__main__":
    # configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

    main()
