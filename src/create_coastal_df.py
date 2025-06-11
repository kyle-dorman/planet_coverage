#!/usr/bin/env python3
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
from shapely import wkb
from tqdm import tqdm

from src.config import Instrument, ItemType, PublishingStage, QualityCategory
from src.geo_util import assign_intersection_id
from src.query_udms import DataFrameRow
from src.tides import tide_model

DEFAULT_NUM_PROCS = max(1, cpu_count() - 1)

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
    "query_cell_id": pl.UInt32,
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
    "intersects_grid_centroid": pl.Boolean,
    "coverage_pct": pl.Float32,
    "tide_height": pl.Float32,
    "tide_height_bin": pl.Int32,
    "is_mid_tide": pl.Boolean,
    "has_tide_data": pl.Boolean,
}


def get_save_path(base: Path, index: int) -> Path:
    hex_id = f"{index:06x}"  # unique 6‑digit hex, e.g. '0f1a2b'
    d1, d2, d3 = hex_id[:2], hex_id[2:4], hex_id[4:6]
    save_path = base / d1 / d2 / d3
    return save_path


def join_query_cells_to_grid(query_gdf: gpd.GeoDataFrame, grid_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # initial spatial join: match coastal grids to query cells by containment
    joined = gpd.sjoin(
        left_df=grid_gdf[["grid_id", "geometry"]],
        right_df=query_gdf[["cell_id", "geometry"]],
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
            query_gdf[["cell_id", "geometry"]],
            how="left",
        )
        # combine overlap and nearest
        joined = pd.concat([joined_overlap, nearest], ignore_index=True)
        joined = gpd.GeoDataFrame(joined, geometry="geometry")
    else:
        joined = joined_overlap

    joined.loc[joined.cell_id.isna(), "cell_id"] = -1
    joined["cell_id"] = joined["cell_id"].astype(int)

    return joined


# helper so we can use imap_unordered and keep tqdm progress
def _star_compute(args):
    return process_file(*args)


def process_file(
    cell_geom_gdf: gpd.GeoDataFrame,
    grid_gdf: gpd.GeoDataFrame,
    planet_df: pl.LazyFrame,
    height_edges_df: pd.DataFrame,
    out_dir: Path,
    tide_data_dir: Path,
    mid_tide_height: float,
    coast_tidal_grid_mapper: pd.DataFrame,
) -> None:
    """Process one data.parquet file for coastal points."""
    assert grid_gdf.crs is not None
    assert grid_gdf.poly_area is not None
    assert grid_gdf.grid_center is not None
    dest = out_dir / "coastal_points.parquet"
    if dest.exists():
        import os

        os.remove(dest)
        # return

    count_df = planet_df.select(pl.len().alias("n_rows")).collect()
    n_rows = count_df["n_rows"][0]
    if n_rows == 0:
        return

    # load the parquet into pandas to rebuild geometries
    df_pd: pd.DataFrame = planet_df.collect().to_pandas()
    df_pd["geometry"] = df_pd["geometry_wkb"].apply(wkb.loads)  # type: ignore
    df_pd = df_pd.drop(columns=["geometry_wkb"])
    satellite_gdf = gpd.GeoDataFrame(df_pd, geometry="geometry", crs="EPSG:4326").to_crs(grid_gdf.crs)
    valid_rows = satellite_gdf.geometry.is_valid
    if not valid_rows.all():
        satellite_gdf = satellite_gdf[valid_rows]
        logger.warning(f"Found {(~valid_rows).sum()} invalid rows. Skipping them...")

    # --- find intersection of all grids and downloaded satellite captures

    # intersect image footprints with grid polygons
    joined = gpd.overlay(grid_gdf[["grid_id", "geometry", "poly_area"]], satellite_gdf, how="intersection")

    # remove any duplicate captures per grid_id/id combination
    joined = joined.drop_duplicates(subset=["grid_id", "id"]).rename(columns={"cell_id": "query_cell_id"})

    if joined.empty:
        return

    # Assign the a consistent cell_id regardless of where the data comes from
    joined = joined.set_index("grid_id").join(coast_tidal_grid_mapper, how="left").reset_index()
    assert not joined.cell_id.isna().any(), joined.cell_id.isna().sum()

    # Add tide information
    tm = tide_model(tide_data_dir, "GOT4.10", "GOT")
    assert tm is not None
    joined["tide_height"] = 0.0

    # Some cells might not have tide heuristic information
    joined["has_tide_data"] = joined.cell_id.isin(height_edges_df[~height_edges_df.height_edge_0.isna()].index)

    for cell_id, rows in joined.groupby("cell_id"):
        geom = cell_geom_gdf.loc[cell_id].geometry
        latlon = np.array([geom.y, geom.x])  # type: ignore
        try:
            tide_heights = tm.tide_elevations(latlon, [rows.acquired.to_numpy()])  # type: ignore
        except IndexError as e:
            logger.exception(e)
            logger.error(f"No closest point for cell_id {cell_id}")
            tide_heights = np.full((1, len(rows.acquired)), np.nan)
        joined.loc[rows.index, "tide_height"] = tide_heights[0]

    # Bin tide heights
    edge_arr = height_edges_df.loc[joined["cell_id"]].to_numpy()
    # searchsorted in one vectorised call
    joined["tide_height_bin"] = (np.sum(joined["tide_height"].to_numpy()[:, None] >= edge_arr, axis=1) - 1).astype(
        np.int32
    )
    # Add check for middle tide.
    joined["is_mid_tide"] = (joined.tide_height > -mid_tide_height) & (joined.tide_height < mid_tide_height)

    # Add percent of input geometry that is covered by satellite capture
    joined["coverage_pct"] = joined.geometry.area / joined["poly_area"]

    # Compute if a capture intersects a grid_cell centroid
    joined = joined.join(grid_gdf.set_index("grid_id").grid_center, on="grid_id", how="left")
    assert joined.grid_center.isna().sum() == 0
    joined["intersects_grid_centroid"] = joined.intersects(joined.grid_center)  # type: ignore

    # joined now has all columns from gdf + a `grid_id` and the index of the matched pt
    # Drop the extra index column that sjoin adds and the point geometry column
    joined = joined.drop(columns=["poly_area", "geometry", "grid_center"])

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
    "--tide-heuristics-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Parquet/CSV with tide height bin edges per cell",
)
@click.option(
    "--tide-data-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing tidal constituent files for tide_model",
)
@click.option(
    "--save-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Base directory to save results to",
)
@click.option(
    "--mid-tide-height",
    "-mth",
    type=float,
    default=0.5,
    show_default=True,
    help="Half-range (in meters) around mean tide used to flag ‘mid‑tide’ captures",
)
@click.option(
    "--num-procs",
    "-p",
    type=int,
    default=DEFAULT_NUM_PROCS,
    show_default=True,
    help="Number of worker processes for parallel execution",
)
def main(
    base_dir: Path,
    query_grids_path: Path,
    coastal_grids_path: Path,
    tide_heuristics_path: Path,
    tide_data_dir: Path,
    save_dir: Path,
    mid_tide_height: float,
    num_procs: int,
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

    The --mid-tide-height option controls the ± height window for the mid-tide flag.
    The --num-procs option controls how many worker processes are used.
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

    logger.info("Loading query grids from %s", query_grids_path)
    query_gdf = gpd.read_file(query_grids_path)
    all_cell_ids = query_gdf.cell_id.unique()
    assert query_gdf.crs is not None

    logger.info("Loading coastal grids from %s", coastal_grids_path)
    gdf_coastal = gpd.read_file(coastal_grids_path).rename(columns={"cell_id": "grid_id"})
    gdf_coastal["poly_area"] = gdf_coastal.geometry.area
    gdf_coastal["grid_center"] = gdf_coastal.geometry.centroid

    coast_tidal_grid_mapper = assign_intersection_id(
        gdf_coastal[["grid_id", "geometry"]],
        query_gdf[["cell_id", "geometry"]],
        "grid_id",
        "cell_id",
        include_closest=True,
    )[["grid_id", "cell_id"]].set_index("grid_id")

    logger.info("Loading tide heuristics from %s", tide_heuristics_path)
    if tide_heuristics_path.suffix.lower() == ".csv":
        tide_heuristics_df = pd.read_csv(tide_heuristics_path)
    else:
        tide_heuristics_df = pd.read_parquet(tide_heuristics_path)

    logger.info("Finding query grid and coastal grid intersection")
    assert gdf_coastal.crs == query_gdf.crs, "GDFs must be in the same CRS"

    joined = join_query_cells_to_grid(query_gdf, gdf_coastal).set_index("cell_id")
    gdf_coastal = gdf_coastal.set_index("grid_id")

    # filter to just cell_ids that both appear in the data *and* have coastal overlap
    valid_cell_ids = all_lazy.select(pl.col("cell_id").unique().sort()).collect().to_series().to_list()
    valid_cell_ids = [cid for cid in valid_cell_ids if cid in joined.index]

    # restrict to those IDs; since we filtered, no KeyError will arise
    joined = joined.loc[valid_cell_ids].reset_index().set_index("grid_id")

    logger.info(
        f"""
        Found {len(joined)} intersections between
        {len(gdf_coastal)} coastal grids,
        {len(query_gdf)} total query grids and
        {len(valid_cell_ids)} valid query grids."""
    )

    # Gather per cell tide height edges and geometry centroids
    query_gdf.geometry = query_gdf.centroid
    query_gdf = query_gdf.to_crs("EPSG:4326").set_index("cell_id")[["geometry"]]
    height_edges = tide_heuristics_df.set_index("cell_id").filter(like="height_edge_").reindex(labels=all_cell_ids)

    logger.info("Creating task batches grouped by cell_id)")
    tasks = []
    to_run = joined.reset_index().sort_values(by=["cell_id", "grid_id"]).groupby("cell_id").grid_id.apply(list)
    seen = set()
    for cell_id, grid_ids in tqdm(to_run.items(), total=len(to_run)):
        cell_id = int(cell_id)  # type: ignore
        grid_ids = [gid for gid in grid_ids if gid not in seen]
        seen.update(grid_ids)
        # select cell_ids via grid_id lookup
        cell_ids = joined.loc[grid_ids].cell_id.unique()
        lazy_df = all_lazy.filter(pl.col("cell_id").is_in(cell_ids))
        # select coastal rows via index lookup
        coastal_batch = gdf_coastal.loc[grid_ids]
        # create tasks to run
        tasks.append(
            (
                query_gdf,
                coastal_batch.reset_index(),
                lazy_df,
                height_edges,
                get_save_path(save_dir, cell_id),
                tide_data_dir,
                mid_tide_height,
                coast_tidal_grid_mapper,
            )
        )

    # Run in parallel
    logger.info(f"Running {len(tasks)} tasks in parallel with {num_procs} worker process(es)")
    num_workers = max(1, min(num_procs, cpu_count()))
    with Pool(num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(_star_compute, tasks), total=len(tasks), desc="Processing tasks"):
            pass


if __name__ == "__main__":
    # configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

    main()
