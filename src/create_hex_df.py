#!/usr/bin/env python3
import logging
import warnings
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

# Mute known Shapely intersection warning (invalid geometry overlaps)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in intersection",
    category=RuntimeWarning,
    module="shapely",
)

DEFAULT_NUM_PROCS = max(1, cpu_count() - 1)

logger = logging.getLogger(__name__)

SCHEMA = {
    "id": pl.Utf8,
    "acquired": pl.Datetime,
    "item_type": pl.Enum(ItemType),
    "satellite_id": pl.Utf8,
    "instrument": pl.Enum(Instrument),
    "cell_id": pl.UInt32,
    "query_cell_id": pl.UInt32,
    "hex_id": pl.UInt32,
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
}


def is_ocean_year(pth: Path) -> bool:
    year = int(pth.parents[3].stem)
    return year > 2022


def get_save_path(base: Path, index: int) -> Path:
    hex_id = f"{index:06x}"  # unique 6-digit hex, e.g. '0f1a2b'
    d1, d2, d3 = hex_id[:2], hex_id[2:4], hex_id[4:6]
    save_path = base / d1 / d2 / d3
    return save_path


def join_query_cells_to_grid(query_gdf: gpd.GeoDataFrame, grid_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # initial spatial join: match hex grids to query cells by containment
    joined = gpd.sjoin(
        left_df=grid_gdf[["hex_id", "geometry"]],
        right_df=query_gdf[["cell_id", "geometry"]],
        how="left",
    )

    # separate those with overlap from those without
    joined_overlap = joined[joined["cell_id"].notnull()].copy()
    joined_missing = joined[joined["cell_id"].isnull()]

    # for any missing, assign nearest cell
    if not joined_missing.empty:
        logger.info(f"{len(joined_missing)} hex grids lack overlap; assigning nearest cell")
        nearest = gpd.sjoin_nearest(
            joined_missing[["hex_id", "geometry"]],
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
    grid_gdf: gpd.GeoDataFrame,
    planet_df: pl.LazyFrame,
    out_dir: Path,
) -> None:
    """Process one data.parquet file for hex points."""
    assert grid_gdf.crs is not None
    dest = out_dir / "hex_points.parquet"

    if dest.exists():
        return

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
    satellite_gdf["sat_area"] = satellite_gdf.geometry.area
    joined = gpd.overlay(grid_gdf[["hex_id", "geometry"]], satellite_gdf, how="intersection")

    # remove any duplicate captures per hex_id/id combination
    joined = joined.drop_duplicates(subset=["hex_id", "id"]).rename(columns={"cell_id": "query_cell_id"})

    if joined.empty:
        return

    # Filter to at least 50% of sat capture in grid
    joined["coverage_pct"] = joined.geometry.area / joined.sat_area
    joined = joined[joined.coverage_pct > 0.5]

    # joined now has all columns from gdf + a `hex_id` and the index of the matched pt
    # Drop the extra index column that sjoin adds and the geometry column
    joined = joined.drop(columns=["geometry", "sat_area", "coverage_pct"])

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
    "--hex-grids-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="GeoPackage of hex polygons",
)
@click.option(
    "--save-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Base directory to save results to",
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
    hex_grids_path: Path,
    save_dir: Path,
    num_procs: int,
):
    """
    Match hex grid polygons to PlanetScope image footprints,
    compute their intersections, and save matched image metadata
    per hex chunk.

    This script loads hex grid cells and PlanetScope query cells,
    assigns each hex cell to a valid query cell (via overlap or nearest neighbor),
    filters image data by those cells, intersects them with the hex grids,
    and saves all matched points (including geometry and metadata) to partitioned Parquet files.

    Run in parallel, grouped by batches of hex ids.

    The --num-procs option controls how many worker processes are used.
    """
    # Restrict to years >= 2022 (layout: year/*/*/*/data.parquet)
    min_year = 2022
    year_dirs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.isdigit() and int(p.name) >= min_year]
    year_dirs = sorted(year_dirs, key=lambda p: int(p.name))

    if not year_dirs:
        logger.error("No year directories >= %d found under %s", min_year, base_dir)
        return

    files = []
    for yd in year_dirs:
        files.extend(list(yd.glob("*/*/*/data.parquet")))

    if not files:
        logger.error("No data.parquet files found for years >= %d under %s", min_year, base_dir)
        return

    # define the global scan up front (only once, scanning only selected years)
    scan_paths = [str(yd / "*/*/*/data.parquet") for yd in year_dirs]
    all_lazy = pl.scan_parquet(
        scan_paths,
        schema=DataFrameRow.polars_schema(),
    )

    sinus_crs = "ESRI:54008"

    logger.info("Loading query grids from %s", query_grids_path)
    query_gdf = gpd.read_file(query_grids_path)
    assert query_gdf.crs is not None
    query_gdf = query_gdf.to_crs(sinus_crs)

    logger.info("Loading hex grids from %s", hex_grids_path)
    gdf_hex = gpd.read_file(hex_grids_path).to_crs(sinus_crs)[["hex_id", "geometry"]]

    logger.info("Finding query grid and hex grid intersection")
    assert gdf_hex.crs == query_gdf.crs, "GDFs must be in the same CRS"

    joined = join_query_cells_to_grid(query_gdf, gdf_hex).set_index("cell_id")
    gdf_hex = gdf_hex.set_index("hex_id")

    # filter to just cell_ids that both appear in the data *and* have hex overlap
    valid_cell_ids = all_lazy.select(pl.col("cell_id").unique().sort()).collect().to_series().to_list()
    valid_cell_ids = [cid for cid in valid_cell_ids if cid in joined.index]

    # restrict to those IDs; since we filtered, no KeyError will arise
    joined = joined.loc[valid_cell_ids].reset_index().set_index("hex_id")

    logger.info(f"""
        Found {len(joined)} intersections between
        {len(gdf_hex)} hex grids,
        {len(query_gdf)} total query grids and
        {len(valid_cell_ids)} valid query grids.""")

    logger.info("Creating task batches grouped by cell_id)")
    tasks = []
    to_run = joined.reset_index().sort_values(by=["cell_id", "hex_id"]).groupby("cell_id").hex_id.apply(list)
    seen = set()
    for cell_id, hex_ids in tqdm(to_run.items(), total=len(to_run)):
        cell_id = int(cell_id)  # type: ignore
        hex_ids = [gid for gid in hex_ids if gid not in seen]
        seen.update(hex_ids)
        # select cell_ids via hex_id lookup
        cell_ids = joined.loc[hex_ids].cell_id.unique()
        lazy_df = all_lazy.filter(pl.col("cell_id").is_in(cell_ids))
        # select hex rows via index lookup
        hex_batch = gdf_hex.loc[hex_ids]
        # create tasks to run
        tasks.append(
            (
                hex_batch.reset_index(),
                lazy_df,
                get_save_path(save_dir, cell_id),
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
