#!/usr/bin/env python3
import logging
import shutil
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import click
import geopandas as gpd
import polars as pl
import shapely
from tqdm import tqdm

from src.config import Instrument, ItemType, PublishingStage, QualityCategory
from src.geo_util import dedup_satellite_captures, filter_satellite_pct_intersection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


SCHEMA = {
    "id": pl.Utf8,
    "acquired": pl.Datetime,
    "item_type": pl.Enum(ItemType),
    "satellite_id": pl.Utf8,
    "instrument": pl.Enum(Instrument),
    "cell_id": pl.UInt32,
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
}


def process_file(src: Path, query_grids_path: Path, dedup: bool, max_duration_hrs: int, pct_overlap: float) -> None:
    """Process one Parquet file: read, (optional dedupe), and write ocean.parquet."""
    dest = src.parent / "ocean.parquet"

    # if dest.exists():
    #     logger.debug(f"Skipping {src}, {dest.name} already exists")
    #     return

    lazy_df = pl.scan_parquet(src, schema=SCHEMA)
    count_df = lazy_df.select(pl.count().alias("n_rows")).collect()
    n_rows = count_df["n_rows"][0]

    if not dedup or n_rows == 0:
        shutil.copy2(src, dest)
        logger.debug(f"Copied {src} to {dest} (no dedup mode)")
        return

    gdf_cells = gpd.read_file(query_grids_path)
    # add hex identifier to polygons for matching
    gdf_cells["hex_id"] = gdf_cells["cell_id"].apply(lambda x: f"{x:06x}")

    d3 = src.parent.name
    d2 = src.parent.parent.name
    d1 = src.parent.parent.parent.name
    hex_str = f"{d1}{d2}{d3}"

    # get the polygon by hex_id
    match = gdf_cells[gdf_cells.hex_id == hex_str]
    assert not match.empty, f"Hex {hex_str} not found in polygons, skipping"
    best_crs = match.estimate_utm_crs()
    poly = match.to_crs(best_crs).iloc[0].geometry
    assert shapely.is_valid(poly)

    df = filter_satellite_pct_intersection(
        lazy_df=lazy_df,
        polygon=poly,
        crs=best_crs,
        overlap_pct=pct_overlap,
    )

    df_best = dedup_satellite_captures(
        lazy_df=df.lazy(),
        max_duration_hrs=max_duration_hrs,
        column_name="cell_id",
    )

    # 6) Write out
    df_best.write_parquet(dest)
    logger.debug(f"Wrote deduped file: {dest}")


@click.command()
@click.argument("base_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--query-grids-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="GeoPackage of polygons use for planet querying (index must match cell_id)",
)
@click.option(
    "--no-dedup",
    is_flag=True,
    default=False,
    help="If set, skip deduplication and simply copy data.parquet to ocean.parquet. Used for SkySat",
)
@click.option(
    "--max-duration-hrs",
    type=int,
    default=6,
    show_default=True,
    help="Maximum duration in hours for deduplication threshold",
)
@click.option(
    "--pct-overlap",
    type=float,
    default=0.1,
    show_default=True,
    help="Minimum pct of the satallite capture that must reside in the grid to be kelp",
)
def main(base_dir: Path, query_grids_path: Path, no_dedup: bool, max_duration_hrs: int, pct_overlap: float) -> None:
    """Find data.parquet files under base_dir and process them in parallel."""
    dedup = not no_dedup

    pattern = "*/*/*/*/data.parquet"
    files = list(base_dir.glob(pattern))
    if not files:
        logger.warning(f"No files found under {base_dir} matching {pattern}")
        return

    # Use a pool of workers to process files in parallel
    num_workers = max(1, cpu_count() - 1)
    with Pool(num_workers) as pool:
        func = partial(
            process_file,
            dedup=dedup,
            max_duration_hrs=max_duration_hrs,
            query_grids_path=query_grids_path,
            pct_overlap=pct_overlap,
        )
        for _ in tqdm(pool.imap_unordered(func, files), total=len(files), desc="Processing files"):
            pass


if __name__ == "__main__":
    main()
