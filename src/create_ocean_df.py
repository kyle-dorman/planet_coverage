#!/usr/bin/env python3
import logging
import shutil
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import click
import polars as pl
from tqdm import tqdm

from src.geo_util import dedup_satellite_captures

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def process_file(src: Path, dedup: bool, max_duration_hrs: int) -> None:
    """Process one Parquet file: read, (optional dedupe), and write ocean.parquet."""
    dest = src.parent / "ocean.parquet"

    if dest.exists():
        logger.debug(f"Skipping {src}, {dest.name} already exists")
        return

    if not dedup:
        shutil.copy2(src, dest)
        logger.debug(f"Copied {src} to {dest} (no dedup mode)")
        return

    df_best = dedup_satellite_captures(pl.scan_parquet(src), max_duration_hrs, "cell_id")

    # 6) Write out
    df_best.write_parquet(dest)
    logger.debug(f"Wrote deduped file: {dest}")


@click.command()
@click.argument("base_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--no-dedup",
    is_flag=True,
    default=False,
    help="If set, skip deduplication and simply copy data.parquet to ocean.parquet. Used for SkySat",
)
@click.option(
    "--max-duration-hrs",
    type=int,
    default=4,
    show_default=True,
    help="Maximum duration in hours for deduplication threshold",
)
def main(base_dir: Path, no_dedup: bool, max_duration_hrs: int) -> None:
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
        func = partial(process_file, dedup=dedup, max_duration_hrs=max_duration_hrs)
        for _ in tqdm(pool.imap_unordered(func, files), total=len(files), desc="Processing files"):
            pass


if __name__ == "__main__":
    main()
