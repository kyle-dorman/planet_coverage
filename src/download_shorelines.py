import json
import logging
import multiprocessing as mp
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import click
import ee
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

# Create module-level logger
logger = logging.getLogger(__name__)


def download_batch(collection: str, offset: int, base_dir: Path, page_size: int):
    """
    Download a single batch of features for a given collection and offset.
    """
    save_dir = base_dir / collection
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"{offset}.geojson"
    if output_path.exists():
        return offset  # already downloaded

    asset_id = f"projects/sat-io/open-datasets/shoreline/{collection}"
    fc = ee.FeatureCollection(asset_id)
    batch = ee.FeatureCollection(fc.toList(page_size, offset)).getInfo()
    assert batch is not None
    batch_features = batch["features"]

    geojson = {"type": "FeatureCollection", "features": batch_features}
    with open(output_path, "w") as f:
        json.dump(geojson, f)


def load_and_trim(fp: Path, keep_cols: list[str] | None = None) -> gpd.GeoDataFrame:
    """
    Read a GeoJSON file and keep only specified columns plus geometry.
    """
    gdf = gpd.read_file(str(fp), engine="pyogrio")
    if keep_cols is not None:
        keep_cols = keep_cols + ["geometry"]
    else:
        keep_cols = ["geometry"]

    return gdf[keep_cols]  # type: ignore


def download_collections(
    temp_base: Path, collections: Iterable[str], page_size_dict: dict[str, int], workers: int
) -> None:
    tasks = []
    for coll in collections:
        asset_id = f"projects/sat-io/open-datasets/shoreline/{coll}"
        fc = ee.FeatureCollection(asset_id)
        total = fc.size().getInfo()
        assert total is not None
        total = int(total)
        logger.info(f"Collection '{coll}': total features = {total}")
        size = page_size_dict[coll]
        offsets = range(0, total, size)
        tasks.extend((coll, off) for off in offsets)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_batch, coll, off, temp_base, page_size_dict[coll]) for coll, off in tasks]
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()  # will raise on error
    logger.info("All downloads complete.")


@click.command()
@click.option("--base-dir", "-b", type=click.Path(), required=True, help="Base directory to save shorelines")
@click.option("--project", "-p", required=True, help="Google Earth Engine project name")
@click.option(
    "--collection",
    "-c",
    "collections",
    multiple=True,
    default=["mainlands", "big_islands", "small_islands"],
    show_default=True,
    help="Collections to download",
)
@click.option(
    "--page-size",
    "-s",
    "page_sizes",
    multiple=True,
    default=["mainlands=1", "big_islands=500", "small_islands=5000"],
    show_default=True,
    help="Page size per collection, format name=size",
)
@click.option(
    "--workers", "-w", type=int, default=mp.cpu_count() - 1, show_default=True, help="Number of parallel worker threads"
)
@click.option("--log-level", default="INFO", show_default=True, help="Logging level")
def main(
    base_dir: str,
    project: str,
    collections: tuple[str, ...],
    page_sizes: tuple[str, ...],
    workers: int,
    log_level: str,
) -> None:
    # Configure logging
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")
    logger.info(f"Starting download_shorelines workflow for collections {collections}")
    # Initialize Earth Engine
    ee.Initialize(project=project)

    # Parse and validate page sizes
    page_size_dict = {}
    for entry in page_sizes:
        name, size_str = entry.split("=", 1)
        page_size_dict[name] = int(size_str)
    missing = set(collections) - set(page_size_dict)
    if missing:
        raise click.UsageError(f'Missing page-size for collections: {", ".join(missing)}')

    base_path = Path(base_dir)

    keep_columns = ["Area_km2"]

    # Temporary workspace for JSONs
    with tempfile.TemporaryDirectory(prefix="shoreline_") as temp_dir:
        temp_base = Path(temp_dir)
        logger.info(f"Using temporary directory {temp_base} for JSON storage")

        # Pre-create directories for each collection in temp_base
        for coll in collections:
            (temp_base / coll).mkdir(parents=True, exist_ok=True)

        # Download all collections
        download_collections(temp_base, collections, page_size_dict, workers)

        # Load all JSONs per collection and write to Parquet
        for coll in collections:
            json_files = list((temp_base / coll).glob("*.geojson"))
            if not json_files:
                logger.warning(f"No JSON files for collection '{coll}'. Skipping.")
                continue
            df_list = []
            for fp in tqdm(json_files, desc=f"Loading JSONs for {coll}"):
                df_list.append(load_and_trim(fp, keep_columns))

            assert len(df_list)
            gdf = gpd.GeoDataFrame(pd.concat(df_list, ignore_index=True), crs=df_list[0].crs)

            out_path = base_path / f"{coll}.parquet"
            logger.info(f"Writing {len(gdf)} features for '{coll}' to {out_path}")
            gdf.to_parquet(path=out_path, engine="pyarrow")

    logger.info("Workflow complete")


if __name__ == "__main__":
    main()
