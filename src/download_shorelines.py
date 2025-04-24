import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
from tqdm import tqdm

# Configuration
BASE_DIR = Path("/Users/kyledorman/data/shorelines")
COLLECTIONS = ["mainlands", "big_islands", "small_islands"]
PAGE_SIZE = {
    "mainlands": 1,
    "big_islands": 500,
    "small_islands": 5000,
}
MAX_WORKERS = 8  # adjust based on your machine

# Authenticate and initialize
ee.Initialize(project="coastal-base")


def download_batch(collection: str, offset: int):
    """
    Download a single batch of features for a given collection and offset.
    """
    save_dir = BASE_DIR / collection
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"{offset}.geojson"
    if output_path.exists():
        return offset  # already downloaded

    asset_id = f"projects/sat-io/open-datasets/shoreline/{collection}"
    fc = ee.FeatureCollection(asset_id)
    page_size = PAGE_SIZE[collection]
    batch = ee.FeatureCollection(fc.toList(page_size, offset)).getInfo()["features"]

    geojson = {"type": "FeatureCollection", "features": batch}
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    return offset


def main():
    tasks = []
    for collection in COLLECTIONS:
        asset_id = f"projects/sat-io/open-datasets/shoreline/{collection}"
        fc = ee.FeatureCollection(asset_id)
        total = fc.size().getInfo()
        print(f"Collection '{collection}': total features = {total}", file=sys.stderr)

        page_size = PAGE_SIZE[collection]
        offsets = list(range(0, total, page_size))
        for offset in offsets:
            tasks.append((collection, offset))

    # Parallel download
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(download_batch, coll, off): (coll, off) for coll, off in tasks}

        for future in tqdm(as_completed(future_to_task), total=len(tasks)):
            coll, off = future_to_task[future]
            try:
                _ = future.result()
            except Exception as e:
                print(f"Error downloading {coll} at offset {off}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
