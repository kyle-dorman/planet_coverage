import logging
from pathlib import Path

import geopandas as gpd
import tqdm

# Completely disable all logging below CRITICAL
logging.disable(logging.CRITICAL)

STEP = 5_559.7522  # grid spacing in metres
BUFFER_FACTOR = 0.5  # fraction of step for buffer radius
BUFFER_RADIUS = STEP * BUFFER_FACTOR
GROUP_SIZE = 1000

points_coastal = gpd.read_file(Path("/Users/kyledorman/data/shorelines/points_land.gpkg"), layer="points_land")

coastal_circles = points_coastal.copy()
coastal_circles.geometry = coastal_circles.geometry.buffer(BUFFER_RADIUS)
coastal_circles = coastal_circles.to_crs("EPSG:4326")

# output folder
out_dir = Path("/Users/kyledorman/data/planet_coverage/grids/")
out_dir.mkdir(parents=True, exist_ok=True)

# 2) Loop through each polygon, wrap it in a GeoDataFrame, and save
for i, poly in enumerate(tqdm.tqdm(coastal_circles.geometry)):
    hex_id = f"{i:06x}"  # unique 6‑digit hex, e.g. '0f1a2b'
    d1, d2 = hex_id[:2], hex_id[2:4]
    dir_path = out_dir / d1 / d2
    dir_path.mkdir(parents=True, exist_ok=True)

    # build a 1‑row GeoDataFrame
    gdf = gpd.GeoDataFrame({"id": [i]}, geometry=[poly], crs=coastal_circles.crs)

    # write to GeoJSON
    out_path = dir_path / f"circle_{i}.geojson"
    gdf.to_file(out_path, driver="GeoJSON")

logging.disable(logging.NOTSET)
