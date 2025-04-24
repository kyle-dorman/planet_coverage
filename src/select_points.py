#!/usr/bin/env python3
"""
filter_coastal_points.py

Generate a regular Sinusoidal grid, buffer points and retain only those
that intersect coastal strips.  Uses a prepared union for lightning-fast
intersection tests, Dask parallelism, and a live progress bar.
"""
import logging
import multiprocessing as mp
from pathlib import Path

import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
from shapely.prepared import prep

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
COASTAL_GPKG = Path("/Users/kyledorman/data/shorelines/coastal_strips.gpkg")
COASTAL_LAYER = "coastal_strips"
OUTPUT_PATH = Path("/Users/kyledorman/data/shorelines/points_in_coastal_strips.gpkg")
SINUSOIDAL = "ESRI:54008"  # equal-area sinusoidal
PROJ_CRS = "EPSG:6933"
STEP = 5_559.7522  # grid spacing in metres
BUFFER_FACTOR = 0.5  # fraction of step for buffer radius
BUFFER_RADIUS = STEP * BUFFER_FACTOR
N_PARTITIONS = int(mp.cpu_count()) - 1  # Dask partitions

# ------------------------------------------------------------------
# Setup logging
# ------------------------------------------------------------------
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 1. Load & prepare coastal strips
# ------------------------------------------------------------------
logger.info("Loading coastal strips from %s (layer %s)", COASTAL_GPKG, COASTAL_LAYER)
coastal = gpd.read_file(COASTAL_GPKG, layer=COASTAL_LAYER)
assert coastal.crs == PROJ_CRS
assert len(coastal) == 1
logger.info("Loaded %d coastal features", len(coastal))

# ------------------------------------------------------------------
# 2. Build regular grid of points in Sinusoidal
# ------------------------------------------------------------------
minx, miny, maxx, maxy = coastal.to_crs(SINUSOIDAL).total_bounds
xs = np.arange(minx, maxx + STEP, STEP)
ys = np.arange(miny, maxy + STEP, STEP)
xx, yy = np.meshgrid(xs, ys)

logger.info("Generating grid: %d cols × %d rows = %d points", len(xs), len(ys), xx.size)
grid = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()), crs=SINUSOIDAL)
logger.info("Grid created: %d points", len(grid))

# ------------------------------------------------------------------
# 3. Dask GeoDataFrame & prepared intersection filter
# ------------------------------------------------------------------
logger.info("Converting grid to Dask GeoDataFrame with %d partitions", N_PARTITIONS)
dask_grid: dgpd.GeoDataFrame = dgpd.from_geopandas(grid, npartitions=N_PARTITIONS)


def filter_partition(df: gpd.GeoDataFrame, partition_info=None):
    """
    Buffer all points in the partition and test intersects against the
    prepared coastal union. Returns only points that intersect.
    """
    if partition_info is not None:
        part = partition_info["number"]
    else:
        part = -1

    # buffer all at once
    logger.info(f"{part}: buffering and projecting")
    buffered = df.to_crs(PROJ_CRS).geometry.buffer(BUFFER_RADIUS)

    # Build a prepared MultiPolygon for fast intersects
    logger.info(f"{part}: preparing data")
    coastal_union = coastal.union_all(method="coverage")
    prep_coastal = prep(coastal_union)

    # vectorized intersects via prepared geom
    logger.info(f"{part}: compute intersection")
    mask = buffered.apply(lambda geom: prep_coastal.intersects(geom))
    logger.info(f"{part}: returning results")
    return df.loc[mask]


logger.info("Filtering grid in parallel using prepared geometry…")
filtered_dask = dask_grid.map_partitions(filter_partition, meta=dask_grid._meta)

logger.info("Computing results")
points_in_coast: gpd.GeoDataFrame = filtered_dask.compute()
logger.info("Filtered point count: %d", len(points_in_coast))

# ------------------------------------------------------------------
# 4. Save filtered points
# ------------------------------------------------------------------
logger.info("Saving to %s", OUTPUT_PATH)
points_in_coast.to_file(OUTPUT_PATH, driver="GPKG")
logger.info("Done!")
