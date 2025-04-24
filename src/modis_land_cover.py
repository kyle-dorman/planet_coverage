#!/usr/bin/env python3
"""
modis_land_filter.py

Given a set of point locations in Sinusoidal (ESRI:54008) stored in a GeoPackage,
scan a directory of MCD12Q1 HDF files to sample the 2019 "LC_Type1" band
and determine whether each point falls on land (code ≠ 0) or water (code = 0).
Uses multiprocessing to parallelize per-tile sampling.
Outputs a CSV with point coords and an `is_land` flag, and a GeoPackage with
all attributes.
"""
import argparse
import logging
import multiprocessing as mp
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


def parse_args():
    p = argparse.ArgumentParser(description="Sample points against MCD12Q1 2019 LC_Type1 using multiprocessing")
    p.add_argument("--points", required=True, help="GeoPackage of point geometries (Sinusoidal ESRI:54008)")
    p.add_argument("--layer", default=None, help="Layer name in GeoPackage; defaults to first layer")
    p.add_argument("--modis-dir", required=True, help="Directory containing MCD12Q1_2019 HDF files")
    p.add_argument(
        "--output-dir", required=True, help="Directory to write results: points_land.csv and points_land.gpkg"
    )
    p.add_argument("--workers", type=int, default=mp.cpu_count(), help="Number of parallel worker processes")
    return p.parse_args()


def sample_tile(args):
    """
    Sample LC_Type1 for the given tile (.hdf path) against global points array.
    Returns list of (index, code) for points within tile bounds.
    """
    tile_path, points_xy, pts_crs = args
    results = []
    try:
        # Open only the LC_Type1 subdataset
        subds_name = [s for s in rasterio.open(tile_path).subdatasets if "LC_Type1" in s][0]
        with rasterio.open(subds_name) as src:
            # Optionally warp to ensure same CRS/resolution
            with WarpedVRT(src, crs=pts_crs, resampling=Resampling.nearest) as vrt:
                # compute tile bounds
                left, bottom, right, top = vrt.bounds
                # find point indices within tile bbox
                xs, ys = points_xy
                mask = (xs >= left) & (xs <= right) & (ys >= bottom) & (ys <= top)
                indices = np.nonzero(mask)[0]
                if len(indices) == 0:
                    return []
                # sample raster at those points
                coords = [(xs[i], ys[i]) for i in indices]
                codes = [val[0] for val in vrt.sample(coords)]
                results = list(zip(indices, codes))
    except Exception as e:
        logging.warning(f"Error sampling {tile_path.name}: {e}")
    return results


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    pts_fp = Path(args.points)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Read points
    logging.info(f"Loading points from {pts_fp}!{args.layer or ''}")
    pts = gpd.read_file(str(pts_fp), layer=args.layer)
    assert pts.crs is not None
    assert pts.crs.to_string().startswith("ESRI:54008"), "Points must be in ESRI:54008"

    # Prepare arrays of point coords for sampling
    xs = pts.geometry.x.values
    ys = pts.geometry.y.values
    points_xy = (xs, ys)
    n_pts = len(pts)
    logging.info(f"Loaded {n_pts} points")

    # 2) Gather HDF tile paths
    modis_dir = Path(args.modis_dir)
    hdflist = sorted(modis_dir.glob("*.hdf"))
    logging.info(f"Found {len(hdflist)} HDF tiles in {modis_dir}")

    # 3) Initialize codes to -1 (no data)
    codes_arr = np.full(n_pts, -1, dtype=int)

    # 4) Parallel sampling per tile
    logging.info(f"Sampling with {args.workers} workers...")
    pool = mp.Pool(args.workers)
    try:
        # args per task: (tile_path, points_xy)
        tasks = [(path, points_xy, pts.crs) for path in hdflist]
        for result in pool.imap_unordered(sample_tile, tasks):
            for idx, code in result:
                # only overwrite if code >= 0
                if code is not None and code >= 0:
                    if codes_arr[idx] > 0 and code <= 0:
                        continue
                    codes_arr[idx] = code
    finally:
        pool.close()
        pool.join()

    # 5) Flag land vs water
    is_land = codes_arr > 0
    pts["LC_Code"] = codes_arr
    pts["is_land"] = is_land

    # 6) Write outputs
    # csv_fp = out_dir / "points_land.csv"
    gpkg_fp = out_dir / "points_land.gpkg"
    # logging.info(f"Saving CSV to {csv_fp}")
    # pts[['geometry', 'LC_Code', 'is_land']].to_csv(str(csv_fp), index=False)
    logging.info(f"Saving GeoPackage to {gpkg_fp}")
    pts.to_file(str(gpkg_fp), layer="points_land", driver="GPKG")

    logging.info("Done.")


if __name__ == "__main__":
    main()
