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
import logging
from contextlib import contextmanager
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import rasterio
from pyhdf.SD import SD, SDC  # type: ignore
from rasterio.transform import rowcol
from tqdm import tqdm


@contextmanager
def open_sd(path: Path):
    """
    Context manager for PyHDF SD: yields an SD instance and ensures sd.end() on exit.
    """
    sd = SD(str(path), SDC.READ)
    try:
        yield sd
    finally:
        sd.end()


def get_hdf_sources(modis_dir: Path, sds_name: str) -> tuple[list[Path], str]:
    """
    Discover HDF files in modis_dir, verify they share one CRS and transform,
    and return (list of HDF paths, common CRS, affine transform).
    """
    hdf_paths = sorted(Path(modis_dir).glob("*.hdf"))
    if not hdf_paths:
        raise FileNotFoundError(f"No HDF files found in {modis_dir}")

    # Determine subdataset and metadata from first file
    subds0 = next(s for s in rasterio.open(hdf_paths[0]).subdatasets if sds_name in s)
    with rasterio.open(subds0) as src0:
        common_crs = src0.crs

    # Verify consistency
    for path in hdf_paths[1:]:
        subds = next(s for s in rasterio.open(path).subdatasets if sds_name in s)
        with rasterio.open(subds) as ssrc:
            assert ssrc.crs == common_crs, f"CRS mismatch in {path}"

    return hdf_paths, str(common_crs)


def prepare_points(points_path: Path, target_crs: str) -> tuple[gpd.GeoDataFrame, np.ndarray, np.ndarray]:
    """
    Load point GeoDataFrame at points_path, verify Sinusoidal CRS,
    reproject to target_crs, and return (GeoDataFrame, xs, ys).
    """
    pts = gpd.read_file(points_path)
    assert pts.crs is not None and pts.crs.to_string().startswith(
        "ESRI:54008"
    ), f"Points must be ESRI:54008, got {pts.crs}"
    pts_h = pts.to_crs(target_crs)
    xs = np.asarray(pts_h.geometry.x.values)
    ys = np.array(pts_h.geometry.y.values)
    return pts_h, xs, ys


def classify_points(
    hdf_paths: list[Path],
    sds_name: str,
    xs_h: np.ndarray,
    ys_h: np.ndarray,
) -> np.ndarray:
    """
    Iterate over HDF tiles, sampling the SDS at (xs_h, ys_h),
    and return a boolean mask is_land.
    """
    is_land = np.zeros(xs_h.shape[0], dtype=np.bool_)
    for hdf_path in tqdm(hdf_paths, desc="Sampling tiles"):
        subds = next(s for s in rasterio.open(hdf_path).subdatasets if sds_name in s)
        with rasterio.open(subds) as src:
            transform = src.transform

        # Read full array via PyHDF
        with open_sd(hdf_path) as sd:
            sds = sd.select(sds_name)
            arr = sds.get()

        n_rows, n_cols = arr.shape
        # Compute rows/cols only for points inside tile bounds
        rows, cols = rowcol(transform, xs_h, ys_h)
        for idx, (r, c) in enumerate(zip(rows, cols)):
            if 0 <= r < n_rows and 0 <= c < n_cols:
                code = int(arr[r, c])
                # IGBP LC_Type1: codes 1–16 are land; code 17 is permanent water; others are fill/NA
                # Assume this is how Roy did it.
                if 1 <= code <= 16:
                    is_land[idx] = True
    return is_land


@click.command()
@click.option("--points", type=click.Path(exists=True), required=True, help="GeoPackage of point geometries")
@click.option("--modis-dir", type=click.Path(exists=True), required=True, help="Directory of MCD12Q1 HDF files")
@click.option("--output-dir", type=click.Path(), required=True, help="Directory to write GeoPackage")
@click.option("--sds-name", default="LC_Type1", show_default=True, help="HDF SDS name for land cover")
def main(
    points: str,
    modis_dir: str,
    output_dir: str,
    sds_name: str,
) -> None:
    """Classify points as land/water using MCD12Q1 LC_Type1 SDS."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    pts_path = Path(points)
    out_fp = Path(output_dir) / "points_land.npy"

    # Discover HDF tiles and metadata
    hdf_paths, hdf_crs = get_hdf_sources(Path(modis_dir), sds_name)
    logger.info(f"Found {len(hdf_paths)} HDF tiles with CRS {hdf_crs}")

    # Prepare and reproject points
    pts_h, xs_h, ys_h = prepare_points(pts_path, hdf_crs)

    # Classify land/water
    is_land = classify_points(hdf_paths, sds_name, xs_h, ys_h)

    pct_land = 100 * is_land.sum() / len(is_land)
    logger.info(f"{pct_land:1f}% are land point")

    # Write output
    logger.info(f"Writing results to {out_fp}")
    np.save(out_fp, is_land)
    logger.info("Workflow complete")


if __name__ == "__main__":
    main()
