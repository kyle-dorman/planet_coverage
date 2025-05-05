#!/usr/bin/env python3
import logging
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import click
import geopandas as gpd
import pandas as pd
import polars as pl
import shapely
from tqdm import tqdm

from src.config import Instrument, ItemType, PublishingStage, QualityCategory
from src.geo_util import dedup_satellite_captures

# configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
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
    "row_id": pl.UInt32,
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


def process_file(src: Path, polygons_path: Path, points_path: Path, max_duration_hrs: int) -> None:
    """Process one data.parquet file for coastal points."""
    logger.debug("Loading polygons from %s", polygons_path)
    gdf_cells = gpd.read_file(polygons_path)
    # add hex identifier to polygons for matching
    gdf_cells["hex_id"] = gdf_cells["cell_id"].apply(lambda x: f"{x:06x}")
    logger.debug("Loading sample points from %s", points_path)
    gdf_pts = gpd.read_file(points_path).rename(columns={"cell_id": "grid_id"})

    assert gdf_pts.crs == gdf_cells.crs, "GDFs must be in the same CRS"

    # derive hex identifier from directory names
    d3 = src.parent.name
    d2 = src.parent.parent.name
    d1 = src.parent.parent.parent.name
    hex_str = f"{d1}{d2}{d3}"
    dest = src.parent / "coastal_points.parquet"
    logger.debug("Processing %s → hex_id %s → %s", src, hex_str, dest.name)

    # get the polygon by hex_id
    match = gdf_cells[gdf_cells.hex_id == hex_str]
    assert not match.empty, f"Hex {hex_str} not found in polygons, skipping"
    best_crs = match.estimate_utm_crs()
    poly = match.iloc[0].geometry
    cell_id = match.iloc[0].cell_id

    # find points in that polygon
    pts_in = gdf_pts[gdf_pts.geometry.within(poly)]
    if pts_in.empty:
        logger.debug("  no sample points in cell %d, writing empty file", cell_id)
        # write empty schema
        empty = pl.DataFrame(schema=SCHEMA)
        empty.write_parquet(dest)
        return

    # load the parquet into pandas to rebuild geometries
    df = pd.read_parquet(src, engine="pyarrow")
    # reconstruct row geometries
    # Reconstruct geometries
    df["geometry"] = df["geometry_wkb"].apply(shapely.wkb.loads)  # type: ignore

    # Build a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    # Verify all geometries are valid
    assert gdf.geometry.is_valid.all(), "Found invalid geometries!"

    # Convert to a local best CRS for both datasets
    gdf = gdf.to_crs(best_crs)
    pts_in = pts_in.to_crs(best_crs)

    # --- after you’ve built `gdf` (a GeoDataFrame of your rows)
    # and `pts_in` (a GeoDataFrame of your points with a `grid_id` column) ---

    # Perform the spatial join: every row–point pair where row.geom ∩ pt.geom is non-empty
    joined = gpd.sjoin(pts_in[["grid_id", "geometry"]], gdf, how="inner", predicate="within").to_crs("EPSG:4326")

    # Convert point geometry to wkb for saving (overwriting polygon wkb)
    joined["geometry_wkb"] = joined.geometry.map(lambda geom: geom.wkb)

    # joined now has all columns from gdf + a `grid_id` and the index of the matched pt
    # Drop the extra index column that sjoin adds and the point geometry column
    joined = joined.drop(columns=["index_right", "geometry"])

    if joined.empty:
        logger.debug("  no intersection points in cell %d, writing empty file", cell_id)
        # write empty schema
        empty = pl.DataFrame(schema=SCHEMA)
        empty.write_parquet(dest)
        return

    # Convert to Polars (or pandas) to write out
    pl_out = pl.from_pandas(joined, schema_overrides=SCHEMA, include_index=False)

    pl_out = dedup_satellite_captures(pl_out.lazy(), max_duration_hrs=max_duration_hrs, column_name="grid_id")

    pl_out.write_parquet(dest)


@click.command()
@click.argument("base_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--polygons",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="GeoJSON/Parquet of one polygon per cell_id (index must match cell_id)",
)
@click.option(
    "--points",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="GeoJSON/Parquet of points, each with a `grid_id` column",
)
@click.option(
    "--max-duration-hrs",
    type=int,
    default=4,
    show_default=True,
    help="Maximum duration in hours for deduplication threshold",
)
def main(base_dir: Path, polygons: Path, points: Path, max_duration_hrs: int):
    """
    For every data.parquet under BASE_DIR (glob */*/*/*/data.parquet):
      • infer its cell_id from the d1/d2/d3 path
      • load that cell's polygon from POLYGONS
      • load all POINTS whose geometry falls in that polygon
      • read the Parquet, reconstruct each row's geometry via geometry_wkb
      • for each point in that cell, find all rows whose geometry intersects the point
      • emit one record per (row, point) match, carrying row-fields + grid_id + cell_geometry
      • write to coastal_points.parquet alongside the original data.parquet
    """
    files = list(base_dir.glob(GLOB_PATTERN))
    if not len(files):
        logger.error("No data.parquet files found under %s", base_dir)
        return

    num_workers = max(1, cpu_count() - 1)
    with Pool(num_workers) as pool:
        func = partial(process_file, polygons_path=polygons, points_path=points, max_duration_hrs=max_duration_hrs)
        for _ in tqdm(pool.imap_unordered(func, files), total=len(files), desc="Processing files"):
            pass


if __name__ == "__main__":
    main()
