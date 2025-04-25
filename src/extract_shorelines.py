import logging
import math
from typing import Tuple

import click
import geopandas as gpd
import pandas as pd
from shapely import GeometryCollection
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def gc_to_multipolygon(geom):
    """
    If geom is a GeometryCollection, pull out all Polygons and
    MultiPolygons and return a single MultiPolygon. Otherwise return geom.
    """
    if isinstance(geom, GeometryCollection):
        parts = []
        for part in geom.geoms:
            if isinstance(part, Polygon):
                parts.append(part)
            elif isinstance(part, MultiPolygon):
                parts.extend(part.geoms)
        # If we found nothing polygonal, just return the original
        return MultiPolygon(parts) if parts else geom
    else:
        return geom


def clean_invalid(df: gpd.GeoDataFrame):
    # Clean invalid geometries inplace
    invalid = ~df.geometry.is_valid
    df.loc[invalid, "geometry"] = df.geometry[invalid].buffer(0)


# Helper for projection & simplification
def preprocess_geometry(
    df: gpd.GeoDataFrame,
    proj_crs: str,
    tol: float,
) -> gpd.GeoDataFrame:
    """
    Reproject the GeoDataFrame to proj_crs and simplify geometries with given tolerance.
    """
    df = df[~df.geometry.is_empty]
    df = df.to_crs(proj_crs)
    clean_invalid(df)
    df.geometry = df.geometry.apply(gc_to_multipolygon)  # type: ignore
    clean_invalid(df)
    # Merge any multipart or collection geometries into a single unified geometry
    df.geometry = df.geometry.apply(lambda geom: unary_union([geom]))  # type: ignore
    clean_invalid(df)
    df = remove_internal_polygons(df)
    clean_invalid(df)
    df.geometry = df.geometry.simplify(tol, preserve_topology=True)
    clean_invalid(df)
    df = df[~df.geometry.is_empty]
    return df


def remove_internal_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Remove polygons that are holes inside larger polygons.
    """
    cleaned = []
    for geom in gdf.geometry:
        if isinstance(geom, Polygon):
            cleaned.append(geom)
        elif isinstance(geom, MultiPolygon):
            parts = []
            for p in geom.geoms:
                if not any(p.within(other) for other in geom.geoms if p is not other):
                    parts.append(p)
            cleaned.append(parts[0] if len(parts) == 1 else MultiPolygon(parts))
        else:
            cleaned.append(geom)
    out = gdf.copy()
    out.geometry = cleaned
    return out


def finalize_strips(
    geometries,
    proj_crs: str,
) -> gpd.GeoDataFrame:
    """
    Apply final CRS, drop empty geometries, and remove internal holes.
    """
    df = gpd.GeoDataFrame(geometry=geometries, crs=proj_crs)

    df = df[~df.geometry.is_empty]
    df.geometry = df.geometry.apply(gc_to_multipolygon)  # type: ignore
    clean_invalid(df)
    # Merge any multipart or collection geometries into a single unified geometry
    df.geometry = df.geometry.apply(lambda geom: unary_union([geom]))  # type: ignore
    clean_invalid(df)
    df = remove_internal_polygons(df)
    clean_invalid(df)
    return df


# Buffer computation helper
def compute_buffers(land_km: int, water_km: int) -> Tuple[float, float, float]:
    """
    Given land and water coverage in km (>=1), compute:
    buffer_inner, buffer_outer, coastal_buffer in meters.
    """
    # Convert to meters
    land_m = land_km * 1000
    water_m = water_km * 1000
    # Center and ±500 m ring
    center = (water_m - land_m) / 2
    buffer_inner = center - 500
    assert buffer_inner >= 0, "Inner buffer must maintain or expand the geometries to find a shoreline."
    buffer_outer = center + 500
    coastal_buffer = water_m - buffer_outer
    return buffer_inner, buffer_outer, coastal_buffer


def process_large_land(
    df: gpd.GeoDataFrame,
    proj_crs: str,
    pre_simplify_tol: float,
    buffer_inner: float,
    buffer_outer: float,
    coastal_buffer: float,
    self_buffer_extra: float,
) -> gpd.GeoDataFrame:
    """
    Process mainland or large islands: buffer carve and generate coastal strip.
    """
    df = preprocess_geometry(df, proj_crs, pre_simplify_tol)

    inner = df.geometry.buffer(buffer_inner)
    index = gpd.GeoSeries(inner).sindex
    outer = df.geometry.buffer(buffer_outer)

    carved = []
    # Iterate using actual DataFrame indices to remain consistent
    for row_idx, geom in zip(df.index, outer):
        for j in index.intersection(geom.bounds):
            other = inner.iloc[j]
            # If this is the same feature, apply extra self-buffer
            if row_idx == inner.index[j]:
                other = other.buffer(self_buffer_extra)
            if geom.intersects(other):
                geom = geom.difference(other)
        # Buffer the carved geometry by coastal_buffer
        carved.append(geom.buffer(coastal_buffer))

    return finalize_strips(carved, proj_crs)


def process_mid_islands(
    df: gpd.GeoDataFrame,
    proj_crs: str,
    pre_simplify_tol: float,
    buffer_inner: float,
    buffer_outer: float,
    coastal_buffer: float,
) -> gpd.GeoDataFrame:
    """
    Process mid-sized islands: ring buffer yield.
    """
    df = preprocess_geometry(df, proj_crs, pre_simplify_tol)

    outer = df.geometry.buffer(buffer_outer)
    inner = df.geometry.buffer(buffer_inner)
    ring = outer.difference(inner).buffer(coastal_buffer)

    return finalize_strips(ring, proj_crs)


def process_small_islands(
    df: gpd.GeoDataFrame,
    proj_crs: str,
    pre_simplify_tol: float,
    coastal_buffer_width: float,
) -> gpd.GeoDataFrame:
    """
    Process small islands: single buffer.
    """
    df = preprocess_geometry(df, proj_crs, pre_simplify_tol)

    buf = df.geometry.buffer(coastal_buffer_width)

    return finalize_strips(buf, proj_crs)


@click.command()
@click.option(
    "--mainlands", "mainlands_gpkg", type=click.Path(exists=True), required=True, help="Path to mainlands GeoPackage"
)
@click.option(
    "--big-islands",
    "big_islands_gpkg",
    type=click.Path(exists=True),
    required=True,
    help="Path to big islands GeoPackage",
)
@click.option(
    "--small-islands",
    "small_islands_gpkg",
    type=click.Path(exists=True),
    required=True,
    help="Path to small islands GeoPackage",
)
@click.option("--output-file", "-o", type=click.Path(), required=True, help="Path for output GeoPackage file")
@click.option("--proj-crs", default="EPSG:6933", show_default=True, help="Projected CRS for buffering operations")
@click.option(
    "--land-buffer-km",
    type=click.IntRange(1, None),
    default=1,
    show_default=True,
    help="Coverage over land in kilometers (>=1 km)",
)
@click.option(
    "--water-buffer-km",
    type=click.IntRange(1, None),
    default=1,
    show_default=True,
    help="Coverage over water in kilometers (>=1 km)",
)
@click.option(
    "--self-buffer-extra", type=float, default=100, show_default=True, help="Extra buffer for carving self overlaps (m)"
)
@click.option(
    "--pre-simplify-tol",
    type=float,
    default=50,
    show_default=True,
    help="Tolerance for pre-buffer geometry simplification (m)",
)
@click.option(
    "--post-simplify-tol",
    type=float,
    default=200,
    show_default=True,
    help="Tolerance for post-dissolve simplification (m)",
)
def main(
    mainlands_gpkg: str,
    big_islands_gpkg: str,
    small_islands_gpkg: str,
    output_file: str,
    proj_crs: str,
    land_buffer_km: int,
    water_buffer_km: int,
    self_buffer_extra: float,
    pre_simplify_tol: float,
    post_simplify_tol: float,
) -> None:
    """
    Extract coastal strips from mainlands and islands, and write to GeoParquet.
    """

    logger.info("Loading source layers")
    mldf = gpd.read_file(mainlands_gpkg)
    bidf = gpd.read_file(big_islands_gpkg)
    sidf = gpd.read_file(small_islands_gpkg)

    # Compute buffer distances
    buffer_inner, buffer_outer, coastal_buffer = compute_buffers(land_buffer_km, water_buffer_km)
    logger.info(f"Computed buffers (m): inner={buffer_inner}, outer={buffer_outer}, coastal={coastal_buffer}")

    # Partition big islands
    small_area = land_buffer_km**2 * math.pi
    small_area_m = (land_buffer_km * 1000) ** 2 * math.pi
    mid_area = small_area**2
    large_islands = bidf[bidf.Area_km2 > mid_area].reset_index(drop=True)
    mid_islands = bidf[(bidf.Area_km2 <= mid_area) & (bidf.Area_km2 > small_area)].reset_index(drop=True)

    logger.info("Processing mainlands")
    coast_main = process_large_land(
        mldf, proj_crs, pre_simplify_tol, buffer_inner, buffer_outer, coastal_buffer, self_buffer_extra
    )

    logger.info("Processing large islands")
    coast_large = process_large_land(
        large_islands, proj_crs, pre_simplify_tol, buffer_inner, buffer_outer, coastal_buffer, self_buffer_extra
    )

    logger.info("Processing mid islands")
    coast_mid = process_mid_islands(mid_islands, proj_crs, pre_simplify_tol, buffer_inner, buffer_outer, coastal_buffer)

    logger.info("Processing small islands")
    coast_small = process_small_islands(sidf, proj_crs, pre_simplify_tol, water_buffer_km * 1000)

    # Combine
    logger.info("Combining strips")
    combined = pd.concat([coast_main, coast_large, coast_mid, coast_small], ignore_index=True)
    combined_gdf = gpd.GeoDataFrame(geometry=combined.geometry, crs=proj_crs)
    # Filter small geometries
    combined_gdf = combined_gdf[combined_gdf.geometry.area > small_area_m]
    # Post-simplify
    clean_invalid(combined_gdf)
    combined_gdf.geometry = combined_gdf.geometry.simplify(post_simplify_tol, preserve_topology=True)
    clean_invalid(combined_gdf)

    # Dissolve and clean
    logger.info("Dissolving geometries")
    merged = unary_union(combined_gdf.geometry)
    final = gpd.GeoDataFrame(geometry=[merged], crs=proj_crs)
    clean_invalid(final)
    final.geometry = final.geometry.simplify(post_simplify_tol, preserve_topology=True)
    clean_invalid(final)

    logger.info(f"Writing output to {output_file}")
    final.to_file(output_file, driver="GPKG")


if __name__ == "__main__":
    main()
