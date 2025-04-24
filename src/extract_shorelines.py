import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

# === CONFIGURATION ===
MAINLANDS_GPKG = Path("/Users/kyledorman/data/shorelines/mainlands.gpkg")
SMALL_ISLANDS_GPKG = Path("/Users/kyledorman/data/shorelines/small_islands.gpkg")
BIG_ISLANDS_GPKG = Path("/Users/kyledorman/data/shorelines/big_islands.gpkg")
OUTPUT_GPKG = Path("/Users/kyledorman/data/shorelines/coastal_strips.gpkg")
OUTPUT_LAYER = "coastal_strips"
PROJ_CRS = "EPSG:6933"
IN_CRS = "EPSG:4326"
BUFFER_INNER = 1000  # m
BUFFER_OUTER = 2000  # m
COASTAL_BUFFER = 2000  # m
COASTAL_BUFFER_WIDTH = COASTAL_BUFFER + BUFFER_OUTER - BUFFER_INNER
SMALL_LAND_AREA = COASTAL_BUFFER_WIDTH**2 * math.pi / 1e6  # km²
MID_LAND_AREA = SMALL_LAND_AREA**2  # km²
N_PARTITIONS = 16
PRE_SIMPLIFY_TOL = 50  # m
POST_SIMPLIFY_TOL = 200  # m
SELF_BUFFER_EXTRA = 100  # m for same-index carve

# Utility to clean internal holes


def remove_internal_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cleaned = []
    for geom in gdf.geometry:
        if isinstance(geom, Polygon):
            cleaned.append(geom)
        elif isinstance(geom, MultiPolygon):
            polys = list(geom.geoms)
            keepers = []
            for i, p1 in enumerate(polys):
                if not any(i != j and p1.within(p2) for j, p2 in enumerate(polys)):
                    keepers.append(p1)
            geom2 = keepers[0] if len(keepers) == 1 else MultiPolygon(keepers)
            cleaned.append(geom2)
        else:
            cleaned.append(geom)
    out = gdf.copy()
    out.geometry = cleaned
    return out


# Large land carve + buffer logic
def process_large_land(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # project
    df = df.to_crs(PROJ_CRS)
    # pre-simplify
    df.geometry = df.geometry.simplify(PRE_SIMPLIFY_TOL, preserve_topology=True)
    # inner buffer
    df_inner = gpd.GeoDataFrame(df)
    df_inner.geometry = df.geometry.buffer(BUFFER_INNER)
    # build global index once
    index = df_inner.sindex
    geoms_list = list(df_inner.geometry)
    # carve
    carved = []
    for idx, geom in zip(df.index, df.geometry.buffer(BUFFER_OUTER)):
        for j in index.intersection(geom.bounds):
            other = geoms_list[j]
            if idx == j:
                other = other.buffer(SELF_BUFFER_EXTRA)
            if geom.intersects(other):
                geom = geom.difference(other)
        carved.append(geom.buffer(COASTAL_BUFFER))
    out = gpd.GeoDataFrame(geometry=carved, crs=PROJ_CRS)
    out = out[~out.geometry.is_empty]
    out = remove_internal_polygons(out)
    return out


# Mid islands logic
def process_mid_islands(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df = df.to_crs(PROJ_CRS)
    df.geometry = df.geometry.simplify(PRE_SIMPLIFY_TOL, preserve_topology=True)
    outer = df.geometry.buffer(BUFFER_OUTER)
    inner = df.geometry.buffer(BUFFER_INNER)
    ring = outer.difference(inner)
    ring = ring.buffer(COASTAL_BUFFER)
    out = gpd.GeoDataFrame(geometry=ring, crs=PROJ_CRS)
    out = out[~out.geometry.is_empty]
    return remove_internal_polygons(out)


# Small islands logic
def process_small_islands(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df = df.to_crs(PROJ_CRS)
    df.geometry = df.geometry.simplify(PRE_SIMPLIFY_TOL, preserve_topology=True)
    buf = df.geometry.buffer(COASTAL_BUFFER_WIDTH)
    out = gpd.GeoDataFrame(geometry=buf, crs=PROJ_CRS)
    out = out[~out.geometry.is_empty]
    return remove_internal_polygons(out)


# Main
if __name__ == "__main__":
    print("Loading features...")
    mainlands = gpd.read_file(MAINLANDS_GPKG, layer="shorelines")
    assert mainlands.crs == IN_CRS
    big_islands = gpd.read_file(BIG_ISLANDS_GPKG, layer="shorelines")
    assert big_islands.crs == IN_CRS
    small_islands = gpd.read_file(SMALL_ISLANDS_GPKG, layer="shorelines")
    assert small_islands.crs == IN_CRS

    # partition big vs mid vs small
    large_islands = big_islands[big_islands.Area_km2 > MID_LAND_AREA]
    mid_islands = big_islands[(big_islands.Area_km2 <= MID_LAND_AREA) & (big_islands.Area_km2 > SMALL_LAND_AREA)]
    # process each separately
    print("Processing mainlands...")
    coast_main = process_large_land(mainlands)
    print("Processing large islands...")
    coast_large = process_large_land(large_islands)
    print("Processing mid islands...")
    coast_mid = process_mid_islands(mid_islands)
    print("Processing small islands...")
    coast_small = process_small_islands(small_islands)

    # combine all
    print("Combining...")
    all_geo = pd.concat([coast_main, coast_large, coast_mid, coast_small], ignore_index=True)
    all_gdf = gpd.GeoDataFrame(geometry=all_geo.geometry, crs=PROJ_CRS)
    # drop any part smaller than, say, 1 km²
    all_gdf = all_gdf[all_gdf.area > 1e6]  # in CRS units (m²)
    # post-simplify & final clean
    all_gdf.geometry = all_gdf.geometry.simplify(POST_SIMPLIFY_TOL, preserve_topology=True)
    all_gdf = all_gdf[~all_gdf.geometry.is_empty].reset_index(drop=True)
    invalid = ~all_gdf.geometry.is_valid
    if invalid.any():
        all_gdf.loc[invalid, "geometry"] = all_gdf.loc[invalid].geometry.buffer(0)
    print(f"Total features before dissolve: {len(all_gdf)}")

    # final dissolve
    print("Dissolving...")
    # 1) Dissolve in geographic coords to avoid Mercator clipping
    merged = unary_union(all_gdf.geometry)
    merged = gpd.GeoDataFrame(geometry=[merged], crs=PROJ_CRS)

    # 2) Reproject to 6933 to simplify in meters
    merged.geometry = merged.geometry.simplify(POST_SIMPLIFY_TOL, preserve_topology=True)
    invalid = ~merged.geometry.is_valid
    if invalid.any():
        merged.loc[invalid, "geometry"] = merged.loc[invalid].geometry.buffer(0)

    assert merged.geometry.is_valid.all()

    # 4) Write out
    merged.to_file(OUTPUT_GPKG, layer=OUTPUT_LAYER, driver="GPKG")
    print(f"Exported final dissolved strip to {OUTPUT_GPKG}")
