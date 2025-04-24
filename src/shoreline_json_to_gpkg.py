import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
from dask import compute, delayed

# ─── CONFIG ───────────────────────────────────────────────
BASE = Path("/Users/kyledorman/data/shorelines")  # your mainlands/, big_islands/, small_islands/ folders
OUT = BASE / "coastal_strips.gpkg"  # final merged GPKG
if OUT.exists():
    os.remove(OUT)
LAYER = "coastal_strips"
KEEP = ["Area_km2", "geometry"]

# # 1) Gather all GeoJSON paths
# files = (
#     # list((BASE / "mainlands").glob("*.geojson")) # +
#     # list((BASE / "big_islands").glob("*.geojson")) #+
#     list((BASE / "small_islands").glob("*.geojson"))
# )

# # 2) Delayed loader that immediately subsets columns
# def load_and_trim(fp):
#     # read with pyogrio
#     gdf = gpd.read_file(str(fp), engine="pyogrio")
#     # keep only desired columns (plus geometry)
#     return gdf[KEEP]

# # 3) Build delayed tasks
# tasks = [delayed(load_and_trim)(fp) for fp in files]

# # 4) Execute in parallel (adjust n_workers to your CPU count)
# gdf_list = compute(*tasks, scheduler="processes", num_workers=8)

# # 5) Concatenate into a single GeoDataFrame
# shore_gdf = gpd.GeoDataFrame(
#     pd.concat(gdf_list, ignore_index=True),
#     crs="EPSG:4326"
# )

# print(f"Merged {len(shore_gdf):,} features with only {KEEP[:-1]}")

# # 6) Write out as a single GeoPackage layer
# shore_gdf.to_file(
#     OUT,
#     driver="GPKG",
#     layer=LAYER
# )
