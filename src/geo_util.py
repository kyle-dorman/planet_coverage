import logging

import folium
import geopandas as gpd
import pandas as pd
import polars as pl
import shapely
from branca.colormap import linear
from pyproj import CRS
from shapely import wkb

logger = logging.getLogger(__name__)


def clean_invalid(df: gpd.GeoDataFrame):
    # Clean invalid geometries inplace
    invalid = ~df.geometry.is_valid
    df.loc[invalid, "geometry"] = df.geometry[invalid].apply(lambda geom: shapely.make_valid(geom))
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
    assert not df.geometry.is_empty.any()

    # Project
    df = df.to_crs(proj_crs)

    # Remove GeometryCollection types
    non_polygon = df.geometry.geom_type == "GeometryCollection"
    df.loc[non_polygon, "geometry"] = df.geometry[non_polygon].buffer(0)

    # Clean
    clean_invalid(df)

    # Simplify
    df.geometry = df.geometry.simplify(tol, preserve_topology=True)

    # Clean
    clean_invalid(df)
    return df


def finalize_strips(
    geometries,
    proj_crs: str,
) -> gpd.GeoDataFrame:
    """
    Apply final CRS, drop empty geometries, and remove internal holes.
    """
    df = gpd.GeoDataFrame(geometry=geometries, crs=proj_crs)

    # Remove empty
    df = df[~df.geometry.is_empty]

    # Clean
    clean_invalid(df)

    return df


def load_coastal(path: str, proj_crs: str) -> gpd.GeoDataFrame:
    """
    Load coastal strips GeoPackage and verify CRS.
    """
    logger.info("Loading coastal strips from %s", path)
    coastal = gpd.read_file(path)
    assert coastal.crs is not None
    assert coastal.crs.to_string() == proj_crs, f"Expected CRS {proj_crs}, got {coastal.crs}"
    assert len(coastal) == 1, "No coastal features found"
    return coastal


def filter_satellite_pct_intersection(
    lazy_df: pl.LazyFrame,
    polygon: shapely.Polygon,
    crs: CRS,
    overlap_pct: float,
) -> pl.DataFrame:
    # materialize, decode and filter with pandas, then convert back to lazy Polars
    df_pd = lazy_df.collect().to_pandas()
    df_pd["geometry"] = df_pd["geometry_wkb"].apply(wkb.loads)  # type: ignore
    gdf = gpd.GeoDataFrame(df_pd, geometry="geometry", crs="EPSG:4326").to_crs(crs)
    pct_intersection = gdf.intersection(polygon).area / gdf.geometry.area
    df_pd = df_pd[pct_intersection > overlap_pct]
    return pl.from_pandas(df_pd.drop(columns=["geometry"]))


def dedup_satellite_captures(
    lazy_df: pl.LazyFrame,
    max_duration_hrs: int,
    column_name: str,
) -> pl.DataFrame:
    """Deduplicate satellite captures for a single grid point / polygon.

    Captures can overlap but we don't want to double count these in small windows of time

    Args:
        df (pl.LazyFrame): The lazy dataframe
        max_duration_hrs (int): The max duration between frames before this is considered a different capture instance.
        column_name (str): The name of the grid column.

    Returns:
        pl.DataFrame: The fully realized dataframe.
    """
    threshold = pl.duration(hours=max_duration_hrs)

    df_best = (
        lazy_df.sort([column_name, "satellite_id", "acquired"])  # still lazy
        .with_columns(
            [
                # 1) time since previous capture
                pl.col("acquired")
                .diff()
                .over([column_name, "satellite_id"])
                .alias("delta"),
            ]
        )
        .with_columns(
            [
                # 2) flag new groups
                ((pl.col("delta").is_null()) | (pl.col("delta") > threshold))
                .cast(pl.UInt32)
                .alias("new_group"),
            ]
        )
        .with_columns(
            [
                # 3) cumulative sum _in that window_ → group IDs
                pl.col("new_group").cum_sum().over([column_name, "satellite_id"]).alias("group_id"),  # <— use cumsum
                # 4) your ranking columns (unchanged)
                pl.when(pl.col("quality_category") == "standard").then(1).otherwise(0).alias("quality_rank"),
                pl.when(pl.col("publishing_stage") == "finalized")
                .then(3)
                .when(pl.col("publishing_stage") == "standard")
                .then(2)
                .when(pl.col("publishing_stage") == "preview")
                .then(1)
                .otherwise(0)
                .alias("publishing_rank"),
            ]
        )
        # 5) sort best‐first in each group
        .sort(
            by=[column_name, "satellite_id", "group_id", "quality_rank", "publishing_rank", "clear_percent"],
            descending=[False, False, False, True, True, True],
        )
        # 6) take the first (best) row per group
        .group_by([column_name, "satellite_id", "group_id"], maintain_order=True)
        .head(1)
        # 7) drop your helper columns
        .select(pl.exclude(["delta", "new_group", "group_id", "quality_rank", "publishing_rank"]))
        .collect()  # execute the query
    )

    return df_best


def plot_df(df, column_name, title, zoom=5, round_digits=0):
    # --- Folium map for % ---
    if df[column_name].max() == df[column_name].min():
        scale_min = 0
    else:
        scale_min = df[column_name].min()
    color_scale = linear.viridis.scale(scale_min, df[column_name].max())  # type: ignore

    m = folium.Map(
        location=[df.geometry.centroid.y.mean(), df.geometry.centroid.x.mean()],
        zoom_start=zoom,
        tiles="CartoDB positron",
        width=1000,  # type: ignore
        height=800,  # type: ignore
    )

    for grid_id, row in df.iterrows():
        value = row[column_name]
        geom = row.geometry
        folium.GeoJson(
            data=geom,
            style_function=lambda f, col=color_scale(value): {
                "fillColor": col,
                "color": col,  # outline same as fill
                "weight": 1,
                "fillOpacity": 0.8,
            },
            tooltip=f"{grid_id}<br>{column_name}: {round(value, round_digits)}",
        ).add_to(m)

    color_scale.caption = title
    color_scale.add_to(m)

    return m


def assign_intersection_id(
    gdf: gpd.GeoDataFrame,
    other_gdf: gpd.GeoDataFrame,
    left_key: str,
    right_key: str,
    equal_area_crs: str,
    include_closest: bool = False,
) -> gpd.GeoDataFrame:
    assert gdf.crs == equal_area_crs

    gdf = gdf.copy()
    gdf["poly_area"] = gdf.geometry.area
    gdf = gdf.to_crs(other_gdf.crs)  # type: ignore

    # Assign right_key to gdf
    joined = gdf[[left_key, "geometry", "poly_area"]].overlay(other_gdf[[right_key, "geometry"]], how="intersection")
    joined = joined.to_crs(equal_area_crs)
    joined["overlap_pct"] = joined.geometry.area / joined.poly_area
    joined = joined.sort_values(by=[left_key, "overlap_pct"], ascending=[True, False])
    joined = joined.drop_duplicates(subset=left_key)

    if include_closest:
        matched_left_keys = joined[left_key].unique()
        missing_rows = gdf[~gdf[left_key].isin(matched_left_keys)]

        # for any missing, assign nearest cell
        if len(missing_rows):
            logger.info(f"{len(missing_rows)} rows lack overlap; assigning nearest cell")
            nearest = gpd.sjoin_nearest(
                missing_rows[[left_key, "geometry"]],
                other_gdf[[right_key, "geometry"]],
                how="left",
            )
            # combine overlap and nearest
            joined = pd.concat([joined[[left_key, right_key]], nearest[[left_key, right_key]]], ignore_index=True)

    gdf = gdf.drop(columns="poly_area").set_index(left_key)
    gdf = gdf.join(joined.set_index(left_key)[[right_key]], how="left")

    invalid = gdf[right_key].isna()
    gdf.loc[invalid, right_key] = -1
    gdf[right_key] = gdf[right_key].astype(int)
    gdf = gdf.to_crs(equal_area_crs)

    return gdf.reset_index()
