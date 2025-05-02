import logging

import geopandas as gpd
import shapely

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
