import logging

import geopandas as gpd
import numpy as np
from pyproj import Geod, Transformer
from shapely import Point, box

logger = logging.getLogger(__name__)


def compute_step(degrees: float = 1.0) -> float:
    """
    Compute the east–west distance (in meters) at the equator
    corresponding to `degrees` of longitude on the WGS84 ellipsoid.
    """
    # Initialize a geodetic calculator for the WGS84 ellipsoid
    geod = Geod(ellps="WGS84")
    # Inverse geodetic between (lon0, lat0) and (lon0+degrees, lat0):
    # we pick lat0 = 0 to get equatorial distance
    lon0, lat0 = 0.0, 0.0
    lon1, lat1 = degrees, 0.0
    # https://pyproj4.github.io/pyproj/stable/api/geod.html
    _, _, distance_m = geod.inv(lon0, lat0, lon1, lat1)  # meters
    return distance_m


def make_equal_area_grid(cell_size_m: float, crs: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    wgs_transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    # Equator extremes for longitude ±180
    x_min, _ = transformer.transform(-180.0, 0.0)
    x_max, _ = transformer.transform(180.0, 0.0)
    # Pole extremes for latitude ±90 at central meridian
    _, y_min = transformer.transform(0.0, -90.0)
    _, y_max = transformer.transform(0.0, 90.0)
    # Ensure proper ordering
    minx, maxx = min(x_min, x_max), max(x_min, x_max)
    miny, maxy = min(y_min, y_max), max(y_min, y_max)

    # 3. create grid in projected coordinates
    half = cell_size_m / 2
    xs = np.arange(minx + half, maxx - half + 1e-6, cell_size_m)
    ys = np.arange(miny + half, maxy - half + 1e-6, cell_size_m)

    logger.info("Generating global grid: %d cols × %d rows = %d points", len(xs), len(ys), xs.size * ys.size)
    xx, yy = np.meshgrid(xs, ys)

    # all 4 corner offsets
    offsets = [(-half, -half), (-half, half), (half, -half), (half, half)]
    valid = np.ones(len(xx.ravel()), dtype=np.bool_)
    valid_distance = np.ceil(cell_size_m * np.sqrt(2))

    for xoff, yoff in offsets:
        xpts = xx.ravel() + xoff
        ypts = yy.ravel() + yoff
        wgs_points = wgs_transformer.transform(xpts, ypts)
        crs_points2 = transformer.transform(*wgs_points)

        distances = np.hypot(crs_points2[0] - xpts, crs_points2[1] - ypts)
        valid &= distances < valid_distance

    valid_pts = np.array((xx.ravel(), yy.ravel())).T[valid]
    centers = [Point(x, y) for x, y in valid_pts]
    gdf_pts = gpd.GeoDataFrame(geometry=centers, crs=crs)

    polys = [box(x - half, y - half, x + half, y + half) for x, y in valid_pts]
    grid_box = gpd.GeoDataFrame(geometry=polys, crs=crs)
    grid_box["cell_id"] = grid_box.index
    gdf_pts["cell_id"] = gdf_pts.index

    return gdf_pts, grid_box
