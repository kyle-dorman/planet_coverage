import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod, Transformer
from shapely import box
from shapely.geometry import Polygon

# ---------------------------------------------------------------------------
# Additional helpers for constructing equal-area *hexagon* grids
# ---------------------------------------------------------------------------

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


def create_safe_area(crs: str, step_size: float = 0.05, shrink_m: int = 10) -> Polygon:
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    # 0.05° step (≈5 km); +1 so both endpoints appear
    lon = np.linspace(-180, 180, int(360 / step_size) + 1)
    lat = np.linspace(-90, 90, int(180 / step_size) + 1)

    ring = np.concatenate(
        [
            np.column_stack([lon, np.full(len(lon), -90)]),  # bottom edge  (lat = −90°)
            np.column_stack([np.full(len(lat), 180), lat]),  # right edge   (lon =  180°)
            np.column_stack([lon[::-1], np.full(len(lon), 90)]),  # top edge     (lat =  90°)
            np.column_stack([np.full(len(lat), -180), lat[::-1]]),  # left edge    (lon = −180°)
        ]
    )

    lon_sin, lat_sin = transformer.transform(ring[:, 0], ring[:, 1])
    sinus_poly = Polygon(np.array([lon_sin, lat_sin]).T)
    gdf_sinus = gpd.GeoDataFrame(geometry=[sinus_poly], crs=crs)
    sinus_frame_shrunk = gdf_sinus.buffer(-shrink_m)

    return sinus_frame_shrunk.geometry.iloc[0]  # type: ignore


def make_equal_area_grid(cell_size_m: float, crs: str, keep_partial: bool = False) -> gpd.GeoDataFrame:
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
    valid = np.ones([4, len(xx.ravel())], dtype=np.bool_)
    valid_distance = np.ceil(cell_size_m * np.sqrt(2))

    for i, (xoff, yoff) in enumerate(offsets):
        xpts = xx.ravel() + xoff
        ypts = yy.ravel() + yoff
        wgs_points = wgs_transformer.transform(xpts, ypts)
        crs_points2 = transformer.transform(*wgs_points)

        distances = np.hypot(crs_points2[0] - xpts, crs_points2[1] - ypts)
        valid[i] = distances < valid_distance

    all_valid = valid.all(axis=0)
    valid_pts = np.array((xx.ravel(), yy.ravel())).T[all_valid]
    polys = [box(x - half, y - half, x + half, y + half) for x, y in valid_pts]
    grid_box = gpd.GeoDataFrame(geometry=polys, crs=crs)

    if keep_partial:
        some_valid = valid.any(axis=0) & ~all_valid
        valid_pts = np.array((xx.ravel(), yy.ravel())).T[some_valid]
        polys = [box(x - half, y - half, x + half, y + half) for x, y in valid_pts]
        grid_partial = gpd.GeoDataFrame(geometry=polys, crs=crs)
        safe_poly = create_safe_area(crs=crs)
        grid_partial["geometry"] = grid_partial.geometry.intersection(safe_poly)

        grid_box = pd.concat([grid_box, grid_partial], ignore_index=True)
        assert isinstance(grid_box, gpd.GeoDataFrame)

    grid_box["cell_id"] = grid_box.index

    return grid_box


# ---------------------------------------------------------------------------
# Hexagon grid helpers
# ---------------------------------------------------------------------------
def _regular_hexagon(center_x: float, center_y: float, a: float) -> Polygon:
    """
    Construct a corner-topped regular hexagon (6 vertices) centred on (x, y)
    with *side length* ``a`` in the *projected* CRS coordinates.
    """
    # corner-topped orientation: 0°, 60°, … 300°
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    verts = [(center_x + a * np.cos(th), center_y + a * np.sin(th)) for th in angles]
    return Polygon(verts)


def _hex_side_from_equal_area(target_area: float) -> float:
    """
    Given a target *square-cell* area (``target_area`` = ``cell_size_m**2``),
    return the side length ``a`` for a regular hexagon with *equal area*.

    Area_hex = (3 * sqrt(3) / 2) * a²
    => a = sqrt( 2 * target_area / (3 * sqrt(3)) )
    """
    return np.sqrt(2 * target_area / (3 * np.sqrt(3)))


def make_equal_area_hex_grid(
    cell_size_m: float, crs: str, shift_x: float = 0.0, shift_y: float = 0.0
) -> gpd.GeoDataFrame:
    """
    Generate a global grid of *equal-area* corner-topped hexagons whose area
    equals that of a square ``cell_size_m`` x ``cell_size_m``.

    Parameters
    ----------
    cell_size_m : float
        Nominal square side-length in metres whose *area* the hexagon will
        match.
    crs : str
        Any PROJ string / EPSG code describing the *projected* CRS in which
        the grid is built (e.g. an equal-area world projection).

    Returns
    -------
    centers_gdf : GeoDataFrame
        Points at hexagon centres with column ``cell_id``.
    hex_gdf : GeoDataFrame
        Polygon geometries for each hexagon with matching ``cell_id``.
    """
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    wgs_transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    # Compute hexagon geometry parameters
    target_area = cell_size_m**2
    a = _hex_side_from_equal_area(target_area)  # side length
    vertical_spacing = 3 * a / 2  # row-to-row step
    horizontal_spacing = np.sqrt(3) * a
    half_horizontal_spacing = horizontal_spacing / 2

    # Determine map extents in projected coordinates
    x_min, _ = transformer.transform(-180.0, 0.0)
    x_max, _ = transformer.transform(180.0, 0.0)
    _, y_min = transformer.transform(0.0, -90.0)
    _, y_max = transformer.transform(0.0, 90.0)
    minx, maxx = min(x_min, x_max), max(x_min, x_max)
    miny, maxy = min(y_min, y_max), max(y_min, y_max)

    # Build centres in a “staggered” grid (odd rows shifted 0.75 × width)
    centers_list = []
    row = 0
    y = miny + a  # start one side length above bottom
    while y <= maxy - a:
        # horizontal offset: even rows 0, odd rows half_width
        offset = 0 if row % 2 == 0 else half_horizontal_spacing
        x = minx + offset
        while x <= maxx - a:
            centers_list.append((x, y))
            x += horizontal_spacing
        y += vertical_spacing
        row += 1

    logger.info("Generating global hex grid: %d hexagon centres", len(centers_list))

    all_centers = np.array(centers_list)
    all_centers[:, 0] += shift_x
    all_centers[:, 1] += shift_y

    # all 6 corner offsets
    offsets = np.array(_regular_hexagon(0, 0, a).exterior.coords.xy).T
    valid = np.ones(len(all_centers), dtype=np.bool_)
    valid_distance = np.ceil(cell_size_m * np.sqrt(2))

    for xoff, yoff in offsets:
        xpts = all_centers[:, 0] + xoff
        ypts = all_centers[:, 1] + yoff
        wgs_points = wgs_transformer.transform(xpts, ypts)
        crs_points2 = transformer.transform(*wgs_points)

        distances = np.hypot(crs_points2[0] - xpts, crs_points2[1] - ypts)
        valid &= distances < valid_distance

    # Create GeoDataFrames
    centers = all_centers[valid]
    hex_polys = [_regular_hexagon(x, y, a) for x, y in centers]
    hex_gdf = gpd.GeoDataFrame(geometry=hex_polys, crs=crs)
    hex_gdf["cell_id"] = hex_gdf.index

    # (optional) validity check: ensure centres transform round-trip cleanly
    x_proj, y_proj = centers[:, 0], centers[:, 1]
    lon, lat = wgs_transformer.transform(x_proj, y_proj)
    x_back, y_back = transformer.transform(lon, lat)
    max_err = np.max(np.hypot(np.array(x_proj) - np.array(x_back), np.array(y_proj) - np.array(y_back)))
    valid = max_err < (a * 0.1)  # 10 % tolerance
    if not valid:
        logger.warning("CRS round-trip error exceeds tolerance (max %.2f m)", max_err)

    return hex_gdf
