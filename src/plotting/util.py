import logging
from pathlib import Path
from typing import Optional, Tuple

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors, ticker

from src.gen_points_map import compute_step, make_equal_area_hex_grid
from src.geo_util import assign_intersection_id

logger = logging.getLogger(__name__)


def load_grids(base: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    display_crs = "EPSG:4326"
    robinson_crs = "ESRI:54030"
    sinus_crs = "ESRI:54008"

    query_df = gpd.read_file(base / "ocean_grids.gpkg")
    grids_df = gpd.read_file(base / "coastal_grids.gpkg").rename(columns={"cell_id": "grid_id"})
    heuristics_df = pd.read_csv(base / "simulated_tidal_coverage_heuristics.csv").set_index("cell_id")

    logger.info(
        "Loaded %d open‑ocean cells, %d coastal grid cells, and heuristic table with %d rows",
        len(query_df),
        len(grids_df),
        len(heuristics_df),
    )

    cell_size_m = compute_step(1.5)
    _, hex_grid = make_equal_area_hex_grid(cell_size_m, robinson_crs)
    hex_grid = hex_grid.rename(columns={"cell_id": "hex_id"}).to_crs(display_crs)

    logger.info("Generated %d equal‑area hexagons", len(hex_grid))

    # Assign hex_id to query_df and grid_df
    grids_df = assign_intersection_id(grids_df, hex_grid, "grid_id", "hex_id", sinus_crs)
    query_df = assign_intersection_id(query_df, hex_grid, "cell_id", "hex_id", sinus_crs)

    # Assign cell_id to grid_df
    grids_df = assign_intersection_id(grids_df, query_df, "grid_id", "cell_id", sinus_crs)

    logger.info("Finished spatial ID assignments (hex_id ↔ grid_id ↔ cell_id)")

    # Add tidal information to grids_df and query_df
    grids_df = grids_df.set_index("cell_id").join(heuristics_df, how="left").reset_index()
    query_df = query_df.set_index("cell_id").join(heuristics_df, how="left").reset_index()

    logger.info("Merged tidal heuristics into grids and query dataframes")

    # Set plot crs
    query_df = query_df.to_crs(display_crs)
    grids_df = grids_df.to_crs(display_crs)

    # Set indexes
    query_df = query_df.set_index("cell_id")
    grids_df = grids_df.set_index("grid_id")
    hex_grid = hex_grid.set_index("hex_id")

    return query_df, grids_df, hex_grid


def plot_gdf_column(
    gdf: gpd.GeoDataFrame,
    column: str,
    *,
    projection: ccrs.CRS = ccrs.Robinson(),
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    scale: str = "linear",  # "linear"  or  "log"
    figsize: Tuple[int, int] = (12, 6),
    edgecolor: str = "black",
    linewidth: float = 0.15,
    show_coastlines: bool = False,
    show_grid: bool = False,
    title: Optional[str] = None,
    save_path: str | Path | None = None,
    show: bool = True,
    pad_fraction: float = 0.05,  # extra space around data bounds
) -> None:
    """
    Plot a numeric column from a GeoDataFrame on a Cartopy map, zooming to the
    area where data exist.

    `pad_fraction` adds a percentage of the data extent as padding so the data
    don’t touch the frame edge.
    """
    # ------------------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------------------
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        raise ValueError("GeoDataFrame must be in EPSG:4326 (lon/lat degrees)")
    if column not in gdf.columns:
        raise KeyError(f"{column!r} not found in GeoDataFrame")

    orig_column = column

    # ------------------------------------------------------------------
    # Detect datetime columns and transform
    # ------------------------------------------------------------------
    is_datetime = np.issubdtype(gdf[column].dtype, np.datetime64)  # type: ignore
    if is_datetime:
        numeric_vals = mdates.date2num(gdf[column].values)  # float array
    else:
        numeric_vals = gdf[column].astype(float).values

    gdf = gdf.copy()
    gdf["_tmp"] = numeric_vals  # temp numeric column
    column = "_tmp"

    data = gdf[column].astype(float)

    # ------------------------------------------------------------------
    # Colour range & normalisation
    # ------------------------------------------------------------------
    if vmin is None:
        vmin = data[data > 0].min() if scale == "log" else data.min()
    if vmax is None:
        vmax = data.max()

    if is_datetime:
        norm = colors.Normalize(vmin=numeric_vals.min(), vmax=numeric_vals.max())  # type: ignore

        # human-readable ticks every N months
        def _fmt(x, _):
            return mdates.num2date(x).strftime("%Y-%m")

        formatter = ticker.FuncFormatter(_fmt)
        locator = ticker.MaxNLocator(nbins=6)  # or mdates.MonthLocator()
    else:
        # keep your linear / log branch as-is
        if scale == "log":
            if (data <= 0).any():
                raise ValueError("Log scale selected but column contains non-positive values.")
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
            formatter = ticker.FuncFormatter(lambda y, _: f"{y:g}")
            locator = ticker.LogLocator(base=10, numticks=10)
        else:
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            formatter = ticker.ScalarFormatter()
            locator = ticker.MaxNLocator(nbins=6)

    cmap = plt.get_cmap(cmap)  # type: ignore

    # ------------------------------------------------------------------
    # Prepare figure / axis
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=projection)

    # Compute extent from data bounds (EPSG:4326) and add a small margin
    xmin, ymin, xmax, ymax = gdf.total_bounds
    dx, dy = xmax - xmin, ymax - ymin
    if dx == 0 or dy == 0:  # degenerate case (single point / line)
        dx = dy = max(dx, dy) or 1.0  # give it 1° span to avoid zero-width
    pad_x = dx * pad_fraction
    pad_y = dy * pad_fraction
    ax.set_extent([xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y], crs=ccrs.PlateCarree())  # type: ignore

    # ------------------------------------------------------------------
    # Plot data
    # ------------------------------------------------------------------
    gdf.plot(
        column=column,
        cmap=cmap,
        norm=norm,
        ax=ax,
        transform=ccrs.PlateCarree(),
        edgecolor=edgecolor,
        linewidth=linewidth,
    )

    if show_coastlines:
        ax.coastlines(resolution="110m", linewidth=0.3)  # type: ignore
    if show_grid:
        ax.gridlines(draw_labels=False, linewidth=0.2)  # type: ignore

    # Colour bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.65, pad=0.02, format=formatter)
    cbar.locator = locator
    cbar.update_ticks()
    cbar.set_label(orig_column)

    if title:
        ax.set_title(title, pad=12)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

    plt.close()
