import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cartopy
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
    query_df = gpd.read_file(base / "merged_ocean_grids.gpkg").set_index("cell_id")
    grids_df = gpd.read_file(base / "merged_coastal_grids.gpkg").set_index("grid_id")
    hex_grid = gpd.read_file(base / "merged_hex_grids.gpkg").set_index("hex_id")

    return query_df, grids_df, hex_grid


def create_merged_grids(
    base: Path, shorelines: Path, hex_size: float = 1.5
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    display_crs = "EPSG:4326"
    robinson_crs = "ESRI:54030"
    sinus_crs = "ESRI:54008"

    query_df = gpd.read_file(shorelines / "ocean_grids.gpkg")
    grids_df = gpd.read_file(shorelines / "coastal_grids.gpkg").rename(columns={"cell_id": "grid_id"})
    assert query_df.crs == sinus_crs
    assert grids_df.crs == sinus_crs
    heuristics_df = pd.read_csv(base / "simulated_tidal_coverage_heuristics.csv").set_index("cell_id")

    logger.info(
        "Loaded %d open-ocean cells, %d coastal grid cells, and heuristic table with %d rows",
        len(query_df),
        len(grids_df),
        len(heuristics_df),
    )

    cell_size_m = compute_step(hex_size)
    hex_grid = (
        make_equal_area_hex_grid(cell_size_m, robinson_crs).to_crs(sinus_crs).rename(columns={"cell_id": "hex_id"})
    )

    logger.info("Generated %d equal-area hexagons", len(hex_grid))

    # Assign hex_id to query_df and grid_df
    grids_df = assign_intersection_id(grids_df, hex_grid, "grid_id", "hex_id")
    query_df = assign_intersection_id(query_df, hex_grid, "cell_id", "hex_id")

    # Assign cell_id to grid_df
    grids_df = assign_intersection_id(grids_df, query_df, "grid_id", "cell_id")

    logger.info("Finished spatial ID assignments (hex_id ↔ grid_id ↔ cell_id)")

    # Add tidal information to grids_df and query_df
    grids_df = grids_df.set_index("cell_id").join(heuristics_df, how="left").reset_index()
    query_df = query_df.set_index("cell_id").join(heuristics_df, how="left").reset_index()

    logger.info("Merged tidal heuristics into grids and query dataframes")

    # Set plot crs
    query_df = query_df.to_crs(display_crs)
    grids_df = grids_df.to_crs(display_crs)
    hex_grid = hex_grid.to_crs(display_crs)

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
    scale: str = "linear",  # "linear", "log", or "hist"
    bins: Optional[Sequence[float]] = None,  # if given, use discrete bins
    figsize: Tuple[int, int] = (12, 6),
    edgecolor: str = "black",
    linewidth: float = 0.15,
    show_coastlines: bool = False,
    show_land_ocean: bool = True,
    show_grid: bool = False,
    title: Optional[str] = None,
    title_fontsize: int = 20,
    title_padsize: int = 20,
    use_cbar_label: bool = True,
    save_path: str | Path | None = None,
    show: bool = False,
    pad_fraction: float = 0.05,  # extra space around data bounds
    filter_bounds: bool = False,
    add_color_bar: bool = True,
    ax: plt.Axes | None = None,  # existing axis → no per-plot legend
) -> None:
    """
    Plot a numeric column from a GeoDataFrame on a Cartopy map, zooming to the
    area where data exist.

    If an existing Matplotlib Axes is supplied via `ax`, the function draws on that axis and skips adding a per-plot colour bar so you can create a shared legend later.

    `pad_fraction` adds a percentage of the data extent as padding so the data
    don’t touch the frame edge.

    scale : {"linear", "log", "hist"}
        Controls the colormap normalization. "hist" triggers discrete binning.

    bins : Sequence[float] | None
        If provided, the data are coloured **discretely** using these bin
        edges (left-inclusive, right-exclusive; last bin right-inclusive).
        If ``bins`` is *None* and ``scale=="hist"``, the function will
        auto-generate 7 bins spanning *vmin…vmax* using either
        ``np.linspace`` (linear) or ``np.logspace`` (log).

    For ``scale="log"``, features whose value is ``<= 0`` are plotted in gray so
    they remain visible without breaking the logarithmic colour mapping.
    Rows whose value is NaN are *always* plotted in gray.
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

    gdf = gdf[[column, "geometry"]].copy()
    gdf["_tmp"] = numeric_vals  # temp numeric column
    column = "_tmp"

    data = gdf[column].astype(float)
    nan_mask = np.isnan(data)

    # ------------------------------------------------------------------
    # Colour range & normalisation
    # ------------------------------------------------------------------
    if vmin is None:
        vmin = np.nanmin(data[data > 0]) if scale == "log" else np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    use_bins = bins is not None or scale == "hist"

    if is_datetime:
        norm = colors.Normalize(vmin=np.nanmin(numeric_vals), vmax=np.nanmax(numeric_vals))  # type: ignore

        # human-readable ticks every N months
        def _fmt(x, _):
            return mdates.num2date(x).strftime("%Y-%m")

        formatter = ticker.FuncFormatter(_fmt)
        locator = ticker.LinearLocator(numticks=5)
        valid_mask = ~nan_mask  # all non-NaN for datetime
        cmap = plt.get_cmap(cmap)  # type: ignore
    elif use_bins:
        if bins is None:
            nbins = 7
            assert vmin is not None
            assert vmax is not None
            if scale == "log":
                bins = np.logspace(np.log10(vmin), np.log10(vmax), nbins + 1)
            else:  # linear or hist default
                bins = np.linspace(vmin, vmax, nbins + 1).tolist()
        assert bins is not None
        bins = np.asarray(bins, dtype=float)  # type: ignore
        assert bins is not None
        nbins = len(bins) - 1
        cmap = plt.get_cmap(cmap, nbins)  # type: ignore
        norm = colors.BoundaryNorm(bins, nbins)
        valid_mask = ~nan_mask
        formatter = ticker.ScalarFormatter()
        locator = ticker.FixedLocator(locs=bins)

    else:
        cmap = plt.get_cmap(cmap)  # type: ignore

        if scale == "log":
            valid_mask = (data > 0) & (~nan_mask)
            if not valid_mask.any():
                raise ValueError("Log scale selected but column contains no positive values.")

            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
            formatter = ticker.FuncFormatter(lambda y, _: f"{y:g}")
            locator = ticker.LogLocator()
        else:
            valid_mask = ~nan_mask
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            formatter = ticker.ScalarFormatter()
            if vmax is not None and vmin is not None and vmax - vmin > 10.0:
                formatter = ticker.FuncFormatter(lambda y, _: f"{y:.0f}")
            else:
                formatter = ticker.FuncFormatter(lambda y, _: f"{y:.1f}")
            locator = ticker.LinearLocator(numticks=7)

    # ------------------------------------------------------------------
    # Prepare figure / axis
    # ------------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True, subplot_kw={"projection": projection})
        provided_ax = False
    else:
        fig = ax.figure
        provided_ax = True

    if filter_bounds:
        # Compute extent from data bounds (EPSG:4326) and add a small margin
        xmin, ymin, xmax, ymax = gdf.total_bounds
        dx, dy = xmax - xmin, ymax - ymin
        if dx == 0 or dy == 0:  # degenerate case (single point / line)
            dx = dy = max(dx, dy) or 1.0  # give it 1° span to avoid zero-width
        pad_x = dx * pad_fraction
        pad_y = dy * pad_fraction
        startx = max(-180, xmin - pad_x)
        endx = min(180, xmax + pad_x)
        starty = max(-90, ymin - pad_y)
        endy = min(90, ymax + pad_y)
        ax.set_extent([startx, endx, starty, endy], crs=ccrs.PlateCarree())  # type: ignore
    else:
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())  # type: ignore

    # ------------------------------------------------------------------
    # Plot data
    # ------------------------------------------------------------------
    gdf_valid = gdf.loc[valid_mask]
    gdf_invalid = gdf.loc[~valid_mask]  # ≤0 or NaN

    if show_coastlines:
        ax.coastlines(resolution="110m", linewidth=0.3)  # type: ignore
    if show_grid:
        ax.gridlines(draw_labels=False, linewidth=0.2)  # type: ignore
    if show_land_ocean:
        ax.add_feature(cartopy.feature.OCEAN, zorder=0)  # type: ignore
        ax.add_feature(cartopy.feature.LAND, zorder=0)  # type: ignore

    if not gdf_invalid.empty:
        gdf_invalid.plot(
            ax=ax,
            color="lightgray",
            transform=ccrs.PlateCarree(),
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=1,
        )

    if not gdf_valid.empty:
        gdf_valid.plot(
            column=column,
            cmap=cmap,
            norm=norm,
            ax=ax,
            transform=ccrs.PlateCarree(),
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=2,
        )

    # Colour bar
    if add_color_bar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation="vertical",
            shrink=0.65,
            pad=0.02,
            ticks=locator,
            format=formatter,
            location="right",
        )
        if use_cbar_label:
            cbar.set_label(orig_column)
        if not use_bins and not is_datetime:
            assert vmin is not None
            assert vmax is not None
            # Explicitly set ticks for continuous non-datetime case
            ticks = locator.tick_values(vmin, vmax)
            ticks = [vmin] + [t for t in ticks if t > vmin and t < vmax] + [vmax]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([formatter(t, None) for t in ticks])

    if title:
        ax.set_title(title, pad=title_padsize, fontsize=title_fontsize)

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

    if not provided_ax:
        plt.close(fig)  # type: ignore


def make_time_between_query(year: int, pct: int, valid_only: bool, extra_filter: str | None = None) -> str:
    """
    Build the fiscal-year query for a single 12-month window.
    """

    # ------------------------------------------------------------------
    # Compute end date = start + 1 year  (no extra deps needed)
    # ------------------------------------------------------------------
    start_dt = datetime(year, 12, 1).date()
    end_dt = start_dt.replace(year=start_dt.year + 1)
    end_date = end_dt.isoformat()
    start_date = start_dt.isoformat()

    valid_filter = (
        """
        AND publishing_stage = 'finalized'
        AND quality_category = 'standard'
        AND clear_percent    > 75.0
        AND has_sr_asset
        AND ground_control
    """
        if valid_only
        else ""
    )
    if extra_filter is None:
        extra_filter = ""

    return f"""
    WITH rows AS (                       -- 1️⃣  all real samples in the window
        SELECT
            grid_id,
            EXTRACT(epoch FROM acquired) AS ts
        FROM samples_all
        WHERE
            item_type    = 'PSScene'
            AND coverage_pct > 0.5
            AND acquired BETWEEN TIMESTAMP '{start_date}' AND TIMESTAMP '{end_date}'
            {valid_filter}
            {extra_filter}
    ),

    bounds AS (                         -- 2️⃣  add the two end-markers
        -- one row per grid for the window START
        SELECT DISTINCT
            grid_id,
            EXTRACT(epoch FROM TIMESTAMP '{start_date}') AS ts
        FROM rows

        UNION ALL

        -- one row per grid for the window END
        SELECT DISTINCT
            grid_id,
            EXTRACT(epoch FROM TIMESTAMP '{end_date}')   AS ts
        FROM rows
    ),

    ordered AS (                        -- 3️⃣  combine & order in time
        SELECT grid_id, ts FROM rows
        UNION ALL
        SELECT grid_id, ts FROM bounds
    ),

    deltas AS (                         -- 4️⃣  Δt between consecutive rows
        SELECT
            grid_id,
            (ts - LAG(ts) OVER (
                PARTITION BY grid_id ORDER BY ts
            )) / 86400.0                    AS days_between
        FROM ordered
    ),

    filtered AS (                       -- 5️⃣  keep only gaps ≥ 12 h
        SELECT grid_id, days_between
        FROM deltas
        WHERE days_between >= 0.5
    )

    -- 6️⃣  percentile per grid
    SELECT
        grid_id,
        quantile_cont(days_between, 0.{pct}) AS p{pct}_days_between
    FROM filtered
    WHERE days_between IS NOT NULL
    GROUP BY grid_id
    ORDER BY grid_id;
    """


def make_time_between_hist_query(
    year: int, bins: Sequence[float], valid_only: bool, extra_filter: str | None = None
) -> str:
    """
    Build the fiscal-year query for a single 12-month window.
    """

    # ------------------------------------------------------------------
    # Compute end date = start + 1 year  (no extra deps needed)
    # ------------------------------------------------------------------
    start_dt = datetime(year, 12, 1).date()
    end_dt = start_dt.replace(year=start_dt.year + 1)
    end_date = end_dt.isoformat()
    start_date = start_dt.isoformat()

    edges_sql = "LIST_VALUE(" + ", ".join(f"{b}" for b in bins) + ")"

    valid_filter = (
        """
        AND publishing_stage = 'finalized'
        AND quality_category = 'standard'
        AND clear_percent    > 75.0
        AND has_sr_asset
        AND ground_control
    """
        if valid_only
        else ""
    )
    if extra_filter is None:
        extra_filter = ""

    return f"""
    WITH rows AS (                       -- 1️⃣  all real samples in the window
        SELECT
            grid_id,
            EXTRACT(epoch FROM acquired) AS ts
        FROM samples_all
        WHERE
            item_type    = 'PSScene'
            AND coverage_pct > 0.5
            AND acquired BETWEEN TIMESTAMP '{start_date}' AND TIMESTAMP '{end_date}'
            {valid_filter}
            {extra_filter}
    ),

    bounds AS (                         -- 2️⃣  add the two end-markers
        -- one row per grid for the window START
        SELECT DISTINCT
            grid_id,
            EXTRACT(epoch FROM TIMESTAMP '{start_date}') AS ts
        FROM rows

        UNION ALL

        -- one row per grid for the window END
        SELECT DISTINCT
            grid_id,
            EXTRACT(epoch FROM TIMESTAMP '{end_date}')   AS ts
        FROM rows
    ),

    ordered AS (                        -- 3️⃣  combine & order in time
        SELECT grid_id, ts FROM rows
        UNION ALL
        SELECT grid_id, ts FROM bounds
    ),

    deltas AS (                         -- 4️⃣  Δt between consecutive rows
        SELECT
            grid_id,
            (ts - LAG(ts) OVER (
                PARTITION BY grid_id ORDER BY ts
            )) / 86400.0                    AS days_between
        FROM ordered
    ),

    filtered AS (                       -- 5️⃣  keep only gaps ≥ 12 h
        SELECT grid_id, days_between
        FROM deltas
        WHERE days_between >= 0.5
    )

    SELECT
        histogram(
            days_between,
            {edges_sql}
        ) AS bucket
    FROM filtered
    WHERE days_between IS NOT NULL
    """


# ----------------------------------------------------------------------
# Query to count number of days with multiple captures per grid_id
# ----------------------------------------------------------------------
def make_multiple_captures_query(year: int, valid_only: bool = False) -> str:
    """
    Create a DuckDB query that counts the number of days with multiple captures per grid_id.

    Parameters
    ----------
    valid_only : bool
        If True, apply additional quality filters.
    """
    # ------------------------------------------------------------------
    # Compute end date = start + 1 year  (no extra deps needed)
    # ------------------------------------------------------------------
    start_dt = datetime(year, 12, 1).date()
    end_dt = start_dt.replace(year=start_dt.year + 1)
    end_date = end_dt.isoformat()
    start_date = start_dt.isoformat()

    valid_filter = (
        """
        AND publishing_stage = 'finalized'
        AND quality_category = 'standard'
        AND clear_percent    > 75.0
        AND has_sr_asset
        AND ground_control
        """
        if valid_only
        else ""
    )

    return f"""
    WITH filtered AS (
        SELECT
            grid_id,
            DATE_TRUNC('day', acquired) AS day,
            satellite_id
        FROM samples_all
        WHERE
            acquired >= TIMESTAMP '{start_date}'
            AND acquired < TIMESTAMP '{end_date}'
            AND item_type = 'PSScene'
            AND coverage_pct > 0.5
            {valid_filter}
    ),

    grouped AS (
        SELECT grid_id, day
        FROM filtered
        GROUP BY grid_id, day
        HAVING approx_count_distinct(satellite_id) > 1
    )

    SELECT grid_id, COUNT(*) AS multi_capture_days
    FROM grouped
    GROUP BY grid_id
    ORDER BY grid_id;
    """


# ----------------------------------------------------------------------
# Query to count number of high-frequency samples per grid_id
# ----------------------------------------------------------------------
def make_high_frequency_query(year: int, freq_minutes: int, valid_only: bool = False) -> str:
    """
    Create a DuckDB query that counts the number of high-frequency samples (Δt <= freq_minutes)
    within a given date range.

    Parameters
    ----------
    freq_minutes : int
        Maximum time difference between samples in minutes to be considered "high frequency".
    valid_only : bool
        If True, apply quality filters.
    """
    # ------------------------------------------------------------------
    # Compute end date = start + 1 year  (no extra deps needed)
    # ------------------------------------------------------------------
    start_dt = datetime(year, 12, 1).date()
    end_dt = start_dt.replace(year=start_dt.year + 1)
    end_date = end_dt.isoformat()
    start_date = start_dt.isoformat()

    freq_days = freq_minutes / (60 * 24)  # convert to days

    valid_filter = (
        """
        AND publishing_stage = 'finalized'
        AND quality_category = 'standard'
        AND clear_percent    > 75.0
        AND has_sr_asset
        AND ground_control
        """
        if valid_only
        else ""
    )

    return f"""
    WITH ordered AS (
        SELECT
            grid_id,
            EXTRACT(epoch FROM acquired) AS ts
        FROM samples_all
        WHERE
            acquired >= TIMESTAMP '{start_date}'
            AND acquired < TIMESTAMP '{end_date}'
            AND item_type = 'PSScene'
            AND coverage_pct > 0.5
            {valid_filter}
    ),

    deltas AS (
        SELECT
            grid_id,
            (ts - LAG(ts) OVER (PARTITION BY grid_id ORDER BY ts)) / 86400.0 AS delta_days
        FROM ordered
    )

    SELECT
        grid_id,
        COUNT(*) AS high_freq_count
    FROM deltas
    WHERE delta_days IS NOT NULL AND delta_days <= {freq_days}
    GROUP BY grid_id
    ORDER BY grid_id;
    """


# ----------------------------------------------------------------------
# Query to compute max number of captures in a single day per grid_id
# ----------------------------------------------------------------------
def make_max_daily_captures_query(year: int, valid_only: bool = False) -> str:
    """
    Create a DuckDB query that computes the maximum number of captures
    in a single day per grid_id.

    Parameters
    ----------
    year : int
        The starting fiscal year (beginning December 1st of that year).
    valid_only : bool
        If True, apply quality filters.
    """
    start_dt = datetime(year, 12, 1).date()
    end_dt = start_dt.replace(year=start_dt.year + 1)
    start_date = start_dt.isoformat()
    end_date = end_dt.isoformat()

    valid_filter = (
        """
        AND publishing_stage = 'finalized'
        AND quality_category = 'standard'
        AND clear_percent    > 75.0
        AND has_sr_asset
        AND ground_control
        """
        if valid_only
        else ""
    )

    return f"""
    WITH daily_counts AS (
        SELECT
            grid_id,
            DATE_TRUNC('day', acquired) AS sample_day,
            COUNT(*) AS daily_count
        FROM samples_all
        WHERE
            acquired >= TIMESTAMP '{start_date}'
            AND acquired < TIMESTAMP '{end_date}'
            AND item_type = 'PSScene'
            AND coverage_pct > 0.5
            {valid_filter}
        GROUP BY grid_id, sample_day
    )

    SELECT
        grid_id,
        MAX(daily_count) AS max_daily_captures
    FROM daily_counts
    GROUP BY grid_id
    ORDER BY grid_id;
    """


# ----------------------------------------------------------------------
# Query to build a per-grid histogram of time-between-samples
# ----------------------------------------------------------------------
def make_daily_time_between_hist_query(
    year: int,
    bins: Sequence[float],
    *,
    valid_only: bool = False,
    max_hours: float = 24.0,
) -> str:
    """
    Build a DuckDB query that returns **one row per grid_id**
    with counts of time-between-samples falling into the supplied
    ``bins``.  Bins are expressed in **days** (e.g. ``[0, 0.25, 0.5, 1.0]``),
    are left-inclusive / right-exclusive, and must be strictly
    increasing.

    Parameters
    ----------
    year : int
        Starting fiscal year (window is 1 yr from Dec 1 → Nov 30).
    bins : Sequence[float]
        Monotonically increasing bin edges in **days**.
    valid_only : bool, default=False
        Apply Planet quality filters if True.
    max_hours : float, default=24.0
        Discard Δt values larger than this (hours).

    Returns
    -------
    str
        DuckDB SQL that yields:

        | grid_id | bin_0 | bin_1 | … |

        where ``bin_k`` is the count for the k-th interval.
    """
    if len(bins) < 2:
        raise ValueError("bins must have at least two edges")

    # ------------------------------------------------------------------
    # Date window
    # ------------------------------------------------------------------
    start_dt = datetime(year, 12, 1).date()
    end_dt = start_dt.replace(year=start_dt.year + 1)
    start_date = start_dt.isoformat()
    end_date = end_dt.isoformat()

    # ------------------------------------------------------------------
    # Quality filter
    # ------------------------------------------------------------------
    valid_filter = (
        """
        AND s.publishing_stage = 'finalized'
        AND s.quality_category = 'standard'
        AND s.clear_percent    > 75.0
        AND s.has_sr_asset
        AND s.ground_control
        """
        if valid_only
        else ""
    )

    # ------------------------------------------------------------------
    # width_bucket edges  (DuckDB accepts LIST_VALUE(…))
    # ------------------------------------------------------------------
    edges_sql = "LIST_VALUE(" + ", ".join(f"{b}" for b in bins) + ")"
    max_days = max_hours / 24.0

    return f"""
    WITH ordered AS (
        SELECT
            s.grid_id,
            s.satellite_id,
            EXTRACT(epoch FROM s.acquired) AS ts
        FROM samples_all AS s
        JOIN grid_ids_tbl USING (grid_id)
        WHERE
            s.acquired >= TIMESTAMP '{start_date}'
            AND s.acquired <  TIMESTAMP '{end_date}'
            AND s.item_type  = 'PSScene'
            AND s.coverage_pct > 0.5
            {valid_filter}
    ),

    deltas AS (
        SELECT
            grid_id,
            CASE
                WHEN satellite_id <> LAG(satellite_id) OVER (
                        PARTITION BY grid_id
                        ORDER BY     ts
                    )
                THEN
                    (ts - LAG(ts) OVER (
                            PARTITION BY grid_id
                            ORDER BY     ts
                    )) / 86400.0
                ELSE NULL
            END AS delta_days
        FROM ordered
    )

    SELECT
        histogram(
            delta_days,
            {edges_sql}
        ) AS bucket
    FROM deltas
    WHERE delta_days IS NOT NULL
    AND delta_days <= {max_days}
    """


if __name__ == "__main__":
    bb = Path("/Users/kyledorman/data/planet_coverage/shorelines")
    query_df, grids_df, hex_grid = create_merged_grids(
        Path("/Users/kyledorman/data/planet_coverage/points_30km"), bb, 1.0
    )
    query_df.reset_index().to_file(bb / "merged_ocean_grids.gpkg")
    grids_df.reset_index().to_file(bb / "merged_coastal_grids.gpkg")
    hex_grid.reset_index().to_file(bb / "merged_hex_grids.gpkg")
