import warnings
from pathlib import Path
from typing import Sequence

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
from shapely.geometry.base import BaseGeometry
from tqdm.auto import tqdm

warnings.filterwarnings(
    "ignore",
    message=r"invalid value encountered in intersect(ion|s)",
    category=RuntimeWarning,
    module=r"shapely\..*",
)

CAPTURE_COLUMNS = [
    "id",
    "acquired",
    "cell_id",
    "has_8_channel",
    "has_sr_asset",
    "clear_percent",
    "quality_category",
    "ground_control",
    "publishing_stage",
    "geometry_wkb",
]


def partition_parts(cell_id: int) -> tuple[str, str, str]:
    """Return the nested path components used for a query-cell ID."""
    hex_id = f"{cell_id:06x}"
    return hex_id[:2], hex_id[2:4], hex_id[4:6]


def partition_paths(base_dir: Path, satellite: str, cell_id: int) -> list[Path]:
    """Find every yearly Parquet partition for a satellite and query cell."""
    d1, d2, d3 = partition_parts(cell_id)
    return sorted((base_dir / satellite / "results").glob(f"*/{d1}/{d2}/{d3}/data.parquet"))


def load_filter_geometry(path: Path) -> tuple[BaseGeometry, object]:
    """Load the capture filter in WGS84 and choose a local metric CRS."""
    filter_gdf = gpd.read_file(path)
    if filter_gdf.empty or filter_gdf.crs is None:
        raise click.ClickException("The filter GeoJSON must contain geometry with a defined CRS.")

    filter_gdf = filter_gdf.to_crs("EPSG:4326")
    filter_geometry = filter_gdf.geometry.union_all()
    if filter_geometry.is_empty:
        raise click.ClickException("The filter GeoJSON contains no non-empty geometry.")

    area_crs = filter_gdf.estimate_utm_crs()
    if area_crs is None:
        raise click.ClickException("Could not determine a projected CRS for overlap-area calculations.")
    return filter_geometry, area_crs


def relevant_cell_ids(query_grids_path: Path, filter_geometry: BaseGeometry) -> list[int]:
    """Use the query grid only to locate Parquet partitions touching the filter."""
    grids = gpd.read_file(query_grids_path)
    if grids.crs is None or "cell_id" not in grids.columns:
        raise click.ClickException("Query grids must have a CRS and a cell_id column.")

    filter_in_grid_crs = gpd.GeoSeries([filter_geometry], crs="EPSG:4326").to_crs(grids.crs).iloc[0]
    return sorted(grids.loc[grids.intersects(filter_in_grid_crs), "cell_id"].astype(int).unique().tolist())


def load_captures(
    paths: Sequence[Path],
    filter_geometry: BaseGeometry,
    clear_thresh: float,
) -> gpd.GeoDataFrame:
    """Load, quality-filter, deduplicate, and spatially filter captures."""
    frames = [pl.read_parquet(path, columns=CAPTURE_COLUMNS).to_pandas() for path in paths]
    if not frames:
        return gpd.GeoDataFrame(columns=[*CAPTURE_COLUMNS[:-1], "geometry"], geometry="geometry", crs="EPSG:4326")

    captures = pd.concat(frames, ignore_index=True).drop_duplicates(subset="id")
    captures = captures[
        (captures.clear_percent > clear_thresh)
        & (captures.publishing_stage == "finalized")
        & (captures.quality_category == "standard")
        & captures.has_sr_asset
        & captures.ground_control
    ].copy()
    if captures.empty:
        return gpd.GeoDataFrame(columns=[*CAPTURE_COLUMNS[:-1], "geometry"], geometry="geometry", crs="EPSG:4326")

    geometry = gpd.GeoSeries.from_wkb(captures.pop("geometry_wkb"), crs="EPSG:4326")
    captures = gpd.GeoDataFrame(captures, geometry=geometry, crs="EPSG:4326")
    captures = captures[captures.geometry.notna()].copy()
    captures = captures[~captures.geometry.is_empty].copy()
    invalid = ~captures.geometry.is_valid
    if invalid.any():
        captures.loc[invalid, "geometry"] = captures.loc[invalid].geometry.make_valid()
    captures["acquired"] = pd.to_datetime(captures.acquired)
    return captures[captures.intersects(filter_geometry)].reset_index(drop=True)


def temporal_pair_positions(
    dove: gpd.GeoDataFrame,
    skysat: gpd.GeoDataFrame,
    time_window_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return every Dove/SkySat row pair within the acquisition-time window."""
    dove_times = dove.acquired.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    skysat_times = skysat.acquired.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    skysat_order = np.argsort(skysat_times)
    sorted_skysat_times = skysat_times[skysat_order]
    tolerance_ns = int(time_window_hours * 3_600 * 1e9)

    dove_positions: list[np.ndarray] = []
    skysat_positions: list[np.ndarray] = []
    for dove_position, acquired_ns in enumerate(dove_times):
        left = np.searchsorted(sorted_skysat_times, acquired_ns - tolerance_ns, side="left")
        right = np.searchsorted(sorted_skysat_times, acquired_ns + tolerance_ns, side="right")
        if left == right:
            continue
        matches = skysat_order[left:right]
        dove_positions.append(np.full(len(matches), dove_position, dtype=np.int64))
        skysat_positions.append(matches)

    if not dove_positions:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.concatenate(dove_positions), np.concatenate(skysat_positions)


def intersect_cell(
    base_dir: Path,
    cell_id: int,
    filter_geometry: BaseGeometry,
    area_crs: object,
    clear_thresh: float,
    time_window_hours: float,
    overlap_area_m2: float,
) -> gpd.GeoDataFrame | None:
    """Find all qualifying Dove/SkySat intersections for one query cell."""
    dove_paths = partition_paths(base_dir, "dove", cell_id)
    skysat_paths = partition_paths(base_dir, "skysat", cell_id)
    if not dove_paths or not skysat_paths:
        return None

    dove = load_captures(dove_paths, filter_geometry, clear_thresh)
    skysat = load_captures(skysat_paths, filter_geometry, clear_thresh)
    if dove.empty or skysat.empty:
        return None

    dove_positions, skysat_positions = temporal_pair_positions(dove, skysat, time_window_hours)
    if not len(dove_positions):
        return None

    dove_rows = dove.iloc[dove_positions].reset_index(drop=True)
    skysat_rows = skysat.iloc[skysat_positions].reset_index(drop=True)
    overlap_geometry = dove_rows.geometry.intersection(skysat_rows.geometry, align=False)

    overlaps = gpd.GeoDataFrame(
        {
            "dove_id": dove_rows.id,
            "skysat_id": skysat_rows.id,
            "dove_acquired": dove_rows.acquired,
            "skysat_acquired": skysat_rows.acquired,
            "has_8_channel": dove_rows.has_8_channel,
        },
        geometry=overlap_geometry,
        crs="EPSG:4326",
    )
    overlaps = overlaps[~overlaps.geometry.is_empty].copy()
    overlaps = overlaps[overlaps.geometry.notna()].copy()
    if overlaps.empty:
        return None

    overlaps["acquired_delta_hours"] = (
        overlaps.dove_acquired - overlaps.skysat_acquired
    ).abs().dt.total_seconds() / 3_600
    overlaps["overlap_area_m2"] = overlaps.to_crs(area_crs).geometry.area
    overlaps = overlaps[overlaps.overlap_area_m2 > overlap_area_m2].copy()
    return overlaps if not overlaps.empty else None


def format_datetimes_for_geojson(overlaps: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Store UTC acquisition times as portable ISO-8601 strings."""
    overlaps = overlaps.copy()
    for column in ("dove_acquired", "skysat_acquired"):
        overlaps[column] = overlaps[column].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return overlaps


@click.command()
@click.option(
    "--base-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root directory containing dove/results and skysat/results.",
)
@click.option(
    "--filter-geojson",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="GeoJSON region that both source capture footprints must intersect.",
)
@click.option(
    "--query-grids-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Query grid used only to select relevant Parquet cell partitions.",
)
@click.option(
    "--output-path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination GeoJSON file.",
)
@click.option(
    "--time-window-hours",
    default=12.0,
    show_default=True,
    type=click.FloatRange(min=0.0, min_open=True),
    help="Maximum absolute acquisition-time difference, inclusive.",
)
@click.option(
    "--clear-thresh",
    default=75.0,
    show_default=True,
    type=click.FloatRange(min=0.0, max=100.0),
    help="Minimum clear_percent for both capture types.",
)
@click.option(
    "--overlap-area-m2",
    default=300**2,
    show_default=True,
    type=click.FloatRange(min=0.0),
    help="Minimum Dove/SkySat footprint-overlap area in square meters.",
)
def main(
    base_dir: Path,
    filter_geojson: Path,
    query_grids_path: Path,
    output_path: Path,
    time_window_hours: float,
    clear_thresh: float,
    overlap_area_m2: float,
) -> None:
    """Write all qualifying Dove/SkySat footprint intersections to one GeoJSON."""
    if output_path.suffix.lower() not in {".geojson", ".json"}:
        raise click.ClickException("--output-path must end in .geojson or .json")

    filter_geometry, area_crs = load_filter_geometry(filter_geojson)
    cell_ids = relevant_cell_ids(query_grids_path, filter_geometry)
    if not cell_ids:
        raise click.ClickException("The filter GeoJSON does not intersect any query-grid cells.")

    click.echo(f"Processing {len(cell_ids)} query cells...")
    results = []
    for cell_id in tqdm(cell_ids, desc="Intersecting cells"):
        result = intersect_cell(
            base_dir=base_dir,
            cell_id=cell_id,
            filter_geometry=filter_geometry,
            area_crs=area_crs,
            clear_thresh=clear_thresh,
            time_window_hours=time_window_hours,
            overlap_area_m2=overlap_area_m2,
        )
        if result is not None:
            results.append(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        output_path.write_text('{"type":"FeatureCollection","features":[]}\n')
        click.echo(f"No qualifying intersections. Wrote empty GeoJSON to {output_path}")
        return

    overlaps = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs="EPSG:4326")
    overlaps = overlaps.drop_duplicates(subset=["dove_id", "skysat_id"]).sort_values(
        ["dove_acquired", "skysat_acquired"]
    )
    overlaps = format_datetimes_for_geojson(overlaps)
    overlaps.to_file(output_path, driver="GeoJSON", engine="pyogrio")
    click.echo(f"Wrote {len(overlaps):,} intersections to {output_path}")


if __name__ == "__main__":
    main()
