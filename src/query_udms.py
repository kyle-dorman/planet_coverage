import asyncio
import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import AsyncIterator

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
from dotenv import find_dotenv, load_dotenv
from planet import DataClient, Session, data_filter
from shapely.geometry import Point, Polygon, shape

from src.config import ItemType, QueryConfig, planet_asset_string
from src.util import (
    check_and_create_env,
    create_config,
    get_tqdm,
    make_cell_geom,
    polygon_to_geojson_dict,
    retry_task,
    setup_logger,
)

logger = logging.getLogger(__name__)


# Asynchronously creates a search request with the given search filter. Returns the created search request.
async def create_search(sess: Session, search_name: str, search_filter: dict, config: QueryConfig) -> dict:
    logger.debug(f"Creating search request {search_name}")

    async def create_search_inner():
        return await DataClient(sess).create_search(
            name=search_name,
            search_filter=search_filter,
            item_types=[config.item_type.value],
        )

    search_request = await retry_task(create_search_inner, config.download_retries_max, config.download_backoff)

    logger.debug(f"Created search request {search_name} {search_request['id']}")

    return search_request


# Asynchronously performs a search using the given search request.
# Returns a list of items found by the search.
def do_search(sess: Session, search_request: dict, config: QueryConfig) -> AsyncIterator[dict]:
    logger.debug(f"Search for udm2 matches {search_request['id']}")

    items = DataClient(sess).run_search(search_id=search_request["id"], limit=config.udm_limit)

    logger.debug(f"Executed udm2 search {search_request['id']}")

    return items


# Define the search filters used to find the UDMs
def create_search_filter(start_date: datetime, end_date: datetime, grid_geojson: dict, config: QueryConfig) -> dict:
    # geometry filter
    geom_filter = data_filter.geometry_filter(grid_geojson)

    # date range filter
    date_range_filter = data_filter.date_range_filter("acquired", gte=start_date, lt=end_date)

    # Asset filter
    asset_type = planet_asset_string(config)
    asset_filter = data_filter.asset_filter([asset_type])

    # Only get data we can access
    permission_filter = data_filter.permission_filter()

    # Set item type
    item_filter = data_filter.string_in_filter("item_type", [config.item_type.value])

    filters = [
        geom_filter,
        date_range_filter,
        asset_filter,
        permission_filter,
        item_filter,
    ]

    # Set publishing level filter
    if config.item_type == ItemType.PSScene:
        publishing_filter = data_filter.string_in_filter("publishing_stage", [config.publishing_stage])
        filters.append(publishing_filter)

    # combine all of the filters
    all_filters = data_filter.and_filter(filters)

    return all_filters


# Asynchronously performs a search for imagery using the given geometry, and start/end dates.
# Returns a list of items found by the search.
async def search(
    sess: Session,
    grid_geojson: dict,
    index: int,
    config: QueryConfig,
    grid_save_path: Path,
    start_date: datetime,
    end_date: datetime,
) -> AsyncIterator[dict]:
    search_filter = create_search_filter(start_date, end_date, grid_geojson, config)

    search_name = f"{config.udm_search_name}_{index}_{start_date.date().isoformat()}_{end_date.date().isoformat()}"
    search_request = await create_search(sess, search_name, search_filter, config)
    with open(grid_save_path / "search_request.json", "w") as f:
        json.dump(search_request, f)

    # search for matches
    return do_search(sess, search_request, config)


# Data class for DataFrame rows
@dataclass
class DataFrameRow:
    has_8_channel: bool
    id: str
    acquired: datetime
    clear_percent: float
    item_type: str
    quality_category: str
    satellite_azimuth: float
    satellite_id: str
    sun_azimuth: float
    sun_elevation: float
    view_angle: float
    instrument: str
    grid_idx: int

    @classmethod
    @lru_cache(1)
    def polars_schema(cls) -> dict:
        return {
            "has_8_channel": pl.Boolean,
            "id": pl.Utf8,
            "acquired": pl.Datetime,
            "clear_percent": pl.Float32,
            "item_type": pl.Categorical,
            "quality_category": pl.Categorical,
            "satellite_azimuth": pl.Float32,
            "satellite_id": pl.Categorical,
            "sun_azimuth": pl.Float32,
            "sun_elevation": pl.Float32,
            "view_angle": pl.Float32,
            "instrument": pl.Categorical,
            "grid_idx": pl.UInt32,
        }


def dataframe_row(item: dict, index: int) -> DataFrameRow:
    props = item["properties"]
    clear_percent = props.get("clear_percent", 1 - int(props.get("cloud_cover", 0) * 100))

    return DataFrameRow(
        has_8_channel="ortho_analytic_8b_sr" in item.get("assets", {}),
        id=item["id"],
        acquired=datetime.fromisoformat(props["acquired"]),
        clear_percent=clear_percent,
        item_type=props["item_type"],
        quality_category=props["quality_category"],
        satellite_azimuth=props["satellite_azimuth"],
        satellite_id=props["satellite_id"],
        sun_azimuth=props["sun_azimuth"],
        sun_elevation=props["sun_elevation"],
        view_angle=props["view_angle"],
        instrument=props.get("instrument", "SkySat"),
        grid_idx=index,
    )


async def process_cell(
    sess: Session,
    config: QueryConfig,
    start_date: datetime,
    end_date: datetime,
    poly: Polygon | Point,
    cell_df: gpd.GeoDataFrame,
    index: int,
    pbar,
) -> None:
    """Download results for 1 cell (N points) and save the results to a parquet file.

    Args:
        sess (Session): The planet Session
        config (QueryConfig): The QueryConfig
        start_date (datetime): start
        end_date (datetime): end
        poly (Polygon): The search area
        cell_df (GeoDataFrame): The dataframe of points belonging to this cell
        index (int): The cells's index
        pbar (_type_): The progress bar
    """
    try:
        hex_id = f"{index:06x}"  # unique 6‑digit hex, e.g. '0f1a2b'
        d1, d2, d3 = hex_id[:2], hex_id[2:4], hex_id[4:6]
        grid_save_path = config.save_dir / d1 / d2 / d3
        grid_save_path.mkdir(exist_ok=True, parents=True)

        results_path = grid_save_path / "data.parquet"

        if results_path.exists():
            pbar.update(1)
            return

        grid_geojson = polygon_to_geojson_dict(poly)  # type: ignore

        async def _collect_lazy():
            now = datetime.now()
            lazy = await search(sess, grid_geojson, index, config, grid_save_path, start_date, end_date)
            result = [i async for i in lazy]
            time_diff = datetime.now() - now
            logger.info(f"Took {time_diff.seconds} to download {len(result)} items for grid {index}")
            return result

        item_list = await retry_task(_collect_lazy, config.download_retries_max, config.download_backoff)

        rows = []
        if len(item_list):
            # Build spatial index once outside loop
            sindex = cell_df.sindex

            for item in item_list:
                base_row = dataframe_row(item, 0)
                item_geom = shape(item["geometry"])  # still need once

                possible_matches = sindex.intersection(item_geom.bounds)
                matching = cell_df.iloc[possible_matches]

                for idx in matching[matching.intersects(item_geom)].index:
                    rows.append(replace(base_row, grid_idx=idx))

        df = pl.DataFrame(rows, schema=DataFrameRow.polars_schema())  # type: ignore

        # Save as Parquet using Polars (automatically optimizes types)
        df.write_parquet(
            results_path,
        )

    except Exception as e:
        logger.error(f"Grid {index} failed")
        logger.exception(e)

    pbar.update(1)


# Gets a list of all UDMs which match grid points and start/end date range.
async def run_queries(sess: Session, config: QueryConfig, start_date: datetime, end_date: datetime) -> None:
    # Load dataframe
    gdf = gpd.read_file(config.grid_path)
    assert isinstance(gdf.index, pd.RangeIndex), "Expected a simple RangeIndex"

    # convert to new crs
    gdf = gdf.to_crs("EPSG:4326")

    # Prepare for groupings
    degree_size = config.degree_size
    gdf["lon_bin"] = (np.floor(gdf.geometry.x / degree_size) * degree_size).astype(float)
    gdf["lat_bin"] = (np.floor(gdf.geometry.y / degree_size) * degree_size).astype(float)
    gdf["grid_index"] = gdf.index

    # ----------------------------------------------------------------------------
    # 1) group by cell and collect both points AND their original indices
    # ----------------------------------------------------------------------------
    grouped = (
        gdf.groupby(["lon_bin", "lat_bin"])
        .agg(
            {
                "geometry": list,  # list of Point geometries
                "grid_index": list,  # list of original indices
            }
        )
        .reset_index()
    )
    grouped["cell_geom"] = grouped["geometry"].apply(make_cell_geom)  # type: ignore
    cells = gpd.GeoDataFrame(grouped, geometry="cell_geom", crs="EPSG:4326")
    total = len(cells)

    tqdm = get_tqdm(use_async=True)
    with tqdm(total=total, desc="Grids", position=0, dynamic_ncols=True) as pbar:
        coros = []

        for index, row in cells.iterrows():
            cell = row["cell_geom"]
            orig_idxs = row["grid_index"]  # the list of points in this cell
            subset = gdf.take(orig_idxs)  # very fast positional slicing
            subset = subset.set_geometry("geometry")  # ensure geometry col

            task = asyncio.create_task(
                process_cell(
                    sess=sess,
                    config=config,
                    start_date=start_date,
                    end_date=end_date,
                    poly=cell,
                    cell_df=subset,
                    index=index,  # type: ignore
                    pbar=pbar,
                )
            )
            coros.append(task)
        _ = await asyncio.gather(*coros)


# Main loop. Query all overlapping UDMs for a given date and set of grid points.
async def main_loop(config: QueryConfig, start_date: datetime, end_date: datetime) -> None:
    async with Session() as sess:
        await run_queries(sess, config, start_date, end_date)


def query_udms(
    config_file: Path,
    start_date: datetime,
    end_date: datetime,
):
    config = create_config(config_file)

    setup_logger(config.save_dir, log_filename="query_udms.log")

    logger.info(
        f"Querying UDMs for start_date={start_date} end_date={end_date} grid={config.grid_path} to={config.save_dir}"
    )

    return asyncio.run(main_loop(config, start_date, end_date))


@click.command()
@click.option("-c", "--config-file", type=click.Path(exists=True), required=True)
@click.option(
    "-s",
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date in YYYY-MM-DD format.",
    required=True,
)
@click.option(
    "-e", "--end-date", type=click.DateTime(formats=["%Y-%m-%d"]), help="End date in YYYY-MM-DD format.", required=True
)
def main(
    config_file: Path,
    start_date: datetime,
    end_date: datetime,
):
    config_file = Path(config_file)

    # Set the PlanetAPI Key in .env file if not set
    check_and_create_env()

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv(raise_error_if_not_found=True))

    query_udms(config_file=config_file, start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    main()
