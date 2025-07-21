import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import AsyncIterator

import click
import geopandas as gpd
import polars as pl
from dotenv import find_dotenv, load_dotenv
from planet import DataClient, Session, data_filter
from tqdm.asyncio import tqdm_asyncio

from src.config import ItemType, QueryConfig, planet_asset_string
from src.geo_util import polygon_to_geojson_dict
from src.util import (
    check_and_create_env,
    create_config,
    retry_task,
    setup_logger,
)

logger = logging.getLogger(__name__)


def get_grid_save_path(grid_save_dir: Path, index: int) -> Path:
    hex_id = f"{index:06x}"  # unique 6‑digit hex, e.g. '0f1a2b'
    d1, d2, d3 = hex_id[:2], hex_id[2:4], hex_id[4:6]
    grid_save_path = grid_save_dir / d1 / d2 / d3
    grid_save_path.mkdir(exist_ok=True, parents=True)
    return grid_save_path


# Asynchronously performs a search with the given search filter.
# Returns a list of items found by the search.
def do_search(
    sess: Session, cell_id: int, search_filter: dict, grid_geojson: dict, config: QueryConfig
) -> AsyncIterator[dict]:
    logger.debug(f"Search for udm2 matches for cell_id {cell_id}")

    items = DataClient(sess).search(
        item_types=[config.item_type.value],
        search_filter=search_filter,
        geometry=grid_geojson,
        limit=config.udm_limit,
    )

    logger.debug(f"Executed udm2 search for cell_id {cell_id}")

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

    # Set item type
    item_filter = data_filter.string_in_filter("item_type", [config.item_type.value])

    filters = [
        geom_filter,
        date_range_filter,
        asset_filter,
        item_filter,
    ]

    # Set quality filter
    # filters.append(data_filter.not_filter(data_filter.std_quality_filter()))
    # filters.append(data_filter.not_filter(data_filter.string_in_filter("quality_category", ["test"])))

    # combine all of the filters
    all_filters = data_filter.and_filter(filters)

    return all_filters


# Data class for DataFrame rows
@dataclass
class DataFrameRow:
    acquired: datetime
    item_type: str  # ItemType
    instrument: bool  # Instrument
    clear_percent: bool
    cloud_cover: bool
    ground_control: bool
    publishing_stage: bool  # PublishingStage
    assets: bool

    @classmethod
    @lru_cache(1)
    def polars_schema(cls) -> dict:
        return {
            "acquired": pl.Datetime,
            "item_type": pl.Enum(ItemType),
            "instrument": pl.Boolean,
            "clear_percent": pl.Boolean,
            "cloud_cover": pl.Boolean,
            "ground_control": pl.Boolean,
            "publishing_stage": pl.Boolean,
            "assets": pl.Boolean,
        }


def dataframe_row(item: dict) -> DataFrameRow | None:
    props = item["properties"]

    has_assets = "assets" in item
    has_clear_percent = "clear_percent" in props
    has_cloud_cover = "cloud_cover" in props
    has_instrument = "instrument" in props
    has_ground_control = "ground_control" in props
    has_publishing_stage = "publishing_stage" in props

    if all([has_clear_percent, has_cloud_cover, has_assets, has_instrument, has_ground_control, has_publishing_stage]):
        return None

    return DataFrameRow(
        acquired=datetime.fromisoformat(props["acquired"]),
        item_type=ItemType(props["item_type"]).value,
        instrument=has_instrument,
        clear_percent=has_clear_percent,
        cloud_cover=has_cloud_cover,
        ground_control=has_ground_control,
        publishing_stage=has_publishing_stage,
        assets=has_assets,
    )


async def process_cell(
    sess: Session,
    config: QueryConfig,
    search_filter: dict,
    grid_geojson: dict,
    cell_id: int,
) -> None:
    """Download results for 1 geometry and save the results to a parquet file.

    Args:
        sess (Session): The planet Session
        config (QueryConfig): The QueryConfig
        search_filter (dict): the filter for the search request
        grid_geojson (dict): the geometry as a geojson
        cell_id (int): The cells's index
    """
    try:
        grid_save_path = get_grid_save_path(config.save_dir, cell_id)
        results_path = grid_save_path / "data.parquet"

        marker_path = grid_save_path / ".ran"  # sentinel file

        # skip if we've already produced a results file *or* the sentinel
        if results_path.exists() or marker_path.exists():
            return

        async def _collect_lazy():
            lazy = do_search(
                sess=sess, cell_id=cell_id, config=config, grid_geojson=grid_geojson, search_filter=search_filter
            )
            return [i async for i in lazy]

        item_list = await retry_task(_collect_lazy, config.download_retries_max, config.download_backoff)
        data = [dataframe_row(item=item) for item in item_list]
        data = [d for d in data if d is not None]
        df = pl.DataFrame(data, schema=DataFrameRow.polars_schema())  # type: ignore

        # If no items were returned, create an empty sentinel file instead of
        # writing an empty Parquet. The presence of '.ran' marks this cell as
        # processed.
        if not item_list or not len(data):
            marker_path.touch()
            return

        # Save as Parquet using Polars (automatically optimizes types)
        df.write_parquet(
            results_path,
        )

    except Exception as e:
        logger.error(f"Cell {cell_id} failed")
        logger.exception(e)


# Gets a list of all UDMs which match grid points and start/end date range.
async def run_queries(sess: Session, config: QueryConfig, start_date: datetime, end_date: datetime) -> None:
    # Load dataframe
    logger.info("Loading grid")
    gdf = gpd.read_file(config.grid_path)
    gdf_wgs = gdf.to_crs("EPSG:4326")
    valid = gdf_wgs.is_valid
    num_invalid = (~valid).sum()
    logger.info(f"Loaded grid with {len(gdf_wgs) - num_invalid} valid and {num_invalid} invalid geometries.")
    gdf_wgs = gdf_wgs[valid]

    if config.filter_grid_path is not None:
        logger.info(f"Filtering grid using {config.filter_grid_path}")
        filter_geo = gpd.read_file(config.filter_grid_path).to_crs("EPSG:4326").union_all()
        gdf_wgs = gdf_wgs[gdf_wgs.intersects(filter_geo)]
        logger.info(f"Filtered grid to {len(gdf_wgs)} polygons")

    # download items with limited concurrency and one progress bar
    sem = asyncio.Semaphore(config.max_concurrent_tasks)

    async def download_all_items():
        async def worker(row):
            cell_id = int(row.cell_id)
            grid_geojson = polygon_to_geojson_dict(row.geometry)
            search_filter = create_search_filter(start_date, end_date, grid_geojson, config)

            async with sem:
                await process_cell(
                    sess=sess,
                    config=config,
                    search_filter=search_filter,
                    grid_geojson=grid_geojson,
                    cell_id=cell_id,
                )

        tasks = [asyncio.create_task(worker(row)) for _, row in gdf_wgs.iterrows()]
        for task in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Download Items", dynamic_ncols=True):
            await task

    await download_all_items()


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
