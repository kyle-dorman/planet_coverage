import asyncio
import logging
import multiprocessing as mp
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Type

import geopandas as gpd
from omegaconf import OmegaConf
from shapely.geometry import MultiPoint, Point, Polygon, mapping
from tqdm.asyncio import tqdm_asyncio
from tqdm.std import tqdm

from src.config import QueryConfig, validate_config

logger = logging.getLogger(__name__)


# Retry wrapper around an async function that may fail
async def retry_task(task_func, retries: int, retry_delay: float) -> Any:
    attempt = 0
    while attempt < retries:
        try:
            return await task_func()
        except Exception as e:
            attempt += 1
            if attempt < retries:
                wait_time = retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                await asyncio.sleep(wait_time)
            else:
                raise e  # Return the error after max retries


def setup_logger(save_dir: Path | None = None, log_filename: str = "log.log"):
    # Configure third-party loggers
    logging.getLogger("planet").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to prevent duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if a save directory is provided)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        file_handler = logging.FileHandler(save_dir / log_filename)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Confirm setup
    root_logger.info("Logger initialized. Logging to console%s.", f" and {save_dir / log_filename}" if save_dir else "")


def write_env_file(api_key: str, env_path: Path = Path(".env")) -> None:
    """
    Writes the provided API key to a .env file with the variable name PL_API_KEY.
    """
    with env_path.open("w") as file:
        file.write(f"PL_API_KEY={api_key}\n")
    logger.info(f"API key saved to {env_path}")


def check_and_create_env(env_path: Path = Path(".env")) -> None:
    """
    Checks if the .env file exists. If not, prompts the user for an API key and writes it to the file.
    """
    if env_path.exists():
        logger.info(f"🔎 {env_path} already exists. No action needed.")
    else:
        api_key = input("Enter your Planet API key: ").strip()
        assert api_key, "Must pass an API Key!"

        write_env_file(api_key, env_path)


def create_config(config_file: Path) -> QueryConfig:
    base_config = OmegaConf.structured(QueryConfig)
    override_config = OmegaConf.load(config_file)
    config: QueryConfig = OmegaConf.merge(base_config, override_config)  # type: ignore

    validate_config(config)

    assert config.grid_path.exists(), f"grid_path {config.grid_path} does not exist!"

    config.save_dir.mkdir(exist_ok=True, parents=True)

    # Save the configuration to a YAML file
    OmegaConf.save(config, config.save_dir / "config.yaml")

    return config


# Load the correct progress bar class based on the run context (notebook vs CLI)
def get_tqdm(use_async: bool) -> Type[tqdm]:
    if use_async:
        return tqdm_asyncio
    else:
        return tqdm


def has_crs(geojson_path: Path) -> None:
    """Verify geojson file has a CRS

    Args:
        geojson_path (Path): _description_
    """
    gdf = gpd.read_file(geojson_path)
    assert gdf.crs is not None, "{} is missing a CRS"


def check_all_has_crs(paths: list[Path], workers: int):
    """
    Parallelize has_crs over a list of Path objects.
    Errors out on the first failure.
    """
    this_tqdm = get_tqdm(use_async=False)
    # use fork instead of spawn to avoid semaphore leaks on macOS
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        # executor.map will raise the first exception it encounters
        for _ in this_tqdm(
            pool.imap_unordered(has_crs, paths, chunksize=100),
            total=len(paths),
            desc="Checking CRS",
        ):
            pass
    logger.info(f"✅ All {len(paths)} files have a CRS.")


def is_within_n_days(target_date: datetime, date_list: Iterable[datetime], n_days: int) -> bool:
    """
    Returns True if target_date is within n_days of any date in date_list.

    Args:
        target_date (datetime): The date to compare.
        date_list (list of datetime): List of other dates.
        n_days (int): Number of days as threshold.

    Returns:
        bool: True if within n_days of any date in the list.
    """
    return any(abs(target_date - dt) <= timedelta(days=n_days) for dt in date_list)


def polygon_to_geojson_dict(polygon: Polygon, properties: dict | None = None) -> dict:
    """
    Wrap a single Shapely Polygon into a GeoJSON-like dict
    that geopandas.read_file or GeoDataFrame.from_features can consume.
    """
    if properties is None:
        properties = {}

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": properties,
                "geometry": mapping(polygon),
            }
        ],
    }


def make_cell_geom(pt_list: list[Point]) -> Point | Polygon:
    if len(pt_list) == 1:
        # only one point → keep it
        return pt_list[0]
    else:
        # many points → convex hull around their MultiPoint
        mp = MultiPoint([p.coords[0] for p in pt_list])
        return mp.convex_hull  # type: ignore
