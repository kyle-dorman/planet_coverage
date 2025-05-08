from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ItemType(Enum):
    PSScene = "PSScene"
    SkySatCollect = "SkySatCollect"


class Instrument(Enum):
    PS2 = "PS2"
    PS2_SD = "PS2.SD"
    PSB_SD = "PSB.SD"
    SkySat = "SkySat"


class PublishingStage(Enum):
    Preview = "preview"
    Standard = "standard"
    Finalized = "finalized"


class QualityCategory(Enum):
    Test = "test"
    Standard = "standard"


class AssetType(Enum):
    ortho_sr = "ortho_sr"
    ortho = "ortho"
    basic = "basic"
    ortho_pansharpened = "ortho_pansharpened"


@dataclass
class QueryConfig:
    # Path the grid gpfk file
    grid_path: Path = Path("/updateme")

    # Path to a grid geojson file usd to filter the grid_path (for testing)
    filter_grid_path: Path | None = None

    # Path to save the results
    save_dir: Path = Path("/updateme")

    # The type of scene
    item_type: ItemType = ItemType.PSScene

    # Asset Type
    asset_type: AssetType = AssetType.basic

    # Base name for Planet UDM search requests
    udm_search_name: str = "udm2_coverage_search"

    # Stage of imagegry data
    publishing_stage: PublishingStage | None = None

    # Quality Category
    quality_category: QualityCategory | None = None

    # Max number of UDMs to consider
    udm_limit: int = 1000000

    # Max tasks in flight at a time
    max_concurrent_tasks: int = 1000

    # Number of times to retry downloading imagegry data
    download_retries_max: int = 3

    # Seconds to wait before retrying
    download_backoff: float = 1.0


def validate_config(config: QueryConfig):
    """Validate the QueryConfig is compatable with planet api.

    Args:
        config (QueryConfig): The config

    Raises:
        RuntimeError: If it is invalid.
    """
    # Verify we can create a valid asset string
    _ = planet_asset_string(config)


def planet_asset_string(config: QueryConfig) -> str:
    if config.item_type == ItemType.PSScene:
        if config.asset_type == AssetType.ortho_sr:
            return "ortho_analytic_4b_sr"
        elif config.asset_type == AssetType.ortho:
            return "ortho_analytic_4b"
        elif config.asset_type == AssetType.basic:
            return "basic_analytic_4b"
        else:
            raise RuntimeError(f"Unexpected AssetType {config.asset_type}")
    elif config.item_type == ItemType.SkySatCollect:
        if config.asset_type == AssetType.ortho_sr:
            return "ortho_analytic_sr"
        elif config.asset_type == AssetType.ortho:
            return "ortho_analytic"
        elif config.asset_type == AssetType.ortho_pansharpened:
            return "ortho_pansharpened"
        else:
            raise RuntimeError(f"Unexpected AssetType {config.asset_type}")
    else:
        raise RuntimeError(f"Unexpected ItemType {config.item_type}")
