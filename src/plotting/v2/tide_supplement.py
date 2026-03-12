import logging
import warnings
from pathlib import Path

import duckdb

warnings.filterwarnings("ignore")  # hide every warning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


BASE = Path("/Users/kyledorman/data/planet_coverage/points_30km/")  # <-- update this

# Example path patterns
f_pattern = "dove/coastal_results/*/*/*/coastal_points.parquet"
all_files_pattern = str(BASE / f_pattern)

# Combined list used later when we search individual files
all_parquets = list(BASE.glob(f_pattern))

if not all_parquets:
    logger.error("No parquet files found matching pattern %s", all_files_pattern)
    raise FileNotFoundError("No parquet files found")
logger.info("Found %d parquet files", len(all_parquets))


FIG_DIR = BASE.parent / "figs_v2" / "tide_supplement"
FIG_DIR.mkdir(exist_ok=True, parents=True)

# --- Connect to DuckDB ---
con = duckdb.connect()

# Register a view for all files
con.execute(f"""
    CREATE OR REPLACE VIEW samples_all AS
    SELECT * FROM read_parquet('{all_files_pattern}');
""")
logger.info("Registered DuckDB view 'samples_all'")

low_tide_grid = 12652104
high_tide_grid = 12862252
tide_range_grid = 12903762

query = """
SELECT
    grid_id,
    acquired,
    tide_height
FROM samples_all
WHERE
        acquired         >= TIMESTAMP '2017-12-01'
    AND acquired         <  TIMESTAMP '2024-12-01'
    AND item_type        = 'PSScene'
    AND coverage_pct     > 0.5
    AND has_tide_data
    AND grid_id IN (12652104, 12862252, 12903762)
ORDER BY grid_id;
"""

logger.info("Run Query")

df = con.execute(query).fetchdf().set_index("grid_id")
logger.info("Queried tide levels")

df.to_csv(FIG_DIR / "all_results.csv")


query = """
SELECT
    grid_id,
    acquired,
    tide_height
FROM samples_all
WHERE
        acquired         >= TIMESTAMP '2017-12-01'
    AND acquired         <  TIMESTAMP '2024-12-01'
    AND item_type        = 'PSScene'
    AND publishing_stage = 'finalized'
    AND quality_category = 'standard'
    AND coverage_pct     > 0.5
    AND clear_percent    > 75.0
    AND has_sr_asset
    AND ground_control
    AND has_tide_data
    AND grid_id IN (12652104, 12862252, 12903762)
ORDER BY grid_id;
"""

logger.info("Run Query")

df = con.execute(query).fetchdf().set_index("grid_id")
logger.info("Queried filtered tide levels")

df.to_csv(FIG_DIR / "filtered_results.csv")

logger.info("Done!")
