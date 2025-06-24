import logging
import math
import warnings
from datetime import timedelta
from pathlib import Path

import duckdb
import geopandas as gpd
import imageio
import tqdm

from src.plotting.util import create_merged_grids, plot_gdf_column

warnings.filterwarnings("ignore")  # hide every warning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


BASE = Path("/Users/kyledorman/data/planet_coverage/points_30km/")  # <-- update this
SHORELINES = BASE.parent / "shorelines"
FIG_DIR = BASE.parent / "figs_v2" / "sat_coverage"
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Example path patterns
f_pattern = "*/results/2024/*/*/*/data.parquet"
all_files_pattern = str(BASE / f_pattern)

# Combined list used later when we search individual files
all_parquets = list(BASE.glob(f_pattern))

if not all_parquets:
    logger.error("No parquet files found matching pattern %s", all_files_pattern)
    raise FileNotFoundError("No parquet files found")
logger.info("Found %d parquet files", len(all_parquets))


query_df, grids_df, hex_grid = create_merged_grids(BASE, SHORELINES, hex_size=3.0)
hex_grid = hex_grid[hex_grid.index.isin(grids_df.hex_id.unique())].copy()

# --- Connect to DuckDB ---
con = duckdb.connect()

# Register a view for all files
con.execute(
    f"""
    CREATE OR REPLACE VIEW samples_all AS
    SELECT * FROM read_parquet('{all_files_pattern}');
"""
)
logger.info("Registered DuckDB view 'samples_all'")

query = """
SELECT *
FROM samples_all
WHERE
    acquired            >= TIMESTAMP '2024-12-01'
    AND acquired        < TIMESTAMP '2024-12-02'
    AND item_type       = 'PSScene'
    AND satellite_id    = '24b3'
;
"""

df = con.execute(query).fetchdf().set_index("cell_id")
hex_df = query_df[["hex_id"]].join(df, how="inner")

logger.info("Query finished")

# Determine daily date range
min_date = hex_df["acquired"].min()
max_date = hex_df["acquired"].max()
date_delta = max_date - min_date
total_minutes = int(math.ceil(date_delta.total_seconds() / 60))
date_list = [min_date + timedelta(minutes=i) for i in range(total_minutes + 1)]

# regular for‑loop over each minute
for current in tqdm.tqdm(date_list):
    save_path = FIG_DIR / f"coverage_{current.strftime('%Y%m%d_%H%M')}.png"
    if save_path.exists():
        continue
    end = current + timedelta(minutes=1)

    time_df = hex_df[(hex_df.acquired >= current) & (hex_df.acquired < end)]
    agg = time_df.groupby("hex_id").acquired.count().rename("sample_count")
    agg = agg[agg.index >= 0]
    merged = hex_grid[["geometry"]].join(agg, how="inner")
    gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=hex_grid.crs)

    plot_gdf_column(
        gdf,
        "sample_count",
        show_land_ocean=True,
        vmin=0,
        vmax=1,
        use_cbar_label=False,
        save_path=save_path,
    )

# ---- Build video from the saved PNGs using imageio ----
image_files = sorted(FIG_DIR.glob("coverage_*.png"))
if image_files:
    vid_path = FIG_DIR / "coverage_timelapse.mp4"
    fps = 4  # frames per second

    with imageio.get_writer(vid_path, mode="I", fps=fps, codec="libx264") as writer:
        for img_path in tqdm.tqdm(image_files):
            frame = imageio.imread(img_path)
            writer.append_data(frame)

    logger.info("Video saved to %s", vid_path)
else:
    logger.warning("No images found to create video.")
