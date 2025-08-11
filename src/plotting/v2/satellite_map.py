import logging
import warnings
from datetime import timedelta
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import geopandas as gpd
import imageio
import pandas as pd
import tqdm
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib import ticker
from shapely import wkb

from src.gen_points_map import compute_step, make_equal_area_hex_grid
from src.geo_util import assign_intersection_id

warnings.filterwarnings("ignore")  # hide every warning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


BASE = Path("/Users/kyledorman/data/planet_coverage/24b3/")
SHORELINES = BASE.parent / "shorelines"
FIG_DIR = BASE.parent / "figs_v2" / "sat_coverage"
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Example path patterns
f_pattern = "dove/results/2024/*/*/*/data.parquet"
all_files_pattern = str(BASE / f_pattern)

# Combined list used later when we search individual files
all_parquets = list(BASE.glob(f_pattern))

if not all_parquets:
    logger.error("No parquet files found matching pattern %s", all_files_pattern)
    raise FileNotFoundError("No parquet files found")
logger.info("Found %d parquet files", len(all_parquets))

robinson_crs = "ESRI:54030"
display_crs = "EPSG:4326"
cell_size_m = compute_step(3.0)
hex_grid = make_equal_area_hex_grid(cell_size_m, robinson_crs)
hex_grid = hex_grid.to_crs(display_crs)
hex_grid = hex_grid.rename(columns={"cell_id": "hex_id"})
hex_grid = hex_grid.set_index("hex_id")

df_pd: pd.DataFrame = pd.read_parquet(all_parquets[0])
df_pd["geometry"] = df_pd["geometry_wkb"].apply(wkb.loads)  # type: ignore
df_pd = df_pd.drop(columns=["geometry_wkb"])
gdf = gpd.GeoDataFrame(df_pd, geometry="geometry", crs=display_crs)

logger.info("Query finished")

hex_id_mapper = assign_intersection_id(gdf, hex_grid.reset_index(), "id", "hex_id", include_closest=True)[
    ["id", "hex_id"]
].set_index("id")
gdf = gdf.set_index("id").join(hex_id_mapper, how="left").reset_index()
assert not gdf.hex_id.isna().any(), gdf.hex_id.isna().sum()

# Determine daily date range
min_date = gdf["acquired"].min()
max_date = gdf["acquired"].max()
date_delta = max_date - min_date
total_hours = 25  # int(math.ceil(date_delta.total_seconds() / 60 / 60))
date_list = [min_date + timedelta(hours=i) for i in range(total_hours + 1)]

img_dir = FIG_DIR / "images"
img_dir.mkdir(exist_ok=True)

norm = colors.Normalize(vmin=0, vmax=20)
cmap = plt.get_cmap("cool")


def plotplot(savedir, current, end):
    save_path = savedir / f"coverage_{current.strftime('%Y%m%d_%H')}.png"

    time_df = gdf[(gdf.acquired >= current) & (gdf.acquired < end)]
    agg = time_df.groupby("hex_id").acquired.count().rename("sample_count")
    agg = agg[agg.index >= 0]
    merged = hex_grid[["geometry"]].join(agg, how="inner")
    plot_gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=hex_grid.crs)

    fig, ax = plt.subplots(1, 1, figsize=(12, 7), subplot_kw={"projection": ccrs.Robinson()})
    ax.set_extent([-180, 180, -81, 81], crs=ccrs.PlateCarree())  # type: ignore
    ax.add_feature(cartopy.feature.OCEAN, zorder=0)  # type: ignore
    ax.add_feature(cartopy.feature.LAND, zorder=0)  # type: ignore
    # ax.set_title(current.strftime("%Y%m%d_%H"), pad=20, fontsize=20)
    if len(plot_gdf):
        plot_gdf.plot(
            column="sample_count",
            cmap=cmap,
            norm=norm,
            ax=ax,
            transform=ccrs.PlateCarree(),
            edgecolor="black",
            linewidth=0.15,
            zorder=2,
        )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        shrink=0.65,
        pad=0.02,
    )
    formatter = ticker.ScalarFormatter()
    locator = ticker.MaxNLocator(nbins=6)
    cbar.formatter = formatter
    cbar.locator = locator
    cbar.update_ticks()
    ax.set_aspect("auto")
    plt.savefig(save_path, dpi=160)
    plt.close(fig)


plotplot(FIG_DIR, min_date, max_date)
assert False
# regular for-loop over each hours
for i, current in tqdm.tqdm(enumerate(date_list), total=len(date_list)):
    end = current + timedelta(hours=1)
    plotplot(img_dir, current, end)

# ---- Build video from the saved PNGs using imageio ----
image_files = sorted(img_dir.glob("coverage_*.png"))
if image_files:
    vid_path = FIG_DIR / "coverage_timelapse.mp4"
    fps = 2  # frames per second

    with imageio.get_writer(vid_path, mode="I", fps=fps, codec="libx264") as writer:
        for img_path in tqdm.tqdm(image_files):
            frame = imageio.imread(img_path)
            writer.append_data(frame)

    logger.info("Video saved to %s", vid_path)
else:
    logger.warning("No images found to create video.")

logger.info("Done")
