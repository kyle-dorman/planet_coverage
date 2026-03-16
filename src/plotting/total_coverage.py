import logging
import warnings
from datetime import datetime
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd

from src.plotting.util import load_grids, plot_gdf_column

warnings.filterwarnings("ignore")  # hide every warning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


BASE = Path("/Users/kyledorman/data/planet_coverage/points_30km/")
SHORELINES = BASE.parent / "shorelines"
FIG_DIR = BASE.parent / "figs" / "total_coverage"
FIG_DIR.mkdir(exist_ok=True, parents=True)

# path patterns
f_pattern = "*/coastal_results/*/*/*/coastal_points.parquet"
all_files_pattern = str(BASE / f_pattern)

# Combined list used later when we search individual files
all_parquets = list(BASE.glob(f_pattern))

if not all_parquets:
    logger.error("No parquet files found matching pattern %s", all_files_pattern)
    raise FileNotFoundError("No parquet files found")
logger.info("Found %d parquet files", len(all_parquets))


query_df, grids_df, hex_grid = load_grids(SHORELINES)
MIN_DIST = 20.0
valid = ~grids_df.is_land & ~grids_df.dist_km.isna() & (grids_df.dist_km < MIN_DIST)
grids_df = grids_df[valid].copy()

assert grids_df.crs is not None
inter = gpd.sjoin(
    hex_grid.to_crs(grids_df.crs).reset_index()[["geometry", "hex_id"]], grids_df.reset_index()[["geometry", "grid_id"]]
)
counts = inter[["hex_id", "grid_id"]].groupby("hex_id").count()
hex_grid["grid_count"] = 0.0
hex_grid.loc[counts.index, "grid_count"] = counts.grid_id

logger.info("Loaded grid dataframes")

# --- Connect to DuckDB ---
con = duckdb.connect()

# Register a view for all files
con.execute(f"""
    CREATE OR REPLACE VIEW samples_all AS
    SELECT * FROM read_parquet('{all_files_pattern}');
""")
logger.info("Registered DuckDB view 'samples_all'")


def dove_coverage():
    logger.info("Plotting PSScene")

    query = """
        SELECT
            grid_id,
            COUNT(id) AS sample_count,
        FROM samples_all
        WHERE
            item_type           = 'PSScene'
            AND coverage_pct    > 0.5
            AND acquired        <  TIMESTAMP '2025-01-01'
            AND acquired        >  TIMESTAMP '2016-01-01'
        GROUP BY grid_id
    """

    df = con.execute(query).fetchdf().set_index("grid_id")

    logger.info("Query finished")

    grids_data_df = grids_df.join(df, how="left").fillna(0.0)

    logger.info("Plotting Counts")
    agg = grids_data_df.groupby("hex_id").agg(
        median_count=("sample_count", "median"),
        max_count=("sample_count", "max"),
        sum_count=("sample_count", "sum"),
    )
    agg = agg[agg.index >= 0].join(hex_grid[["geometry", "grid_count"]])
    for key in ["median_count", "max_count", "sum_count"]:
        agg.loc[agg.sum_count == 0, key] = np.nan
    gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=grids_df.crs)

    plot_gdf_column(
        gdf=gdf,
        column="median_count",
        title="Median PlanetScope Sample Count (2016-2024)",
        title_fontsize=15,
        vmin=10,
        vmax=3000,
        save_path=FIG_DIR / "median_dove.png",
        use_cbar_label=False,
    )

    logger.info("Saving results to ShapeFile")
    (FIG_DIR / "hex_data").mkdir(exist_ok=True)
    gdf.to_file(FIG_DIR / "hex_data" / "data.shp")

    (FIG_DIR / "grid_data").mkdir(exist_ok=True)
    gdf = gpd.GeoDataFrame(grids_data_df, geometry="geometry", crs=grids_df.crs)
    gdf.to_file(FIG_DIR / "grid_data" / "data.shp")


def dove_yearly_coverage():
    logger.info("Plotting Yearly Dove Coverage")

    for valid_only in [False, True]:
        hex_dfs = []
        grid_dfs = []

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

        for year in range(2013, 2025):
            start_dt = datetime(year, 1, 1).date()
            end_dt = start_dt.replace(year=start_dt.year + 1)
            end_date = end_dt.isoformat()
            start_date = start_dt.isoformat()

            query = f"""
                SELECT
                    grid_id,
                    COUNT(id)                       AS sample_count,
                FROM samples_all
                WHERE
                    item_type           = 'PSScene'
                    AND coverage_pct    > 0.5
                    AND acquired BETWEEN TIMESTAMP '{start_date}' AND TIMESTAMP '{end_date}'
                    {valid_filter}
                GROUP BY grid_id
                ORDER BY grid_id
            """
            df = con.execute(query).fetchdf().set_index("grid_id")

            logger.info(f"Query finished {year} valid_only={valid_only}")

            grids_data_df = (
                grids_df[["cell_id", "dist_km", "is_land", "is_coast", "hex_id", "geometry"]]
                .join(df, how="left")
                .fillna(0.0)
            )
            grids_data_df["year"] = year
            grids_data_df["valid"] = valid_only
            agg = grids_data_df.groupby("hex_id").agg(
                median_count=("sample_count", "median"),
                sum_count=("sample_count", "sum"),
            )
            agg = agg[agg.index >= 0].join(hex_grid[["geometry", "grid_count"]])
            agg.loc[agg.median_count == 0, "median_count"] = np.nan
            gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=grids_df.crs)
            gdf["year"] = year
            gdf["valid"] = valid_only

            valid_title = "Valid" if valid_only else "All Data"
            valid_name = "valid" if valid_only else "all_data"

            plot_gdf_column(
                gdf=gdf,
                column="median_count",
                title=f"Median PlanetScope Sample Count - {year} - {valid_title}",
                title_fontsize=15,
                vmin=1,
                vmax=480,
                save_path=FIG_DIR / f"median_dove_{year}_{valid_name}.png",
                use_cbar_label=False,
            )

            hex_dfs.append(gdf)
            gdf = gpd.GeoDataFrame(grids_data_df, geometry="geometry", crs=grids_df.crs)
            grid_dfs.append(gdf)

        logger.info("Saving results to ShapeFile")
        hex_dir = FIG_DIR / ("hex_data_by_year_valid" if valid_only else "hex_data_by_year")
        hex_dir.mkdir(exist_ok=True)
        gdf = gpd.GeoDataFrame(pd.concat(hex_dfs), geometry="geometry", crs=hex_dfs[0].crs)
        gdf.to_file(hex_dir / "data.shp")

        grid_dir = FIG_DIR / ("grid_data_by_year_valid" if valid_only else "grid_data_by_year")
        grid_dir.mkdir(exist_ok=True)
        gdf = gpd.GeoDataFrame(pd.concat(grid_dfs), geometry="geometry", crs=grid_dfs[0].crs)
        gdf.to_file(grid_dir / "data.shp")


def dove_seasonal_coverage():
    logger.info("Plotting Seasonal Dove Coverage")

    hex_dfs = []
    grid_dfs = []

    season_months = {
        "spring": (3, 4, 5),
        "summer": (6, 7, 8),
        "fall": (9, 10, 11),
        "winter": (12, 1, 2),
    }

    for year in [2024, None]:
        if year is not None:
            start_dt = datetime(year, 1, 1).date()
            end_dt = start_dt.replace(year=start_dt.year + 1)

        else:
            start_dt = datetime(2016, 1, 1)
            end_dt = datetime(2025, 1, 1)

        end_date = end_dt.isoformat()
        start_date = start_dt.isoformat()

        for valid_only in [True, False]:
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
            for season in ["spring", "summer", "fall", "winter"]:
                months = season_months[season]
                month_list = ",".join(str(m) for m in months)

                query = f"""
                    SELECT
                        grid_id,
                        COUNT(id)                       AS sample_count,
                    FROM samples_all
                    WHERE
                        item_type           = 'PSScene'
                        AND coverage_pct    > 0.5
                        AND acquired BETWEEN TIMESTAMP '{start_date}' AND TIMESTAMP '{end_date}'
                        AND EXTRACT(MONTH FROM acquired) IN ({month_list})
                        {valid_filter}
                    GROUP BY grid_id
                    ORDER BY grid_id
                """

                df = con.execute(query).fetchdf().set_index("grid_id")

                logger.info(f"Query finished {year} {season}")

                grids_data_df = (
                    grids_df[["cell_id", "dist_km", "is_land", "is_coast", "hex_id", "geometry"]]
                    .join(df, how="left")
                    .fillna(0.0)
                )
                grids_data_df["season"] = season
                grids_data_df["year"] = year
                grids_data_df["valid"] = valid_only

                agg = grids_data_df.groupby("hex_id").agg(
                    median_count=("sample_count", "median"),
                    sum_count=("sample_count", "sum"),
                )
                agg = agg[agg.index >= 0].join(hex_grid[["geometry", "grid_count"]])
                agg.loc[agg.sum_count == 0, "median_count"] = np.nan
                gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=grids_df.crs)
                gdf["season"] = season
                gdf["year"] = year
                gdf["valid"] = valid_only

                valid_title = "Valid" if valid_only else "All Data"
                valid_name = "valid" if valid_only else "all_data"
                if year is not None:
                    title_extra = f"{season.capitalize()} - {year} - {valid_title}"
                    year_name = str(year)
                else:
                    title_extra = f"{season.capitalize()} - All Years - {valid_title}"
                    year_name = "all_years"

                if year is None:
                    if valid_only:
                        vmax = 360
                    else:
                        vmax = 900
                else:
                    if valid_only:
                        vmax = 60
                    else:
                        vmax = 180

                plot_gdf_column(
                    gdf=gdf,
                    column="median_count",
                    title=f"Median PlanetScope Sample Count ({title_extra})",
                    title_fontsize=15,
                    vmin=1,
                    vmax=vmax,
                    save_path=FIG_DIR / f"median_dove_{season}_{year_name}_{valid_name}.png",
                    use_cbar_label=False,
                )

                hex_dfs.append(gdf)
                gdf = gpd.GeoDataFrame(grids_data_df, geometry="geometry", crs=grids_df.crs)
                grid_dfs.append(gdf)

    logger.info("Saving results to ShapeFile")
    (FIG_DIR / "hex_data_by_season_year_valid").mkdir(exist_ok=True)
    gdf = gpd.GeoDataFrame(pd.concat(hex_dfs), geometry="geometry")
    gdf.to_file(FIG_DIR / "hex_data_by_season_year_valid" / "data.shp")

    (FIG_DIR / "grid_data_by_season_year_valid").mkdir(exist_ok=True)
    gdf = gpd.GeoDataFrame(pd.concat(grid_dfs), geometry="geometry")
    gdf.to_file(FIG_DIR / "grid_data_by_season_year_valid" / "data.shp")


def skysat_coverage():
    logger.info("Plotting Skysat")

    query = """
        SELECT
            grid_id,
            COUNT(id) AS sample_count,
        FROM samples_all
        WHERE
            item_type           = 'SkySatCollect'
            AND coverage_pct    > 0.5
            AND acquired        <  TIMESTAMP '2025-01-01'
            AND acquired        >  TIMESTAMP '2016-01-01'
        GROUP BY grid_id
    """

    df = con.execute(query).fetchdf().set_index("grid_id")

    logger.info("Query finished")

    near_grids_df = grids_df[(grids_df.dist_km < 4.0) & ~grids_df.is_land]
    grids_data_df = near_grids_df.join(df, how="left").fillna(0.0)

    print("% Grids with atleast 1 samples")
    print(round(100 * (grids_data_df.sample_count > 0).sum() / len(grids_data_df), 2))

    print("% Grids with atleast 5 samples")
    print(round(100 * (grids_data_df.sample_count > 4).sum() / len(grids_data_df), 2))

    print("% Grids with atleast 100 samples")
    print(round(100 * (grids_data_df.sample_count > 99).sum() / len(grids_data_df), 2))

    logger.info("Plotting Counts")
    agg = grids_data_df.groupby("hex_id").agg(
        sum_count=("sample_count", "sum"),
        max_count=("sample_count", "max"),
    )
    agg = agg[agg.index >= 0].join(hex_grid[["geometry", "grid_count"]])
    for key in ["sum_count", "max_count"]:
        agg.loc[agg.sum_count == 0, key] = np.nan
    gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=grids_df.crs)

    plot_gdf_column(
        gdf=gdf,
        column="sum_count",
        title="Sum SkySat Sample Count (2016-2024)",
        title_fontsize=15,
        scale="log",
        # vmin=10,
        # vmax=4500,
        save_path=FIG_DIR / "sky_sat_sum.png",
        use_cbar_label=False,
    )
    plot_gdf_column(
        gdf=gdf,
        column="max_count",
        title="Max SkySat Sample Count (2016-2024)",
        title_fontsize=15,
        scale="log",
        # vmin=10,
        vmax=1800,
        save_path=FIG_DIR / "sky_sat_max.png",
        use_cbar_label=False,
    )

    logger.info("Saving results to ShapeFile")
    (FIG_DIR / "skysat_hex_data").mkdir(exist_ok=True)
    gdf.to_file(FIG_DIR / "hex_data" / "data.shp")

    (FIG_DIR / "skysat_grid_data").mkdir(exist_ok=True)
    gdf = gpd.GeoDataFrame(grids_data_df, geometry="geometry", crs=grids_df.crs)
    gdf.to_file(FIG_DIR / "grid_data" / "data.shp")


dove_coverage()
# skysat_coverage()
dove_yearly_coverage()
dove_seasonal_coverage()

logger.info("Done")
