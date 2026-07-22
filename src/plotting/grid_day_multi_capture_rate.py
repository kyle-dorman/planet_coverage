"""Count coastal grid-days with multiple qualifying PlanetScope captures.

This is the denominator companion to ``multi_capture_counter.py``. A captured
grid-day is a unique ``(grid_id, solar day)`` with at least one qualifying
capture. A multi-capture grid-day has captures from more than one distinct
satellite on that solar day, matching the satellite-level treatment used by
the existing within-day interval histogram.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd

LOGGER = logging.getLogger(__name__)
DEFAULT_DATA_ROOT = Path("/Volumes/x10pro/planet_coverage")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2025, help="Exclusive end year")
    parser.add_argument("--max-distance-km", type=float, default=20.0)
    parser.add_argument("--clear-percent", type=float, default=75.0)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: <data-root>/figs/multi_capture",
    )
    return parser.parse_args()


def sql_literal(value: str | Path) -> str:
    """Quote a string for use as a DuckDB SQL literal."""
    return "'" + str(value).replace("'", "''") + "'"


def load_selected_grid_ids(shorelines_dir: Path, max_distance_km: float) -> pd.DataFrame:
    grids = gpd.read_file(shorelines_dir / "merged_coastal_grids.gpkg")
    selected = grids.loc[
        ~grids["is_land"].fillna(True) & grids["dist_km"].notna() & (grids["dist_km"] < max_distance_km),
        ["grid_id"],
    ].drop_duplicates()
    if selected.empty:
        raise ValueError("The coastal-grid filter selected no grid IDs")
    return pd.DataFrame({"grid_id": selected["grid_id"].astype("uint32")})


def make_daily_counts_sql(start_year: int, end_year: int, clear_percent: float) -> str:
    start_date = date(start_year, 1, 1).isoformat()
    end_date = date(end_year, 1, 1).isoformat()
    return f"""
        CREATE OR REPLACE TEMP TABLE daily_capture_counts AS
        WITH satellite_days AS (
            SELECT
                s.grid_id,
                YEAR(s.acquired)::INTEGER AS acquisition_year,
                DATE_TRUNC('day', s.solar_time)::DATE AS solar_day,
                s.satellite_id
            FROM samples_all AS s
            SEMI JOIN selected_grid_ids AS g USING (grid_id)
            WHERE
                s.acquired >= TIMESTAMP '{start_date}'
                AND s.acquired < TIMESTAMP '{end_date}'
                AND s.item_type = 'PSScene'
                AND s.coverage_pct > 0.5
                AND s.publishing_stage = 'finalized'
                AND s.quality_category = 'standard'
                AND s.clear_percent > {clear_percent}
                AND s.has_sr_asset
                AND s.ground_control
                AND s.satellite_id IS NOT NULL
            GROUP BY
                s.grid_id,
                acquisition_year,
                solar_day,
                s.satellite_id
        )
        SELECT
            grid_id,
            acquisition_year,
            solar_day,
            COUNT(*)::USMALLINT AS distinct_satellite_count
        FROM satellite_days
        GROUP BY grid_id, acquisition_year, solar_day
    """


def summarize(
    con: duckdb.DuckDBPyConnection,
    grid_count: int,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual = con.execute("""
        SELECT
            acquisition_year AS year,
            COUNT(*)::UBIGINT AS captured_grid_days,
            COUNT_IF(distinct_satellite_count > 1)::UBIGINT AS multi_capture_grid_days
        FROM daily_capture_counts
        GROUP BY acquisition_year
        ORDER BY acquisition_year
        """).fetchdf()

    expected_years = pd.DataFrame({"year": range(start_year, end_year)})
    annual = expected_years.merge(annual, on="year", how="left").fillna(0)
    annual[["captured_grid_days", "multi_capture_grid_days"]] = annual[
        ["captured_grid_days", "multi_capture_grid_days"]
    ].astype("uint64")
    annual["eligible_grids"] = grid_count
    annual["calendar_days"] = annual["year"].map(lambda year: (date(year + 1, 1, 1) - date(year, 1, 1)).days)
    annual["all_eligible_grid_days"] = annual["eligible_grids"] * annual["calendar_days"]
    annual["multi_capture_pct_of_captured"] = 100 * annual["multi_capture_grid_days"] / annual["captured_grid_days"]
    annual["multi_capture_pct_of_all_eligible"] = (
        100 * annual["multi_capture_grid_days"] / annual["all_eligible_grid_days"]
    )

    overall = pd.DataFrame(
        {
            "year": ["overall"],
            "captured_grid_days": [annual["captured_grid_days"].sum()],
            "multi_capture_grid_days": [annual["multi_capture_grid_days"].sum()],
            "eligible_grids": [grid_count],
            "calendar_days": [annual["calendar_days"].sum()],
            "all_eligible_grid_days": [annual["all_eligible_grid_days"].sum()],
        }
    )
    overall["multi_capture_pct_of_captured"] = 100 * overall["multi_capture_grid_days"] / overall["captured_grid_days"]
    overall["multi_capture_pct_of_all_eligible"] = (
        100 * overall["multi_capture_grid_days"] / overall["all_eligible_grid_days"]
    )
    summary = pd.concat([annual, overall], ignore_index=True)

    distribution = con.execute("""
        SELECT
            distinct_satellite_count,
            COUNT(*)::UBIGINT AS grid_days
        FROM daily_capture_counts
        GROUP BY distinct_satellite_count
        ORDER BY distinct_satellite_count
        """).fetchdf()
    distribution["pct_of_captured_grid_days"] = 100 * distribution["grid_days"] / distribution["grid_days"].sum()
    return summary, distribution


def main() -> None:
    args = parse_args()
    if args.end_year <= args.start_year:
        raise ValueError("--end-year must be greater than --start-year")

    output_dir = args.output_dir or args.data_root / "figs" / "multi_capture"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / ".duckdb_tmp"
    temp_dir.mkdir(exist_ok=True)

    parquet_dir = args.data_root / "points_30km" / "dove" / "coastal_results"
    parquet_pattern = parquet_dir / "*" / "*" / "*" / "coastal_points.parquet"
    if next(parquet_dir.glob("*/*/*/coastal_points.parquet"), None) is None:
        raise FileNotFoundError(f"No Parquet files found at {parquet_pattern}")

    selected_grid_ids = load_selected_grid_ids(args.data_root / "shorelines", args.max_distance_km)
    LOGGER.info("Selected %s coastal grids", f"{len(selected_grid_ids):,}")

    con = duckdb.connect()
    con.execute(f"SET memory_limit = {sql_literal(args.memory_limit)}")
    con.execute(f"SET threads = {args.threads}")
    con.execute(f"SET temp_directory = {sql_literal(temp_dir)}")
    con.register("selected_grid_ids", selected_grid_ids)
    con.execute(
        f"CREATE VIEW samples_all AS SELECT * FROM read_parquet({sql_literal(parquet_pattern)}, union_by_name=true)"
    )

    LOGGER.info("Counting captured grid-days from %d through %d", args.start_year, args.end_year - 1)
    con.execute(make_daily_counts_sql(args.start_year, args.end_year, args.clear_percent))
    summary, distribution = summarize(con, len(selected_grid_ids), args.start_year, args.end_year)

    summary_path = output_dir / "grid_day_multi_capture_summary.csv"
    distribution_path = output_dir / "grid_day_capture_count_distribution.csv"
    summary.to_csv(summary_path, index=False, float_format="%.8f")
    distribution.to_csv(distribution_path, index=False, float_format="%.8f")

    overall = summary.iloc[-1]
    LOGGER.info("Wrote %s", summary_path)
    LOGGER.info("Wrote %s", distribution_path)
    print(
        f"{int(overall.multi_capture_grid_days):,} of "
        f"{int(overall.captured_grid_days):,} captured grid-days "
        f"({overall.multi_capture_pct_of_captured:.3f}%) had captures from multiple satellites."
    )
    print(
        f"They represent {overall.multi_capture_pct_of_all_eligible:.3f}% of all "
        f"{int(overall.all_eligible_grid_days):,} eligible grid-days in the study window."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
