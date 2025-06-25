from multiprocessing import Pool, cpu_count
from pathlib import Path

import click
import polars as pl
from tqdm import tqdm


def validate(pth: Path):
    try:
        _ = pl.read_parquet(pth)
    except pl.exceptions.ComputeError:
        return pth


def run(base_dir: Path):
    num_workers = max(1, cpu_count())
    with Pool(num_workers) as pool:
        for work_dir in ["ca_only", "points_30km"]:
            for sat in ["skysat", "dove"]:
                input_dir = base_dir / work_dir / sat / "coastal_results"
                tasks = list(input_dir.glob("*/*/*/coastal_points.parquet"))

                invalid = []
                for result in tqdm(pool.imap_unordered(validate, tasks), total=len(tasks), desc="Processing tasks"):
                    if result is not None:
                        invalid.append(result)

                print(work_dir, sat)
                print([f"{p.parent.parent.parent.name}{p.parent.parent.name}{p.parent.name}" for p in invalid])


@click.command()
@click.option(
    "--base-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Root directory that contains the Planet coverage subfolders.",
)
def main(base_dir: Path):
    """Validate Parquet files under BASE_DIR."""
    run(base_dir)


if __name__ == "__main__":
    main()
