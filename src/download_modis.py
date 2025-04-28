#!/usr/bin/env python3
"""
download_modis.py

Download every granule of the MODIS Land Cover product (MCD12Q1 v061)
using NASA's earthaccess library.
"""

import logging
import os
from pathlib import Path

import click
import earthaccess

# -------------------------------------------------------------------
# SET UP LOGGING
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--output-dir", "-o", required=True, help="Directory to save granules")
@click.option("--short-name", default="MCD12Q1", show_default=True, help="EarthAccess short name")
@click.option("--version", default="061", show_default=True, help="EarthAccess collection version")
@click.option("--start-date", default="2019-01-01", show_default=True, help="Temporal start date (YYYY-MM-DD)")
@click.option("--end-date", default="2020-01-01", show_default=True, help="Temporal end date (YYYY-MM-DD)")
@click.option("--page-size", default=20000, show_default=True, type=int, help="Number of granules per CMR request")
@click.option("--earthdata-username", default=None, help="NASA Earthdata username (optional)")
@click.option("--earthdata-password", default=None, help="NASA Earthdata password (optional)")
def main(output_dir, short_name, version, start_date, end_date, page_size, earthdata_username, earthdata_password):
    # Set Earthdata credentials if provided
    if earthdata_username:
        os.environ["EARTHDATA_USERNAME"] = earthdata_username
    if earthdata_password:
        os.environ["EARTHDATA_PASSWORD"] = earthdata_password

    # Validate credentials: both username and password must be provided together
    if bool(earthdata_username) ^ bool(earthdata_password):
        raise click.UsageError(
            "You must provide both --earthdata-username and --earthdata-password together, or neither."
        )

    # 1) Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving {short_name}{version} granules to: {output_dir}")

    # 2) Authenticate with NASA Earthdata
    logger.info("Authenticating with NASA Earthdata Login via earthaccess.login()")
    _ = earthaccess.login(persist=True)

    # 3) Search for granules
    temporal = (start_date, end_date)
    logger.info(f"Searching for short_name='{short_name}', version='{version}', temporal={temporal}")
    granules = earthaccess.search_data(short_name=short_name, version=version, temporal=temporal, count=page_size)
    ng = len(granules)
    logger.info(f"Found {ng} granule(s) of {short_name} v{version}")

    if ng == 0:
        raise click.ClickException("No granules found! Check your short-name/version or temporal range.")

    # 4) Download granules
    logger.info("Beginning download of all granules...")
    earthaccess.download(granules, output_dir)
    logger.info("Download complete! 🎉")


if __name__ == "__main__":
    main()
