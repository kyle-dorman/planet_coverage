#!/usr/bin/env python3
"""
download_mcd12q1.py

Download every granule of the MODIS Land Cover product (MCD12Q1 v061)
using NASA’s earthaccess library.
"""

import logging
import os
import sys

import earthaccess

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
# Local directory where granules will be saved
OUTPUT_DIR = "/Users/kyledorman/data/MCD12Q1v061"

# EarthAccess search parameters
SHORT_NAME = "MCD12Q1"  # MODIS Land Cover
VERSION = "061"  # Collection 6 version 061
# Temporal range: from first MODIS Land Cover release (2001-01-01)
# through today.
TEMPORAL = ("2019-01-01", "2020-01-01")

# How many granules to request per CMR call (max is 20000)
PAGE_SIZE = 20000

# -------------------------------------------------------------------
# SET UP LOGGING
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

os.environ["EARTHDATA_USERNAME"] = "TODO"
os.environ["EARTHDATA_PASSWORD"] = "TODO"


def main(output_dir: str):
    # 1) Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving MCD12Q1v061 granules to: {output_dir}")

    # 2) Authenticate (will use ~/.netrc or prompt)
    logger.info("Authenticating with NASA Earthdata Login via earthaccess.login()")
    # Authenticate NASA Earthdata
    _ = earthaccess.login(strategy="environment", persist=True)

    # 3) Search for all granules of MCD12Q1 v061
    logger.info(f"Searching for short_name='{SHORT_NAME}', version='{VERSION}', temporal={TEMPORAL}")
    granules = earthaccess.search_data(short_name=SHORT_NAME, version=VERSION, temporal=TEMPORAL, count=PAGE_SIZE)
    ng = len(granules)
    logger.info(f"Found {ng} granule(s) of {SHORT_NAME} v{VERSION}")

    if ng == 0:
        logger.warning("No granules found! Check your short_name/version or temporal range.")
        sys.exit(1)

    # 4) Download all granules
    #    This will stream each granule to OUTPUT_DIR, showing progress.
    logger.info("Beginning download of all granules...")
    earthaccess.download(granules, output_dir)
    logger.info("Download complete! 🎉")


if __name__ == "__main__":
    main(OUTPUT_DIR)
