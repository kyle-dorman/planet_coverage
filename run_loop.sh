#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/Users/kyledorman/data/planet_coverage"
TIDE_DIR="/Users/kyledorman/data/tides"

declare -a REGIONS=("ca_only" "points_30km")
declare -a SATS=("skysat" "dove")

for region in "${REGIONS[@]}"; do
  for sat in "${SATS[@]}"; do
    echo "▶ Processing ${region}/${sat} …"

    python src/create_coastal_df.py \
      "${BASE_DIR}/${region}/${sat}/results/" \
      --query-grids-path      "${BASE_DIR}/${region}/ocean_grids.gpkg" \
      --coastal-grids-path    "${BASE_DIR}/${region}/coastal_grids.gpkg" \
      --tide-heuristics-path  "${BASE_DIR}/${region}/simulated_tidal_coverage_heuristics.csv" \
      --tide-data-dir         "${TIDE_DIR}" \
      --save-dir              "${BASE_DIR}/${region}/${sat}/coastal_results"

    echo "✔ Done ${region}/${sat}"
  done
done

