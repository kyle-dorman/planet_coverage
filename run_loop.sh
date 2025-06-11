#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/Users/kyledorman/data/planet_coverage"
TIDE_DIR="/Users/kyledorman/data/tides"

python src/select_coastal_grids.py \
  -c /Users/kyledorman/data/planet_coverage/points_30km/coastal_strips.gpkg \
  -o /Users/kyledorman/data/planet_coverage/points_30km/coastal_grids.gpkg \
  --mainlands /Users/kyledorman/data/planet_coverage/shorelines/mainlands.gpkg \
  --big-islands /Users/kyledorman/data/planet_coverage/shorelines/big_islands.gpkg \
  --small-islands /Users/kyledorman/data/planet_coverage/shorelines/small_islands.gpkg  \
  --antarctica /Users/kyledorman/data/planet_coverage/shorelines/antarctica.geojson;

cp /Users/kyledorman/data/planet_coverage/points_30km/coastal_grids.gpkg \
  /Users/kyledorman/data/planet_coverage/ca_only/;

python src/simulated_tidal.py \
  --base-dir /Users/kyledorman/data/planet_coverage/points_30km/ \
  --tide-data-dir /Users/kyledorman/data/tides/ \
  --out-path /Users/kyledorman/data/planet_coverage/points_30km/simulated_tidal_coverage.csv;

cp /Users/kyledorman/data/planet_coverage/points_30km/simulated_tidal* \
  /Users/kyledorman/data/planet_coverage/ca_only/;

declare -a REGIONS=("ca_only" "points_30km")
declare -a SATS=("skysat" "dove")

for region in "${REGIONS[@]}"; do
  for sat in "${SATS[@]}"; do
    echo "▶ Processing ${region}/${sat} …"

    rm -rf "${BASE_DIR}/${region}/${sat}/coastal_results";

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

