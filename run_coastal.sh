#!/bin/bash

set -euo pipefail

DATA="/Users/kyledorman/data/"
BASE="$DATA/planet_coverage/points_30km"
SHORELINE="$DATA/planet_coverage/shorelines"

# python src/simulated_tidal.py \
#     --base-dir $SHORELINE \
#     --tide-data-dir $DATA/tides \
#     --out-path $BASE/simulated_tidal_coverage.csv

for SAT in {"skysat","dove"}
do
    BASESAT="${BASE}/${SAT}/results"

    echo "Processing satellite ${SAT}"

    # Run the Python script
    python src/create_coastal_df.py $BASESAT \
        --query-grids-path $SHORELINE/ocean_grids.gpkg \
        --coastal-grids-path $SHORELINE/coastal_grids.gpkg \
        --tide-heuristics-path $BASE/simulated_tidal_coverage_heuristics.csv \
        --tide-data-dir $DATA/tides \
        --save-dir ${BASE}/${SAT}/coastal_results \
        ;
    
done
