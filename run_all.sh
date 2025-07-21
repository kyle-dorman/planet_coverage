#!/bin/bash

set -euo pipefail

BASE="/Users/kyledorman/data/planet_coverage/count_missing"

# Loop from 2013 to 2024
for YEAR in {2013..2024}
do
    # Loop over dove and skysat
    for SAT in {"skysat","dove"}
    do
        BASESAT="${BASE}/${SAT}"
        YAML_FILE="${BASESAT}/config.yaml"

        echo "Processing satellite ${SAT} and year ${YEAR}"

        # Update the save_dir line inside the YAML file
        sed -i '' "s|save_dir: \".*\"|save_dir: \"${BASESAT}/results/${YEAR}\"|" "$YAML_FILE"

        # Define start and end dates
        START_DATE="${YEAR}-12-01"
        END_DATE="$((YEAR + 1))-12-01"

        # Run the Python script
        python src/count_missing.py -c "$YAML_FILE" -s "$START_DATE" -e "$END_DATE"
        
    done
done
