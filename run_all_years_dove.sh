#!/bin/bash

BASE="/Users/kyledorman/data/planet_coverage/dove"
YAML_FILE="${BASE}/config.yaml"

# Loop from 2022 down to 2013
for YEAR in {2022..2013}
do
    echo "Processing year starting from December $YEAR"

    # Update the save_dir line inside the YAML file
    sed -i '' "s|save_dir: \".*\"|save_dir: \"${BASE}/results/${YEAR}\"|" "$YAML_FILE"

    # Define start and end dates
    START_DATE="${YEAR}-12-01"
    END_DATE="$((YEAR + 1))-12-01"

    # Run the Python script
    python src/query_udms.py -c "$YAML_FILE" -s "$START_DATE" -e "$END_DATE"
done
