#!/bin/bash
# remove_date_lines.sh
# Recursively remove lines containing "date" from all mlperf_log_detail.txt files

BASE_DIR=${1:-.}

echo "Removing lines containing 'date' from mlperf_log_detail.txt files under $BASE_DIR"

# Find all matching files
find "$BASE_DIR" -type f -name "mlperf_log_detail.txt" | while read -r file; do
    echo "Processing: $file"
    # Use sed to delete lines containing "date" in-place
    sed -i '/date/d' "$file"
done

echo "Done."

