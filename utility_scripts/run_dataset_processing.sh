#!/bin/bash

CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Create output directory if it doesn't exist
mkdir -p "$REL_INFO_OUTPUT_DIR"

mkdir -p "$REL_INFO_OUTPUT_DIR"/slice_infos

cleanup() {
    echo "Terminating all background processes..."
    pkill -P $$  # Kill all child processes of the current script
    exit 1
}

# Set up trap to catch SIGINT (Ctrl+C) and call the cleanup function
trap cleanup SIGINT

# Download the dataset
python3 -c "import datasets; datasets.load_dataset('${DATASET_PATH}', '${DATASET_NAME}', split='train')"

# Loop through each slice and assign a Python processing job
for (( i=0; i<NUM_SLICES; i++ )); do

    # Run the slice processor in the background with the specified GPU
    python3 ${CUR_DIR}/slice_processor.py \
        --dataset_path "$DATASET_PATH" \
        --dataset_name "$DATASET_NAME" \
        --bear_data_path "$BEAR_DATA_PATH" \
        --bear_facts_path "$BEAR_FACTS_PATH" \
        --path_to_all_entities "$PATH_TO_ALL_ENTITIES" \
        --exclude_aliases "$EXCLUDE_ALIASES" \
        --rel_info_output_dir "$REL_INFO_OUTPUT_DIR" \
        --matcher_type "$MATCHER_TYPE" \
        --total_slices $NUM_SLICES \
        --slice_num $((i)) \
        --text_key "$TEXT_KEY" \
        --save_file_content "$SAVE_FILE_CONTENT_IN_SLICE" &

done

# Wait for all background processes to finish
wait
echo "All slices processed."

# Merge the slices
python3 -c "from utility.utility import join_relation_occurrences_info_json_files; join_relation_occurrences_info_json_files('${REL_INFO_OUTPUT_DIR}')"

# Create Diagram
python3 -c "from utility.utility import create_fact_occurrence_histogram; create_fact_occurrence_histogram('${REL_INFO_OUTPUT_DIR}/joined_relation_occurrence_info.json')"
