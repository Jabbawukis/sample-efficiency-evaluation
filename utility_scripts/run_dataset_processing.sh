#!/bin/bash

CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Configuration
#NUM_SLICES=4                                   # Number of slices to divide the dataset into
#DATASET_PATH=""                                # Pass as string
#DATASET_NAME=""                                # Optional
#BEAR_DATA_PATH="BEAR"
#BEAR_FACTS_PATH="BEAR-big"
#REL_INFO_OUTPUT_DIR="output"
#MATCHER_TYPE="simple"
#ENTITY_LINKER_MODEL="en_core_web_trf"
#SAVE_FILE_CONTENT="True"                       # Pass as string (Should always be True)
#REQUIRE_GPU="False"                            # Pass as string
#GPU_ID=0                                       # Pass as int
#REMOVE_SENTENCES_IN_JOINED_OUTPUT="False"      # Pass as string

# Create output directory if it doesn't exist
mkdir -p "$REL_INFO_OUTPUT_DIR"

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
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 ${CUR_DIR}/slice_processor.py \
        --dataset_path "$DATASET_PATH" \
        --dataset_name "$DATASET_NAME" \
        --bear_data_path "$BEAR_DATA_PATH" \
        --bear_facts_path "$BEAR_FACTS_PATH" \
        --rel_info_output_dir "$REL_INFO_OUTPUT_DIR" \
        --matcher_type "$MATCHER_TYPE" \
        --entity_linker_model "$ENTITY_LINKER_MODEL" \
        --gpu_id "$GPU_ID" \
        --total_slices $NUM_SLICES \
        --slice_num $((i)) \
        --save_file_content "$SAVE_FILE_CONTENT" \
        --require_gpu "$REQUIRE_GPU" &

done

# Wait for all background processes to finish
wait
echo "All slices processed."

# Merge the slices
python3 -c "from utility.utility import join_relation_info_json_files; join_relation_info_json_files('${REL_INFO_OUTPUT_DIR}', remove_sentences='${REMOVE_SENTENCES_IN_JOINED_OUTPUT}' , correct_possible_duplicates='${SAVE_FILE_CONTENT}')"

# Create Diagram
python3 -c "from utility.utility import create_fact_occurrence_histogram; create_fact_occurrence_histogram('${REL_INFO_OUTPUT_DIR}/joined_relation_info.json')"
