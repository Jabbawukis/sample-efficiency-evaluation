#!/bin/bash

# Configuration
#NUM_SLICES=4                                   # Number of slices to divide the dataset into
#DATASET_PATH=""                                # Pass as string
#DATASET_NAME=""                                # Optional
#BEAR_DATA_PATH="BEAR"
#REL_INFO_OUTPUT_DIR="output"
#MATCHER_TYPE="simple"
#ENTITY_LINKER_MODEL="en_core_web_trf"
#SAVE_FILE_CONTENT="True"                       # Pass as string
#READ_EXISTING_INDEX="False"                    # Pass as string
#REQUIRE_GPU="False"                            # Pass as string
#GPU_ID=0                                       # Pass as int

# Create output directory if it doesn't exist
mkdir -p "$REL_INFO_OUTPUT_DIR"

# Download the dataset
python3 -c "import datasets; datasets.load_dataset('${DATASET_PATH}', '${DATASET_NAME}', split='train')"

# Loop through each slice and assign a Python processing job
for (( i=0; i<NUM_SLICES; i++ )); do
    FILE_INDEX_DIR=".index_dir_$((i + 1))"   # Unique index directory per slice

    # Run the slice processor in the background with the specified GPU
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 slice_processor.py \
        --dataset_path "$DATASET_PATH" \
        --dataset_name "$DATASET_NAME" \
        --bear_data_path "$BEAR_DATA_PATH" \
        --rel_info_output_dir "$REL_INFO_OUTPUT_DIR" \
        --matcher_type "$MATCHER_TYPE" \
        --entity_linker_model "$ENTITY_LINKER_MODEL" \
        --gpu_id "$GPU_ID" \
        --total_slices $NUM_SLICES \
        --slice_num $((i)) \
        --file_index_dir "$FILE_INDEX_DIR" \
        --save_file_content "$SAVE_FILE_CONTENT" \
        --read_existing_index "$READ_EXISTING_INDEX" \
        --require_gpu "$REQUIRE_GPU" &

done

# Wait for all background processes to finish
wait
echo "All slices processed."
