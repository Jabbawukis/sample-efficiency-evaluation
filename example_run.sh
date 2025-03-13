#!/bin/bash

export NUM_SLICES=3                                   # Number of slices to divide the dataset into
export DATASET_PATH="PatrickHaller/pile-10M-words"                                # Pass as string
export DATASET_NAME=""                                # Optional
export BEAR_DATA_PATH="BEAR"
export BEAR_FACTS_PATH="BEAR/BEAR-big"
export PATH_TO_ALL_ENTITIES="BEAR/all_entities.json"                     # Optional (Pass as string)
export EXCLUDE_ALIASES="False"                         # Optional (Pass as string)
export REL_INFO_OUTPUT_DIR="output"
export MATCHER_TYPE="simple"
export SAVE_FILE_CONTENT_IN_SLICE="True"                       # Pass as string

./utility_scripts/run_dataset_processing.sh