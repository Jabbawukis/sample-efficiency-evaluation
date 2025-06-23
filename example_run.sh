#!/bin/bash

export NUM_SLICES=3
export DATASET_PATH="PatrickHaller/pile-10M-words"
export DATASET_NAME=""
export BEAR_DATA_PATH="BEAR"
export BEAR_FACTS_PATH="BEAR/BEAR-big"
export PATH_TO_ALL_ENTITIES="BEAR/all_entities.json"
export EXCLUDE_ALIASES="False"
export REL_INFO_OUTPUT_DIR="output"
export MATCHER_TYPE="simple"
export SAVE_FILE_CONTENT_IN_SLICE="True"
export TEXT_KEY="text"

./utility_scripts/run_dataset_processing.sh