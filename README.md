# Sample Efficiency Evaluation

[![Build status](https://img.shields.io/github/actions/workflow/status/Jabbawukis/sample_efficiency_evaluation/test.yml?logo=github&label=Tests)](https://github.com/Jabbawukis/sample_efficiency_evaluation/actions)
[![Code style: black](https://img.shields.io/badge/Code%20style-black-000000.svg)](https://github.com/psf/black)

This project aims to measure the sample efficiency of different language model architectures using the BEAR dataset.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- `pip` package manager

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Jabbawukis/sample-efficiency-evaluation.git
    cd sample-efficiency-evaluation
    ```

2. Install the required dependencies:
    ```bash
    make install
    ```

### Get the BEAR Dataset

1. Download the BEAR dataset from the following link: [BEAR Dataset](https://github.com/lm-pub-quiz/BEAR)

2. Place the downloaded dataset in the appropriate directory as specified in your configuration (or simply inside this directory).

### Usage

To extract BEAR fact occurrences from a dataset, export the necessary environment variables and run the dataset processing script:

```bash

export NUM_SLICES=3 # Number of slices to process
export DATASET_PATH="PatrickHaller/pile-10M-words" # Path to the dataset
export DATASET_NAME="" # Name of the dataset (optional, can be left empty)
export BEAR_DATA_PATH="BEAR" # Path to the BEAR dataset
export BEAR_FACTS_PATH="BEAR/BEAR-big" # Path to the BEAR facts
export PATH_TO_ALL_ENTITIES="BEAR/all_entities.json" # Path to all entities in the BEAR dataset
export EXCLUDE_ALIASES="False" # Whether to exclude aliases in the BEAR dataset
export REL_INFO_OUTPUT_DIR="output" # Directory to save the output
export MATCHER_TYPE="simple" # Type of matcher to use (currently only 'simple' is supported)
export SAVE_FILE_CONTENT_IN_SLICE="True" # Whether to save file content in each slice
export TEXT_KEY="text" # Key for text content in the dataset

./utility_scripts/run_dataset_processing.sh
```

## Citation
```
@inproceedings{
anonymous2025from,
title={From Data to Knowledge: Evaluating How Efficiently Language Models Learn Facts},
author={Anonymous},
booktitle={First Workshop on Large Language Model Memorization},
year={2025},
url={https://openreview.net/forum?id=iXHpdSGd8o}
}
```
