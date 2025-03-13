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

3. Download the BEAR dataset from the following link: [BEAR Dataset](https://github.com/lm-pub-quiz/BEAR)

4. Place the downloaded dataset in the appropriate directory as specified in your configuration (or simply inside this directory).

## Experiment Results
Clone the repository containing the experiment results and place it next to the `sample-efficiency-evaluation` directory.:
```bash
git clone https://github.com/Jabbawukis/sample-efficiency-evaluation-results
```
Extract the contents describe in the `sample-efficiency-evaluation-results` README.md file.