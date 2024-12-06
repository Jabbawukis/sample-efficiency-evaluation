# Model Experiments

## Fact Matching Dataset: [wikipedia_20231101_en](https://huggingface.co/datasets/wikimedia/wikipedia)

### BEAR-big
- fact matching results: [fact_matching_results](/fact_matching_results/BEAR-big/wikimedia_wikipedia_20231101_en)

#### 1. gpt2_off_the_shelve

- Model: gpt2
- repo: [openai-community/gpt2](https://huggingface.co/gpt2)
- link to probing results: [probing results](/probing_results/BEAR-big/gpt2_off_the_shelve/)
- trained on: pre-trained model

#### 2. gpt2_from_scratch

- Model: gpt2
- repo: [J4bb4wukis/gpt2_wikipedia_en](https://huggingface.co/J4bb4wukis/gpt2_wikipedia_en)
- link to probing results: [probing results](/probing_results/BEAR-big/gpt2_from_scratch/)
- trained on: [wikipedia_20231101_en](https://huggingface.co/datasets/wikimedia/wikipedia)
- training script: [train.py](../model_training_setups/GPT2/train.py)

### BEAR(-small)
- fact matching results: [fact_matching_results](/fact_matching_results/BEAR-small/wikimedia_wikipedia_20231101_en)

#### 1. gpt2_off_the_shelve

- Model: gpt2
- repo: [openai-community/gpt2](https://huggingface.co/gpt2)
- link to probing results: [probing results](/probing_results/BEAR-small/gpt2_off_the_shelve/)
- trained on: pre-trained model

#### 2. gpt2_from_scratch

- Model: gpt2
- repo: [J4bb4wukis/gpt2_wikipedia_en](https://huggingface.co/J4bb4wukis/gpt2_wikipedia_en)
- link to probing results: [probing results](/probing_results/BEAR-small/gpt2_from_scratch/)
- trained on: [wikipedia_20231101_en](https://huggingface.co/datasets/wikimedia/wikipedia)
- training script: [train.py](../model_training_setups/GPT2/train.py)