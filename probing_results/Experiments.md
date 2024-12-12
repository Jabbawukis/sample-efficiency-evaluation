# Fact Matching Dataset: [wikipedia_20231101_en](https://huggingface.co/datasets/wikimedia/wikipedia)

## Probing on the whole dataset:

Train a model on the whole dataset and probe it after training.

### BEAR-big
- fact matching results: [fact_matching_results](/fact_matching_results/BEAR-big/wikimedia_wikipedia_20231101_en)

#### Aliases Stats:

- % of subjects with alias: 0.4271537429432166
- % of objects with alias: 0.7703662182361734
- Overall number of instances with matches: 28144
- Number of instances with more matches achieved due to aliases: 13032
- Number of instances with matches achieved without the need for aliases: 15112
- Number of instances with no matches with aliases: 12772
- Number of instances with no matches without aliases: 15875
- % of instances with more matches having subject and object aliases: 0.5860957642725598
- % of instances with more matches having only subject aliases: 0.06269183548189074
- % of instances with more matches having only object aliases: 0.3512124002455494
- % of instances with more matches due to aliases (over all instances with matches): 0.4630471859010802
- Average increase in matches due to aliases: 187.09484346224679

##### 1. gpt2_off_the_shelve (for comparison)

- Model: gpt2
- repo: [openai-community/gpt2](https://huggingface.co/gpt2)
- link to probing results: [probing results](/probing_results/BEAR-big/gpt2_off_the_shelve/)
- trained on: a pre-trained model

#### 2. gpt2_from_scratch

- Model: gpt2
- repo: [J4bb4wukis/gpt2_wikipedia_en](https://huggingface.co/J4bb4wukis/gpt2_wikipedia_en)
- link to probing results: [probing results](/probing_results/BEAR-big/gpt2_from_scratch/)
- trained on: [wikipedia_20231101_en](https://huggingface.co/datasets/wikimedia/wikipedia)
- training script: [train.py](../model_training_setups/GPT2/wikimedia_wikipedia_20231101_en/train.py)

### BEAR(-small)
- fact matching results: [fact_matching_results](/fact_matching_results/BEAR-small/wikimedia_wikipedia_20231101_en)

#### Aliases Stats:

- % of subjects with alias: 0.5051875498802874
- % of objects with alias: 0.8207317073170731
- Overall number of instances with matches: 6775
- Number of instances with more matches achieved due to aliases: 3603
- Number of instances with matches achieved without the need for aliases: 3172
- Number of instances with no matches with aliases: 956
- Number of instances with no matches without aliases: 1463
- % of instances with more matches having subject and object aliases: 0.6319733555370525
- % of instances with more matches having only subject aliases: 0.04607271718012767
- % of instances with more matches having only object aliases: 0.32195392728281985
- % of instances with more matches due to aliases (over all instances with matches): 0.5318081180811808
- Average increase in matches due to aliases: 297.6960865945046

#### 1. gpt2_off_the_shelve (for comparison)

- Model: gpt2
- repo: [openai-community/gpt2](https://huggingface.co/gpt2)
- link to probing results: [probing results](/probing_results/BEAR-small/gpt2_off_the_shelve/)
- trained on: a pre-trained model

#### 2. gpt2_from_scratch

- Model: gpt2
- repo: [J4bb4wukis/gpt2_wikipedia_en](https://huggingface.co/J4bb4wukis/gpt2_wikipedia_en)
- link to probing results: [probing results](/probing_results/BEAR-small/gpt2_from_scratch/)
- trained on: [wikipedia_20231101_en](https://huggingface.co/datasets/wikimedia/wikipedia)
- training script: [train.py](../model_training_setups/GPT2/wikimedia_wikipedia_20231101_en/train.py)

## Probing on training data slices

Train a model on the whole dataset and probe it after a slice of the training data has been processed.

#### 1. gpt2_from_scratch
- Model: gpt2
- dataset shuffle seed: 42
- number of slices: 42