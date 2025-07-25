# Performance Analysis on the BEAR-probe

## 1. Evaluation of the model answers for each checkpoint

[Evaluation of checkpoints](model_accuracy/accuracy_evaluation.py):

This script evaluates the performance of model checkpoints on the BEAR-probe dataset. It performs the following tasks:

- **Get Model Checkpoint Answers**: Counts the increasing occurrences of BEAR facts in the training data up until the slice and evaluates the model checkpoints answers for each fact.
- **Get Model Checkpoint Accuracies**: Calculates the accuracy of each checkpoint over all facts.
- **Get Model Checkpoint Accuracies over Occurrence Buckets**: Calculates the accuracy for each occurrence bucket.
- **Get Model Checkpoint Weighted Accuracies**: Calculates a weighted accuracy score based on fact occurrences.

The results are plotted and saved as diagrams and JSON files.

# Correct Answer Probability Analysis

## 1. Probability Function Optimization (requires step 1. of model performance analysis)

Probability [functions](correct_answer_probability_analysis/)
that approximate the probability of the model checkpoint to give the correct answer given 
the number of occurrences of the fact in the training data, are optimized.

See [here](https://github.com/Jabbawukis/sample-efficiency-evaluation-results/blob/main/probing_on_dataset_slices.md)
for the probability function definitions.
The optimized parameters of the probability functions are saved as .json files.
The optimization can be performed for each model checkpoint or for all model checkpoints at once.

### 1.1. Probability Function Optimization for multiple Parameters

The PSF can be optimized/evaluated for multiple parameters ($L_0$, $x_0$ and $\alpha$).
Here, we can optimize these values as well by concatenating all models predictions
and minimizing the negative log-likelihood
by optimizing a separate $\alpha$ for each model and a global $x_0$ and $L_0$ value all at once.
Hence, the optimized $x_0$ and $L_0$ values are used for the PSF_EXT2 function and are dataset-specific parameters.

## 2. Evaluate the probability functions (requires step 1. of answer probability analysis)

The probability functions are [evaluated](correct_answer_probability_analysis/eval_probability_functions_nll.py) for each model checkpoint and the results are saved as .png files.
Here, the negative log likelihood loss is calculated for each model checkpoint and for each probability function.
The results are saved as .png files and .json files.