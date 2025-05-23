# Performance Analysis on the BEAR-probe

## 1. Evaluation of the model answers for each checkpoint

[Evaluation of checkpoints](model_accuracy/get_model_checkpoint_answer_for_occurrences_in_slices_data.py):

Counts the [increasing occurrences](https://github.com/Jabbawukis/sample-efficiency-evaluation-results/tree/main/fact_matching_results/BEAR-big/wikimedia_wikipedia_20231101_en/evaluation_on_slices) of BEAR facts in the training data up until the slice.
The model checkpoints answers are evaluated for each fact.

## 2. Get the model checkpoint accuracies over the occurrence buckets (requires step 1. of model performance analysis)

[Get accuracies over occurrence buckets](model_accuracy/eval_model_checkpoint_occurrence_bucket_accuracy_on_slices.py):

For each checkpoint probing result, the accuracy is calculated for each occurrence bucket.
The accuracy results for each checkpoint are plotted and saved.

## 3. Get the model checkpoint weighted accuracies (requires step 1. of model performance analysis)

[Get weighted accuracies](model_accuracy/eval_model_checkpoint_weighted_accuracy_on_slices.py):

For each checkpoint probing result, the bucket accuracy (dependent on fact occurrence) is calculated.
The weighted accuracy is calculated by weighting the bucket accuracies by the lower bound of the bucket
The weighted accuracy results for each checkpoint, and for each model are plotted and saved.

It is also possible to calculate the weighted accuracy for each fact.
The result is an accuracy score by taking the 
sum of weights (0 if answer is wrong, the weight value if answer is correct)
and dividing it by the total sum of weights.

## 4. Get the model checkpoint accuracies over all facts (requires step 1. of model performance analysis)

[Get accuracies over all facts](model_accuracy/eval_model_checkpoint_accuracy_on_slices.py):

For each checkpoint probing result, the accuracy is calculated for all facts.

# Correct Answer Probability Analysis

## 1. Probability Function Optimization (requires step 1. of model performance analysis)

Probability [functions](correct_answer_probability_analysis/probability_function_optimization)
that approximate the probability of the model checkpoint to give the correct answer given 
the number of occurrences of the fact in the training data, are optimized.

See [here](https://github.com/Jabbawukis/sample-efficiency-evaluation-results/blob/main/probing_on_dataset_slices.md)
for the probability function definitions.
The optimized parameters of the probability functions are saved as .json files.
The optimization can be performed for each model checkpoint or for all model checkpoints at once.

### 1.1. Probability Function Optimization for multiple Parameters

The PSF_EXT2 probability function can be optimized/evaluated for multiple parameters ($L_0$, $x_0$ and $\alpha$).
Here, we can optimize these values as well by concatenating all models predictions
and minimizing the negative log-likelihood
by optimizing a separate $\alpha$ for each model and a global $x_0$ and $L_0$ value all at once.
Hence, the optimized $x_0$ and $L_0$ values are used for the PSF_EXT2 function and are dataset-specific parameters.

## 2. Evaluate the probability functions (requires step 1. of answer probability analysis)

The probability functions are [evaluated](correct_answer_probability_analysis/eval_probability_functions_nll.py) for each model checkpoint and the results are saved as .png files.
Here, the negative log likelihood loss is calculated for each model checkpoint and for each probability function.
The results are saved as .png files and .json files.