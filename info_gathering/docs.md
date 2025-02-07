# Scripts for Evaluation

## 1. Evaluation of the model answers for each checkpoint

[Evaluation of checkpoints](model_performance_analysis/get_model_checkpoint_answer_for_occurrences_in_slices_data.py):

Counts the [increasing occurrences](https://github.com/Jabbawukis/sample-efficiency-evaluation-results/tree/main/fact_matching_results/BEAR-big/wikimedia_wikipedia_20231101_en/evaluation_on_slices) of BEAR facts in the training data up until the slice.
The model checkpoints answers are evaluated for each fact.

## 2. Get the model checkpoint accuracies over the occurrence buckets (requires step 1.)

[Get accuracies over occurrence buckets](model_performance_analysis/eval_model_checkpoint_accuracy_on_slices.py):

For each checkpoint, the model is probed and the accuracy is calculated for each occurrence bucket. The results are plotted
and saved as a .png file.