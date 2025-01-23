import os
import numpy as np
from tqdm import tqdm
import logging

import matplotlib.pyplot as plt
from scipy.optimize import minimize

from utility.utility import load_json_dict

path_to_checkpoints_probing_results = "../../sample_efficiency_evaluation_results/probing_results/BEAR-big/xlstm_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices/probing_results_on_checkpoints/checkpoint_extracted"
path_to_increasing_occurrences_in_slices = "../../sample_efficiency_evaluation_results/probing_results/BEAR-big/xlstm_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices/increasing_occurrences_in_slices.json"


def get_num(x: str) -> int:
    """Extract numerical suffix from a string."""
    number = x.split("-")[-1]
    return int(number)


# Main processing
checkpoints = os.listdir(path_to_checkpoints_probing_results)
sorted_checkpoints = sorted(checkpoints, key=get_num)
increasing_occurrences = load_json_dict(path_to_increasing_occurrences_in_slices)
data_on_slices = {}

for idx, checkpoint in enumerate(tqdm(sorted_checkpoints, desc="Get results in slices")):
    # Load checkpoint metadata
    metadata = load_json_dict(f"{path_to_checkpoints_probing_results}/{checkpoint}/metadata_results.json")
    occurrences_list = []
    answer_list = []

    for relation_id, entity_dict in increasing_occurrences.items():
        # Get number of possible answers for this relation
        num_possible_answers = len(metadata[relation_id]["answer_space_labels"])
        for entity_id, occurrences_increase in entity_dict.items():
            slice_info = occurrences_increase["occurrences_increase"][idx]

            # Ensure slice and checkpoint match expectations
            assert slice_info["Slice"] == idx
            assert slice_info["checkpoint"] == checkpoint

            # Extract occurrence and correctness
            occurrences = slice_info["total"]
            T = 1 if slice_info["correct"] else 0

            occurrences_list.append(occurrences)
            answer_list.append(T)

    # Sum scores for the current slice
    data_on_slices[f"{idx}"] = {"occurrences": occurrences_list, "answers": answer_list}

initial_slice_len = len(data_on_slices["0"]["occurrences"])
for slice_id in data_on_slices.keys():
    # Ensure the number of occurrences and answers
    assert len(data_on_slices[slice_id]["occurrences"]) == len(data_on_slices[slice_id]["answers"])
    assert len(data_on_slices[slice_id]["occurrences"]) == initial_slice_len


def cumulative_distribution_function(lambd, x):
    return 1 - np.exp(-lambd * x) if x >= 0 else 0


# Vectorize the CDF to handle arrays
vectorized_cdf = np.vectorize(cumulative_distribution_function, excluded=["lambd"])


# Define the log-likelihood function
def compute_log_likelihood(t, p_i):
    """
    Compute log-likelihood score for a given set of outcomes and probabilities.
    t: array-like, binary outcomes (1 for correct, 0 for incorrect)
    p_i: array-like, predicted probabilities
    """
    return t * np.log(p_i) + (1 - t) * np.log(1 - p_i)


# Define the negative log-likelihood loss
def negative_log_likelihood(lambd, _occurrences, _outcomes):
    """
    Compute the negative log-likelihood for a given lambda.

    lambd: float, the parameter to optimize
    occurrences: array-like, number of occurrences of each fact
    outcomes: array-like, binary outcomes (1 for correct, 0 for incorrect)
    """
    p_i = vectorized_cdf(lambd, _occurrences)
    # Ensure probabilities are within a valid range to avoid log(0)
    p_i = np.clip(p_i, 1e-10, 1 - 1e-10)
    log_likelihood = compute_log_likelihood(_outcomes, p_i)
    return -np.sum(log_likelihood)


# Initial guess for lambda
initial_lambda = np.array(0.1)
optimized_lambdas = []
for slice_id, slice_data in data_on_slices.items():
    occurrences = np.array(slice_data["occurrences"])
    outcomes = np.array(slice_data["answers"])

    # Minimize the negative log-likelihood
    result = minimize(
        negative_log_likelihood,
        x0=initial_lambda,
        args=(occurrences, outcomes),
        bounds=[(1e-5, None)],  # Lambda must be positive
        method="L-BFGS-B",
    )

    # Optimized lambda
    optimized_lambda = result.x[0]
    print(f"Slice {slice_id}: Optimized lambda: {optimized_lambda}")

    # Check the result
    if result.success:
        print(f"Slice {slice_id}: Optimization was successful. Optimized lambda: {optimized_lambda}")
        optimized_lambdas.append({"slice": slice_id, "lambda": optimized_lambda})
    else:
        logging.warning(f"Slice {slice_id}: Optimization failed:", result.message)


def plot_cdf_for_lambda(lambdas, occurrences_range):
    """
    Plot the cumulative distribution function for all lambdas in a single plot.

    lambdas: list of optimized lambda values
    occurrences_range: range of occurrences to plot (array-like)
    """
    plt.figure(figsize=(24, 18))
    for lambd in lambdas:
        probabilities = [cumulative_distribution_function(lambd["lambda"], x) for x in occurrences_range]
        plt.plot(occurrences_range, probabilities, label=f"Lambda = {lambd['lambda']:.4f} Slice {lambd['slice']}")

    plt.title("CDF for Optimized Lambdas")
    plt.xlabel("Occurrences")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid()
    plt.show()


# Plot the log-likelihood scores over the checkpoints
max_occurrence = 1024
plot_cdf_for_lambda(optimized_lambdas, np.linspace(0, max_occurrence, 100))
