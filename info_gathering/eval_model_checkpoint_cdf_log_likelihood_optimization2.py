import os
import numpy as np
from tqdm import tqdm
import logging

import matplotlib.pyplot as plt
from scipy.optimize import minimize

from utility.utility import load_json_dict

path_to_checkpoints_probing_results = "../../sample_efficiency_evaluation_results/probing_results/BEAR-big/gpt2_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices/probing_results_on_checkpoints/checkpoint_extracted"
path_to_increasing_occurrences_in_slices = "../../sample_efficiency_evaluation_results/probing_results/BEAR-big/gpt2_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices/increasing_occurrences_in_slices.json"


def get_num(x: str) -> int:
    """Extract numerical suffix from a string."""
    number = x.split("-")[-1]
    return int(number)


# Main processing
checkpoints = os.listdir(path_to_checkpoints_probing_results)
sorted_checkpoints = sorted(checkpoints, key=get_num)
increasing_occurrences = load_json_dict(path_to_increasing_occurrences_in_slices)
data_on_slices = {}
max_occurrence = 0
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
            max_occurrence = max(occurrences, max_occurrence)
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


def cumulative_distribution_function(lambd, x, x_max):
    if x > x_max:
        x = x_max  # Cap occurrences at x_max
    return 1 - np.exp(-lambd * (x / x_max))


# Vectorize the CDF to handle arrays
vectorized_cdf = np.vectorize(cumulative_distribution_function, excluded=["lambd", "x_max"])


# Define the log-likelihood function
def compute_log_likelihood(t, p_i):
    """
    Compute log-likelihood score for a given set of outcomes and probabilities.
    t: array-like, binary outcomes (1 for correct, 0 for incorrect)
    p_i: array-like, predicted probabilities
    """
    return t * np.log(p_i) + (1 - t) * np.log(1 - p_i)


# Define the negative log-likelihood loss
def negative_log_likelihood(params, _occurrences, _outcomes):
    """
    Compute the negative log-likelihood for a given lambda.

    params: lambd value and x_max
    occurrences: array-like, number of occurrences of each fact
    outcomes: array-like, binary outcomes (1 for correct, 0 for incorrect)
    """
    lambd, x_max = params
    p_i = vectorized_cdf(lambd, _occurrences, x_max)
    # Ensure probabilities are within a valid range to avoid log(0)
    p_i = np.clip(p_i, 1e-10, 1 - 1e-10)
    log_likelihood = compute_log_likelihood(_outcomes, p_i)
    return -np.sum(log_likelihood)


# Initial guess for lambda
initial_params = np.array([0.1, max_occurrence])  # Initial lambda and x_max
bounds = [(1e-5, None), (1, max_occurrence * 2)]  # Bounds for lambda and x_max
optimized_lambdas = []
for slice_id, slice_data in data_on_slices.items():
    occurrences = np.array(slice_data["occurrences"])
    outcomes = np.array(slice_data["answers"])

    # Minimize the negative log-likelihood
    result = minimize(
        negative_log_likelihood,
        x0=initial_params,
        args=(occurrences, outcomes),
        bounds=bounds,  # Lambda must be positive
        method="L-BFGS-B",
    )

    # Optimized lambda
    optimized_lambda, optimized_x_max = result.x
    print(f"Slice {slice_id}: Optimized lambda: {optimized_lambda}")
    print(f"Slice {slice_id}: Optimized x_max: {optimized_x_max}")

    # Check the result
    if result.success:
        print(f"Slice {slice_id}: Optimization was successful. Optimized lambda: {optimized_lambda}")
        optimized_lambdas.append({"slice": slice_id, "lambda": optimized_lambda, "x_max": optimized_x_max})
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
        probabilities = [
            cumulative_distribution_function(lambd["lambda"], x, lambd["x_max"]) for x in occurrences_range
        ]
        plt.plot(
            occurrences_range,
            probabilities,
            label=f"Lambda = {lambd['lambda']:.4f}; " f"Slice {lambd['slice']}; " f"x_max = {lambd['x_max']:.4f}",
        )

    plt.title("CDF for Optimized Lambdas")
    plt.xlabel("Occurrences")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid()
    plt.show()


# Plot the log-likelihood scores over the checkpoints
max_occurrence = 1024
plot_cdf_for_lambda(optimized_lambdas, np.linspace(0, max_occurrence, 100))
