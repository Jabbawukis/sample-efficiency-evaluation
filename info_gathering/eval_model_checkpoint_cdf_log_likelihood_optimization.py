import os
import numpy as np
from tqdm import tqdm
import logging

import matplotlib.pyplot as plt
from scipy.optimize import minimize

import utility.utility
from utility.utility import load_json_dict

def get_num(x: str) -> int:
    """Extract numerical suffix from a string."""
    number = x.split("-")[-1]
    return int(number)


def get_slice_data(path_probing_results, path_increasing_occurrences_in_slices):
    checkpoints = os.listdir(path_probing_results)
    sorted_checkpoints = sorted(checkpoints, key=get_num)
    increasing_occurrences = load_json_dict(path_increasing_occurrences_in_slices)
    data_on_slices = {}
    abs_min_acc = 1.0
    for idx, checkpoint in enumerate(tqdm(sorted_checkpoints, desc="Get results in slices")):
        # Load checkpoint metadata
        metadata = load_json_dict(f"{path_probing_results}/{checkpoint}/metadata_results.json")
        occurrences_list = []
        answer_list = []
        answer_space = []

        for relation_id, entity_dict in increasing_occurrences.items():
            # Get number of possible answers for this relation
            num_possible_answers = len(metadata[relation_id]["answer_space_labels"])
            abs_min_acc = min(abs_min_acc, 1/num_possible_answers)
            for entity_id, occurrences_increase in entity_dict.items():
                slice_info = occurrences_increase["occurrences_increase"][idx]

                # Ensure slice and checkpoint match expectations
                assert slice_info["Slice"] == idx
                assert slice_info["checkpoint"] == checkpoint

                # Extract occurrence and correctness
                T = 1 if slice_info["correct"] else 0

                occurrences_list.append(slice_info["total"])
                answer_list.append(T)
                answer_space.append(1/num_possible_answers)

        # Sum scores for the current slice
        data_on_slices[f"{idx}"] = {"occurrences": occurrences_list, "answers": answer_list, "answer_space": answer_space}

    initial_slice_len = len(data_on_slices["0"]["occurrences"])
    for slice_id in data_on_slices.keys():
        # Ensure the number of occurrences and answers
        assert len(data_on_slices[slice_id]["occurrences"]) == len(data_on_slices[slice_id]["answers"])
        assert len(data_on_slices[slice_id]["occurrences"]) == initial_slice_len

    return data_on_slices, abs_min_acc


def cumulative_distribution_function(lambd, x, max_ac, min_ac):
    prob = 1 - np.exp(-lambd * x)
    return min_ac + (max_ac - min_ac) * prob


# Vectorize the CDF to handle arrays
vectorized_cdf = np.vectorize(cumulative_distribution_function, excluded=["lambd", "max_ac"])


# Define the log-likelihood function
def compute_log_likelihood(t, p_i):
    """
    Compute log-likelihood score for a given set of outcomes and probabilities.
    t: array-like, binary outcomes (1 for correct, 0 for incorrect)
    p_i: array-like, predicted probabilities
    """
    return t * np.log(p_i) + (1 - t) * np.log(1 - p_i)


# Define the negative log-likelihood loss
def negative_log_likelihood(params, _occurrences, _outcomes, _min_acc):
    """
    Compute the negative log-likelihood for a given lambda.

    params: lambd value and x_max
    occurrences: array-like, number of occurrences of each fact
    outcomes: array-like, binary outcomes (1 for correct, 0 for incorrect)
    """
    lambd, max_ac = params
    p_i = vectorized_cdf(lambd, _occurrences, max_ac, _min_acc)
    # Ensure probabilities are within a valid range to avoid log(0)
    p_i = np.clip(p_i, 1e-10, 1 - 1e-10)
    log_likelihood = compute_log_likelihood(_outcomes, p_i)
    return -np.sum(log_likelihood)

def optimize_lambdas(data_slice_info):
    # Initial guess for lambda
    initial_params = np.array([0.1, 1.0])
    bounds = [(1e-5, None), (0.0, 1.0)]
    optimized_lambdas = []
    for slice_id, slice_data in data_slice_info.items():
        occurrences = np.array(slice_data["occurrences"])
        outcomes = np.array(slice_data["answers"])
        min_acc = np.array(slice_data["answer_space"])

        # Minimize the negative log-likelihood
        result = minimize(
            negative_log_likelihood,
            x0=initial_params,
            args=(occurrences, outcomes, min_acc),
            bounds=bounds,  # Lambda must be positive
            method="L-BFGS-B",
        )

        # Optimized lambda
        optimized_lambda, optimized_max_acc = result.x
        print(f"Slice {slice_id}: Optimized lambda: {optimized_lambda}")
        print(f"Slice {slice_id}: Optimized optimized_max_acc: {optimized_max_acc}")

        # Check the result
        if result.success:
            print(f"Slice {slice_id}: Optimization was successful. Optimized lambda: {optimized_lambda}")
            optimized_lambdas.append({"slice": slice_id, "lambda": optimized_lambda, "optimized_max_acc": optimized_max_acc})
        else:
            logging.warning(f"Slice {slice_id}: Optimization failed:", result.message)
    return optimized_lambdas


def plot_cdf_for_lambda(lambdas, occurrences_range, abs_min_acc):
    """
    Plot the cumulative distribution function for all lambdas in a single plot.

    lambdas: list of optimized lambda values
    occurrences_range: range of occurrences to plot (array-like)
    """
    plt.figure(figsize=(24, 18))
    for lambd in lambdas:
        probabilities = [
            cumulative_distribution_function(lambd["lambda"], x, lambd["optimized_max_acc"], abs_min_acc) for x in occurrences_range
        ]
        plt.plot(
            occurrences_range,
            probabilities,
            label=f"Lambda = {lambd['lambda']:.4f}; " f"Slice {lambd['slice']}; " f"optimized_max_acc = {lambd['optimized_max_acc']:.4f}")

    plt.title("CDF for Optimized Lambdas")
    plt.xlabel("Occurrences")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid()
    plt.show()


def plot_lambdas(lambdas_of_models: list):
    """
    Plot the lambda values over the slices and annotate the average lambda lines with model names and averages.
    """
    plt.figure(figsize=(24, 18))

    for _model_lambdas in lambdas_of_models:
        # Extract slices and lambda values
        x = [lambd["slice"] for lambd in _model_lambdas["Lambdas"]]
        y = [lambd["lambda"] for lambd in _model_lambdas["Lambdas"]]

        # Plot lambda values for the model
        plt.plot(x, y, marker="o", linestyle="-", label=f"{_model_lambdas['Model']}")

        # Calculate and plot the average line
        avg_lambda = float(np.mean(y))
        plt.axhline(y=avg_lambda, color='r', linestyle='--', alpha=0.7)

        # Annotate the average line with model name and value
        plt.text(
            x[-1],  # Place the text near the last x-value
            avg_lambda,
            f"{_model_lambdas['Model']}; Avg. Lambda: {avg_lambda:.4f}",
            color='red',
            fontsize=12,
            ha='right',
            va='bottom',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', boxstyle='round,pad=0.3')
        )

    # Add titles, labels, and legend
    plt.title("Optimized Lambda Values", fontsize=16)
    plt.xlabel("Slice", fontsize=14)
    plt.ylabel("Lambda", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.show()


# Plot the log-likelihood scores over the checkpoints
# max_occurrence = 1024
# plot_cdf_for_lambda(optimized_lambdas, np.linspace(0, max_occurrence, 100))

optimized_lambdas = []
models = ["gpt2", "mamba2", "xlstm"]

for model in models:
    path_to_checkpoints_probing_results = f"../../sample_efficiency_evaluation_results/probing_results/BEAR-big/{model}_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices/probing_results_on_checkpoints/checkpoint_extracted"
    path_to_increasing_occurrences_in_slices = f"../../sample_efficiency_evaluation_results/probing_results/BEAR-big/{model}_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices/increasing_occurrences_in_slices.json"

    slice_data, abs_min_accuracy = get_slice_data(path_to_checkpoints_probing_results, path_to_increasing_occurrences_in_slices)

    optimized_lambdas.append({"Model": model, "Lambdas": optimize_lambdas(slice_data)})

for model in optimized_lambdas:
    utility.utility.save_dict_as_json(model,
                                      f"../../sample_efficiency_evaluation_results/probing_results/BEAR-big/{model['Model']}_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices/cdf_optimized_lambdas.json")
plot_lambdas(optimized_lambdas)
