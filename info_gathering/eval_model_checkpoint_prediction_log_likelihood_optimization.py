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
    for idx, checkpoint in enumerate(tqdm(sorted_checkpoints, desc="Get results in slices")):
        # Load checkpoint metadata
        metadata = load_json_dict(f"{path_probing_results}/{checkpoint}/metadata_results.json")
        occurrences_list = []
        answer_list = []
        answer_space = []

        for relation_id, entity_dict in increasing_occurrences.items():
            # Get number of possible answers for this relation
            num_possible_answers = len(metadata[relation_id]["answer_space_labels"])

            for entity_id, occurrences_increase in entity_dict.items():
                slice_info = occurrences_increase["occurrences_increase"][idx]

                # Ensure slice and checkpoint match expectations
                assert slice_info["Slice"] == idx
                assert slice_info["checkpoint"] == checkpoint

                # Extract occurrence and correctness
                T = 1 if slice_info["correct"] else 0

                # Skip slices with no occurrences
                if slice_info["total"] == 0:
                    continue

                occurrences_list.append(slice_info["total"])
                answer_list.append(T)
                answer_space.append(1 / num_possible_answers)

        # Sum scores for the current slice
        data_on_slices[f"{idx}"] = {
            "occurrences": occurrences_list,
            "answers": answer_list,
            "answer_space": answer_space,
        }
    return data_on_slices


def scaling_law_function(alpha, x):
    return 1 - np.power(1 / x, alpha)


vectorized_slf = np.vectorize(scaling_law_function, excluded=["alpha"])


# Define the log-likelihood function
def compute_log_likelihood(t, p_i):
    return t * np.log(p_i) + (1 - t) * np.log(1 - p_i)


# Define the negative log-likelihood loss
def negative_log_likelihood(params, _occurrences, _outcomes):
    alpha = params
    p_i = vectorized_slf(alpha, _occurrences)
    # Ensure probabilities are within a valid range to avoid log(0)
    p_i = np.clip(p_i, 1e-10, 1 - 1e-10)
    log_likelihood = compute_log_likelihood(_outcomes, p_i)
    return -np.sum(log_likelihood)


def optimize_alphas(data_slice_info):
    # Initial guess for alpha
    initial_params = np.array([0.07])
    bounds = [(0.037, None)]
    _optimized_alphas = []
    for slice_id, _slice_data in data_slice_info.items():
        occurrences = np.array(_slice_data["occurrences"])
        outcomes = np.array(_slice_data["answers"])
        # min_acc = np.array(slice_data["answer_space"])

        # Minimize the negative log-likelihood
        result = minimize(
            negative_log_likelihood,
            x0=initial_params,
            args=(occurrences, outcomes),
            bounds=bounds,  # Lambda must be positive
            method="L-BFGS-B",
        )

        # Optimized alpha
        optimized_alpha = result.x[0]
        print(f"Slice {slice_id}: optimized_alpha: {optimized_alpha}")

        # Check the result
        if result.success:
            print(f"Slice {slice_id}: Optimization was successful. Optimized alpha: {optimized_alpha}")
            _optimized_alphas.append(
                {
                    "slice": slice_id,
                    "alpha": optimized_alpha,
                }
            )
        else:
            logging.warning(f"Slice {slice_id}: Optimization failed:", result.message)
    return _optimized_alphas


def plot_slf(alphas_of_models, occurrences_range):
    plt.figure(figsize=(24, 18))
    for _model_alpha in alphas_of_models:
        for alpha in _model_alpha["Alphas"]:
            probabilities = [scaling_law_function(alpha["alpha"], x) for x in occurrences_range]
            plt.plot(
                occurrences_range, probabilities, label=f"Alpha = {alpha['alpha']:.4f}; " f"Slice {alpha['slice']}; "
            )

    plt.title("Optimized Alphas")
    plt.xlabel("Occurrences")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid()
    plt.show()


def plot_alphas(alphas_of_models: list):
    plt.figure(figsize=(24, 18))

    for _model_alpha in alphas_of_models:
        x = [alpha["slice"] for alpha in _model_alpha["Alphas"]]
        y = [alpha["alpha"] for alpha in _model_alpha["Alphas"]]

        plt.plot(x, y, marker="o", linestyle="-", label=f"{_model_alpha['Model']}")

        # Calculate and plot the average line
        avg_alpha = float(np.mean(y))
        plt.axhline(y=avg_alpha, color="r", linestyle="--", alpha=0.7)

        # Annotate the average line with model name and value
        plt.text(
            x[-1],  # Place the text near the last x-value
            avg_alpha,
            f"{_model_alpha['Model']}; Avg. Alpha: {avg_alpha:.4f}",
            color="red",
            fontsize=12,
            ha="right",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="red", boxstyle="round,pad=0.3"),
        )

    # Add titles, labels, and legend
    plt.title("Optimized Alpha Values", fontsize=16)
    plt.xlabel("Slice", fontsize=14)
    plt.ylabel("Alpha", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.show()


# Plot the log-likelihood scores over the checkpoints
max_occurrence = 8192

optimized_alphas = []
models = ["gpt2_124m", "mamba2_172m", "xlstm_247m"]

for model in models:
    path_to_checkpoints_probing_results = f"../../sample_efficiency_evaluation_results/probing_results/BEAR-big/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/probing_results_on_checkpoints/checkpoint_extracted"
    path_to_increasing_occurrences_in_slices = f"../../sample_efficiency_evaluation_results/probing_results/BEAR-big/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/increasing_occurrences_in_slices.json"

    slice_data = get_slice_data(path_to_checkpoints_probing_results, path_to_increasing_occurrences_in_slices)

    optimized_alphas.append({"Model": model, "Alphas": optimize_alphas(slice_data)})

for model in optimized_alphas:
    utility.utility.save_dict_as_json(
        model,
        f"../../sample_efficiency_evaluation_results/probing_results/BEAR-big/{model['Model']}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/optimized_alphas.json",
    )
# plot_slf(optimized_alphas, np.linspace(1, max_occurrence, 300))
plot_alphas(optimized_alphas)
