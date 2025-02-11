import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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
        occurrences_list = []
        answer_list = []
        total = 0

        for relation_id, entity_dict in increasing_occurrences.items():
            # Get number of possible answers for this relation
            for entity_id, occurrences_increase in entity_dict.items():
                slice_info = occurrences_increase["occurrences_increase"][idx]

                # Ensure slice and checkpoint match expectations
                assert slice_info["Slice"] == idx
                assert slice_info["checkpoint"] == checkpoint

                total += 1

                # Extract occurrence and correctness
                T = 1 if slice_info["correct"] else 0
                occurrences_list.append(slice_info["total"])
                answer_list.append(T)

        # Sum scores for the current slice
        data_on_slices[f"{idx}"] = {"occurrences": occurrences_list, "answers": answer_list, "total_samples": total}
    return data_on_slices


# Define the log-likelihood function
def compute_log_likelihood(t, p_i):
    return t * np.log(p_i) + (1 - t) * np.log(1 - p_i)


# Define the negative log-likelihood loss
def negative_log_likelihood(param, _occurrences, _outcomes, _total_samples, vectorized_function):
    p_i = vectorized_function(param, _occurrences)
    # Ensure probabilities are within a valid range to avoid log(0)
    p_i = np.clip(p_i, 1e-10, 1 - 1e-10)
    log_likelihood = compute_log_likelihood(_outcomes, p_i)
    return -(1 / _total_samples) * np.sum(log_likelihood)


def plot_alphas(alphas_of_models: list, _output_path: str, output_diagram_name: str):
    plt.figure(figsize=(24, 18))

    # Ensure all x-axis values are shown
    plt.xticks(range(0, 42))

    for _model_alphas in alphas_of_models:

        alphas = []
        count = 0
        for slice_param in _model_alphas["Alphas"]:
            if slice_param["slice"] != str(count):
                alphas.append(np.nan)
                count += 1
            alphas.append(slice_param["alpha"])
            count += 1

        alphas = np.array(alphas)
        alphas_mask = np.isfinite(alphas)
        xs = np.arange(42)

        plt.plot(xs[alphas_mask], alphas[alphas_mask], marker="o", linestyle="-", label=f"{_model_alphas['Model']}")

        # Exclude NaN values from mean calculation
        avg_alpha = float(np.nanmean(alphas))

        plt.axhline(y=avg_alpha, color="r", linestyle="--", alpha=0.7)

        # Annotate the average line with model name and value
        plt.text(
            41,  # Place the text near the last x-value
            avg_alpha,
            f"{_model_alphas['Model']}; Avg. Alpha: {avg_alpha:.4f}",
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
    plt.savefig(os.path.join(_output_path, f"{output_diagram_name}.png"))
    plt.clf()
    plt.close()


def plot_lambdas(lambdas_of_models: list, _output_path: str, output_diagram_name: str):
    plt.figure(figsize=(24, 18))

    # Ensure all x-axis values are shown
    plt.xticks(range(0, 42))

    for _model_lambdas in lambdas_of_models:
        # Get available x and y values (excluding NaNs)
        lambdas = []
        count = 0
        for slice_param in _model_lambdas["Lambdas"]:
            if slice_param["slice"] != str(count):
                lambdas.append(np.nan)
                count += 1
            lambdas.append(slice_param["lambda"])
            count += 1

        lambdas = np.array(lambdas)
        lambdas_mask = np.isfinite(lambdas)
        xs = np.arange(42)

        # Plot lambda values for the model, allowing for missing values without interpolation
        plt.plot(xs[lambdas_mask], lambdas[lambdas_mask], marker="o", linestyle="-", label=f"{_model_lambdas['Model']}")

        # Exclude NaN values from mean calculation
        avg_lambda = float(np.nanmean(lambdas))

        # Plot average lambda line
        plt.axhline(y=avg_lambda, color="r", linestyle="--", alpha=0.7)

        # Annotate the average line with model name and value
        plt.text(
            41,  # Place the text near the last x-value
            avg_lambda,
            f"{_model_lambdas['Model']}; Avg. Lambda: {avg_lambda:.4f}",
            color="red",
            fontsize=12,
            ha="right",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="red", boxstyle="round,pad=0.3"),
        )

    # Add titles, labels, and legend
    plt.title("Optimized Lambda Values", fontsize=16)
    plt.xlabel("Slice", fontsize=14)
    plt.ylabel("Lambda", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(_output_path, f"{output_diagram_name}.png"))
    plt.clf()
    plt.close()
