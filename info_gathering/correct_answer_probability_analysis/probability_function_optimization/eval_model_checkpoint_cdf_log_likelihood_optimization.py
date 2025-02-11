import os
import numpy as np
from tqdm import tqdm
import logging

import matplotlib.pyplot as plt
from scipy.optimize import minimize

from utility.utility import load_json_dict, save_dict_as_json


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


def cumulative_distribution_function(lambd, x):
    return 0 if x == 0 else 1 - np.exp(-lambd * x)


# Vectorize the CDF to handle arrays
vectorized_cdf = np.vectorize(cumulative_distribution_function, excluded=["lambd"])


# Define the log-likelihood function
def compute_log_likelihood(t, p_i):
    return t * np.log(p_i) + (1 - t) * np.log(1 - p_i)


# Define the negative log-likelihood loss
def negative_log_likelihood(lambd, _occurrences, _outcomes, _total_samples):
    p_i = vectorized_cdf(lambd, _occurrences)
    # Ensure probabilities are within a valid range to avoid log(0)
    p_i = np.clip(p_i, 1e-10, 1 - 1e-10)
    log_likelihood = compute_log_likelihood(_outcomes, p_i)
    return -(1 / _total_samples) * np.sum(log_likelihood)


def optimize_lambdas(data_slice_info):
    # Initial guess for lambda
    initial_params = np.array(0.1)
    bounds = [(1e-5, None)]
    _optimized_lambdas = []
    for slice_id, _slice_data in data_slice_info.items():
        occurrences = np.array(_slice_data["occurrences"])
        outcomes = np.array(_slice_data["answers"])
        total_samples = _slice_data["total_samples"]

        # Minimize the negative log-likelihood
        result = minimize(
            negative_log_likelihood,
            x0=initial_params,
            args=(occurrences, outcomes, total_samples),
            bounds=bounds,  # Lambda must be positive
            method="L-BFGS-B",
        )

        # Optimized lambda
        optimized_lambda = result.x[0]
        print(f"Slice {slice_id}: Optimized lambda: {optimized_lambda}")

        # Check the result
        if result.success:
            print(f"Slice {slice_id}: Optimization was successful. Optimized lambda: {optimized_lambda}")
            _optimized_lambdas.append({"slice": slice_id, "lambda": optimized_lambda})
        else:
            logging.warning(f"Slice {slice_id}: Optimization failed:", result.message)
    return _optimized_lambdas


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


if __name__ == "__main__":
    abs_path = os.path.abspath(os.path.dirname(__file__)).split("sample_efficiency_evaluation")[0]
    models = ["gpt2_124m", "gpt2_209m", "mamba2_172m", "xlstm_247m"]
    bear_sizes = ["big", "small"]

    for bear_size in bear_sizes:
        optimized_lambdas = []
        for model in models:
            path_to_checkpoints_probing_results = f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-big/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/probing_results_on_checkpoints/checkpoint_extracted"
            path_to_increasing_occurrences_in_slices = f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-{bear_size}/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/increasing_occurrences_in_slices.json"

            slice_data = get_slice_data(path_to_checkpoints_probing_results, path_to_increasing_occurrences_in_slices)

            optimized_lambdas.append({"Model": model, "Lambdas": optimize_lambdas(slice_data)})

        for model in optimized_lambdas:

            _output_path = f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-{bear_size}/{model['Model']}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/correct_answer_probability_optimized_params/optimized_params/"

            if not os.path.exists(_output_path):
                os.makedirs(_output_path)

            save_dict_as_json(
                model,
                f"{_output_path}/cdf_optimized_lambdas.json",
            )

        output_path_diagram = f"{abs_path}/sample_efficiency_evaluation_results/correct_answer_probability_analysis_plots/BEAR-{bear_size}/cumulative_distribution_function/"

        if not os.path.exists(output_path_diagram):
            os.makedirs(output_path_diagram)

        plot_lambdas(
            optimized_lambdas, output_path_diagram, output_diagram_name=f"cdf_optimized_lambdas_bear_{bear_size}"
        )
