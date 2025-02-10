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
        for relation_id, entity_dict in increasing_occurrences.items():
            # Get number of possible answers for this relation
            for entity_id, occurrences_increase in entity_dict.items():
                slice_info = occurrences_increase["occurrences_increase"][idx]

                # Ensure slice and checkpoint match expectations
                assert slice_info["Slice"] == idx
                assert slice_info["checkpoint"] == checkpoint

                # Extract occurrence and correctness
                T = 1 if slice_info["correct"] else 0

                occurrences_list.append(slice_info["total"])
                answer_list.append(T)

        # Sum scores for the current slice
        data_on_slices[f"{idx}"] = {
            "occurrences": occurrences_list,
            "answers": answer_list,
        }
    return data_on_slices


def power_scaling_function_ext(alpha, x):
    return 1 - np.power(1 / (1 + x), alpha)


vectorized_psf_ext = np.vectorize(power_scaling_function_ext, excluded=["alpha"])


# Define the log-likelihood function
def compute_log_likelihood(t, p_i):
    return t * np.log(p_i) + (1 - t) * np.log(1 - p_i)


# Define the negative log-likelihood loss
def negative_log_likelihood(params, _occurrences, _outcomes):
    alpha = params
    p_i = vectorized_psf_ext(alpha, _occurrences)
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
            bounds=bounds,
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


def plot_alphas(alphas_of_models: list, _output_path: str, output_diagram_name: str):
    plt.figure(figsize=(24, 18))

    # Find the union of all x values and convert them to integers for proper sorting
    all_slices = sorted(
        set(int(slice_val) for model in alphas_of_models for slice_val in [alpha["slice"] for alpha in model["Alphas"]])
    )

    for _model_alphas in alphas_of_models:

        model_slices = {alpha["slice"]: alpha["alpha"] for alpha in _model_alphas["Alphas"]}

        # Get available x and y values (excluding NaNs)
        x_available = np.array([x for x in model_slices.keys()])
        y_available = np.array([model_slices[str(x)] for x in x_available])

        plt.plot(x_available, y_available, marker="o", linestyle="-", label=f"{_model_alphas['Model']}")

        # Exclude NaN values from mean calculation
        avg_alpha = float(np.nanmean(y_available))

        plt.axhline(y=avg_alpha, color="r", linestyle="--", alpha=0.7)

        # Annotate the average line with model name and value
        plt.text(
            all_slices[-1],  # Place the text near the last x-value
            avg_alpha,
            f"{_model_alphas['Model']}; Avg. Alpha: {avg_alpha:.4f}",
            color="red",
            fontsize=12,
            ha="right",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="red", boxstyle="round,pad=0.3"),
        )

    # Ensure all x-axis values are shown
    plt.xticks(all_slices)

    # Add titles, labels, and legend
    plt.title("Optimized Alpha Values", fontsize=16)
    plt.xlabel("Slice", fontsize=14)
    plt.ylabel("Alpha", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(_output_path, f"{output_diagram_name}.png"))
    plt.clf()
    plt.close()


if __name__ == "__main__":
    abs_path = os.path.abspath(os.path.dirname(__file__)).split("sample_efficiency_evaluation")[0]
    output_path = f"{abs_path}/sample_efficiency_evaluation_results/"
    models = ["gpt2_124m", "gpt2_209m", "mamba2_172m", "xlstm_247m"]
    bear_sizes = ["big", "small"]

    for bear_size in bear_sizes:
        optimized_alphas = []
        for model in models:
            path_to_checkpoints_probing_results = f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-big/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/probing_results_on_checkpoints/checkpoint_extracted"
            path_to_increasing_occurrences_in_slices = f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-{bear_size}/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/increasing_occurrences_in_slices.json"

            slice_data = get_slice_data(path_to_checkpoints_probing_results, path_to_increasing_occurrences_in_slices)

            optimized_alphas.append({"Model": model, "Alphas": optimize_alphas(slice_data)})

        for model in optimized_alphas:

            _output_path = f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-{bear_size}/{model['Model']}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/correct_answer_probability_optimized_params/optimized_params/"

            if not os.path.exists(_output_path):
                os.makedirs(_output_path)

            save_dict_as_json(
                model,
                f"{_output_path}/psf-ext_optimized_alphas.json",
            )
        plot_alphas(optimized_alphas, output_path, output_diagram_name=f"psf-ext_optimized_alphas_bear_{bear_size}")
