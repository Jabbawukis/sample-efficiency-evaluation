import logging
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import info_gathering.paths as paths

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
        data_on_slices[str(idx)] = {"occurrences": occurrences_list, "answers": answer_list, "total_samples": total}
    return data_on_slices


# Define the log-likelihood function
def compute_log_likelihood(t, p_i):
    return t * np.log(p_i) + (1 - t) * np.log(1 - p_i)


# Define the negative log-likelihood loss
def negative_log_likelihood(param, _occurrences, _outcomes, _total_samples, vectorized_function, *args):
    if args:
        p_i = vectorized_function(param, _occurrences, *args)
    else:
        p_i = vectorized_function(param, _occurrences)
    # Ensure probabilities are within a valid range to avoid log(0)
    p_i = np.clip(p_i, 1e-10, 1 - 1e-10)
    log_likelihood = compute_log_likelihood(_outcomes, p_i)
    return -(1 / _total_samples) * np.sum(log_likelihood)


def plot_params(
    params_of_models: list, _output_path: str, output_diagram_name: str, param_name: str, param_name_key: str
):
    plt.figure(figsize=(24, 18))

    # Ensure all x-axis values are shown
    plt.xticks(range(0, 42))

    for _model_params in params_of_models:
        # Get available x and y values (excluding NaNs)
        params = []
        count = 0
        for slice_param in _model_params[param_name]:
            if slice_param["slice"] != str(count):
                params.append(np.nan)
                count += 1
            params.append(slice_param[param_name_key])
            count += 1

        params = np.array(params)
        params_mask = np.isfinite(params)
        xs = np.arange(42)
        plt.plot(xs[params_mask], params[params_mask], marker="o", linestyle="-", label=f"{_model_params['Model']}")

    # Add titles, labels, and legend
    plt.title("Optimized Values", fontsize=16)
    plt.xlabel("Slices", fontsize=14)
    plt.ylabel(param_name, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(_output_path, f"{output_diagram_name}.png"))
    plt.clf()
    plt.close()


def run_optimization(
    optimize: callable,
    vectorized_function: callable,
    abs_path: str,
    models: list[str],
    bear_sizes: list[str],
    optimized_params_output_file_name: str,
    optimized_param_all_slices_output_file_name: str,
    function_dir_name: str,
    comparative_plot_output_file_name: str,
    param_name: str,
    param_name_key: str,
    optimize_over_all_slices: bool,
    force_optimization: bool,
):

    if optimize_over_all_slices:
        output_file_name_json = optimized_param_all_slices_output_file_name
    else:
        output_file_name_json = optimized_params_output_file_name

    for bear_size in bear_sizes:
        optimized_params = []
        for model in models:
            path_to_checkpoints_probing_results = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-big/{model}/{paths.checkpoints_extracted_wikipedia_20231101_en}"
            path_to_increasing_occurrences_in_slices = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.increasing_occurrences_in_slices_wikipedia_20231101_en}"

            _output_path = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.model_optimized_params_wikipedia_20231101_en}"

            if os.path.exists(f"{_output_path}/{output_file_name_json}"):
                if force_optimization:
                    logging.info(f"Optimizing for model {model}")
                    print(f"Optimizing for model {model}")
                    slice_data = get_slice_data(
                        path_to_checkpoints_probing_results, path_to_increasing_occurrences_in_slices
                    )
                    model_dict = {
                        "Model": model,
                        param_name: optimize(slice_data, vectorized_function, optimize_over_all_slices),
                    }
                    optimized_params.append(model_dict)
                    save_dict_as_json(
                        model_dict,
                        f"{_output_path}/{output_file_name_json}",
                    )
                else:
                    logging.info(f"Optimisation for model {model} already exist")
                    print(f"Optimisation for model {model} already exist")
                    if not optimize_over_all_slices:
                        optimized_params.append(load_json_dict(f"{_output_path}/{output_file_name_json}"))
            else:
                logging.info(f"Optimizing for model {model}")
                print(f"Optimizing for model {model}")
                if not os.path.exists(_output_path):
                    os.makedirs(_output_path)
                slice_data = get_slice_data(
                    path_to_checkpoints_probing_results, path_to_increasing_occurrences_in_slices
                )
                model_dict = {
                    "Model": model,
                    param_name: optimize(slice_data, vectorized_function, optimize_over_all_slices),
                }
                optimized_params.append(model_dict)
                save_dict_as_json(
                    model_dict,
                    f"{_output_path}/{output_file_name_json}",
                )

        if not optimize_over_all_slices:
            output_path_diagram = f"{abs_path}/sample-efficiency-evaluation-results/correct_answer_probability_analysis_plots/BEAR-{bear_size}/{function_dir_name}/"

            if not os.path.exists(output_path_diagram):
                os.makedirs(output_path_diagram)

            plot_params(
                optimized_params,
                output_path_diagram,
                output_diagram_name=f"{comparative_plot_output_file_name}_bear_{bear_size}",
                param_name=param_name,
                param_name_key=param_name_key,
            )
