import logging
import os
import numpy as np
import info_gathering.paths as paths

from scipy.optimize import minimize
from tqdm import tqdm
from info_gathering.correct_answer_probability_analysis.probability_function_optimization.util import (
    get_slice_data,
    plot_params,
)
from utility.utility import save_dict_as_json, load_json_dict


def power_scaling_function_ext3(alpha, x, L_0, x_0):
    return 1 - (L_0 + x_0 / (np.power((1 + x), alpha)))


vectorized_psf_ext3 = np.vectorize(power_scaling_function_ext3, excluded=["alpha", "L_0", "x_0"])


def compute_log_likelihood(t, p_i):
    return t * np.log(p_i) + (1 - t) * np.log(1 - p_i)


def negative_log_likelihood(params, _occurrences, _outcomes, _total_samples, num_models, vectorized_function):
    alphas = params[:num_models]

    L_0 = params[num_models]
    x_0 = params[num_models + 1]

    _occurrences_splits = np.split(_occurrences, num_models)
    _outcomes_splits = np.split(_outcomes, num_models)

    log_likelihoods = []

    for alpha, occurrence_split, outcomes_split in zip(alphas, _occurrences_splits, _outcomes_splits):
        p_i = vectorized_function(alpha, occurrence_split, L_0, x_0)
        p_i = np.clip(p_i, 1e-10, 1 - 1e-10)
        log_likelihoods.extend(compute_log_likelihood(outcomes_split, p_i))

    return -(1 / _total_samples) * np.sum(log_likelihoods)


def optimize(data_slice_info, vectorized_function, num_slices=42, _optimize_over_all_slices=False):
    initial_params = [0.07] * len(data_slice_info)  # alphas
    initial_params.append(0)  # L_0
    initial_params.append(1)  # x_0
    initial_params = np.array(initial_params)

    bounds = [(0.037, None)] * len(data_slice_info)  # alphas
    bounds.append((0, None))  # L_0
    bounds.append((0, None))  # x_0

    final_optimized_params_dict = {}
    if _optimize_over_all_slices:
        all_occurrences = []
        all_outcomes = []
        all_total_samples = 0

        for _model, _slice_data in data_slice_info.items():
            for slice_id in range(num_slices):
                all_occurrences.extend(_slice_data[str(slice_id)]["occurrences"])
                all_outcomes.extend(_slice_data[str(slice_id)]["answers"])
                all_total_samples += _slice_data[str(slice_id)]["total_samples"]

        all_occurrences = np.array(all_occurrences)
        all_outcomes = np.array(all_outcomes)

        # Minimize the negative log-likelihood
        result = minimize(
            negative_log_likelihood,
            x0=initial_params,
            args=(all_occurrences, all_outcomes, all_total_samples, len(data_slice_info), vectorized_function),
            bounds=bounds,
            method="L-BFGS-B",
        )
        L_0 = result.x[-2]
        x_0 = result.x[-1]
        alphas = result.x[:-2]
        print(f"Final x_0: {x_0}")
        print(f"Final L_0: {L_0}")

        for optimized_param, _model in zip(alphas, data_slice_info.keys()):
            final_optimized_params_dict[_model] = {"Model": _model, "Alphas": optimized_param}
            print(f"Final optimized alpha for {_model}: {optimized_param}")
        return final_optimized_params_dict

    for slice_id in tqdm(range(num_slices), desc=f"Optimizing parameters for each slice (total: {num_slices})"):
        all_occurrences = []
        all_outcomes = []
        all_total_samples = 0

        for _model, _slice_data in data_slice_info.items():
            if _model not in final_optimized_params_dict:
                final_optimized_params_dict[_model] = {"Model": _model, "Alphas": []}
            all_occurrences.extend(_slice_data[str(slice_id)]["occurrences"])
            all_outcomes.extend(_slice_data[str(slice_id)]["answers"])
            all_total_samples += _slice_data[str(slice_id)]["total_samples"]

        all_occurrences = np.array(all_occurrences)
        all_outcomes = np.array(all_outcomes)

        # Minimize the negative log-likelihood
        result = minimize(
            negative_log_likelihood,
            x0=initial_params,
            args=(all_occurrences, all_outcomes, all_total_samples, len(data_slice_info), vectorized_function),
            bounds=bounds,
            method="L-BFGS-B",
        )
        L_0 = result.x[-2]
        x_0 = result.x[-1]
        print(f"Final x_0: {x_0}")
        print(f"Final L_0: {L_0}")

        for optimized_param, _model in zip(result.x[:-2], data_slice_info.keys()):
            final_optimized_params_dict[_model]["Alphas"].append(
                {
                    "slice": str(slice_id),
                    "alpha": optimized_param,
                    "args": [L_0, x_0],
                }
            )
            print(f"Final optimized alpha for {_model}: {optimized_param}")

    return final_optimized_params_dict


if __name__ == "__main__":
    abs_path = os.path.abspath(os.path.dirname(__file__)).split("sample-efficiency-evaluation")[0]
    models = [
        "gpt2_124m",
        "gpt2_209m",
        "gpt2_355m",
        "mamba2_172m",
        "mamba2_432m",
        "xlstm_247m",
    ]  # results depend on other models
    bear_sizes = ["big", "small"]
    optimized_params_for_all_slices_output_file_name = "psf-ext3_optimized_alphas.json"
    optimized_params_over_all_slices_output_file_name = "psf-ext3_optimized_alpha_over_all_slices.json"
    function_dir_name = "power_scaling_function_extended3"
    comparative_plot_output_file_name = "psf-ext3_optimized_alphas"
    param_name = "Alphas"
    param_name_key = "alpha"
    num_slices = 42
    optimize_over_all_slices = False  # optimize the values over all slices for each model at once (takes a lot of time)
    skip_optimisation = False  # skip optimization and load the optimized parameters (if already optimized)

    for bear_size in bear_sizes:
        model_dict = {}

        if optimize_over_all_slices:
            output_file_name_json = optimized_params_over_all_slices_output_file_name
        else:
            output_file_name_json = optimized_params_for_all_slices_output_file_name
        if not skip_optimisation:
            for model in models:
                path_to_checkpoints_probing_results = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-big/{model}/{paths.checkpoints_extracted_wikipedia_20231101_en}"
                path_to_increasing_occurrences_in_slices = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.increasing_occurrences_in_slices_wikipedia_20231101_en}"

                _output_path = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.model_optimized_params_wikipedia_20231101_en}"

                logging.info(f"Optimizing for model {model}")
                print(f"Optimizing for model {model}")
                if not os.path.exists(_output_path):
                    os.makedirs(_output_path)
                slice_data = get_slice_data(num_slices, path_to_increasing_occurrences_in_slices)
                model_dict[model] = slice_data

            optimized_params = optimize(
                model_dict, vectorized_psf_ext3, num_slices, _optimize_over_all_slices=optimize_over_all_slices
            )

            for model, optimized_param in optimized_params.items():
                _output_path = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.model_optimized_params_wikipedia_20231101_en}"
                save_dict_as_json(
                    optimized_param,
                    f"{_output_path}/{output_file_name_json}",
                )
                print(f"Optimized parameters for model {model} saved successfully")
        else:
            optimized_params = {}
            for model in models:
                logging.info(f"Skipping optimizing for model {model} and loading the optimized parameters")
                print(f"Skipping optimizing for model {model} and loading the optimized parameters")
                _output_path = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.model_optimized_params_wikipedia_20231101_en}"
                optimized_params[model] = load_json_dict(f"{_output_path}/{output_file_name_json}")

        if not optimize_over_all_slices:
            output_path_diagram = f"{abs_path}/sample-efficiency-evaluation-results/correct_answer_probability_analysis_plots/BEAR-{bear_size}/{function_dir_name}/"

            if not os.path.exists(output_path_diagram):
                os.makedirs(output_path_diagram)

            optimized_params_list = []
            for _, optimized_param in optimized_params.items():
                optimized_params_list.append(optimized_param)

            plot_params(
                optimized_params_list,
                output_path_diagram,
                output_diagram_name=f"{comparative_plot_output_file_name}_bear_{bear_size}",
                param_name=param_name,
                param_name_key=param_name_key,
            )
