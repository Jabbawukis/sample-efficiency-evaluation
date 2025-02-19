import os
import numpy as np

from scipy.optimize import minimize
from tqdm import tqdm
from info_gathering.correct_answer_probability_analysis.probability_function_optimization.util import get_slice_data
from utility.utility import save_dict_as_json

def power_scaling_function_ext(alpha, x_0, L_0, x):
    return 1 - (L_0 + np.power(x_0 / (1 + x), alpha))


vectorized_psf_ext = np.vectorize(power_scaling_function_ext, excluded=["alpha", "x_0", "L_0"])


def compute_log_likelihood(t, p_i):
    return t * np.log(p_i) + (1 - t) * np.log(1 - p_i)


def negative_log_likelihood(params, _occurrences, _outcomes, _total_samples, num_models, vectorized_function):

    alphas = params[:num_models]

    x_0 = params[num_models]
    L_0 = params[num_models + 1]

    _occurrences_splits = np.split(_occurrences, num_models)
    _outcomes_splits = np.split(_outcomes, num_models)

    log_likelihoods = []

    for alpha, occurrence_split, outcomes_split in zip(alphas, _occurrences_splits, _outcomes_splits):
        p_i = vectorized_function(alpha, x_0, L_0, occurrence_split)
        p_i = np.clip(p_i, 1e-10, 1 - 1e-10)
        log_likelihoods.extend(compute_log_likelihood(outcomes_split, p_i))

    return -(1 / _total_samples) * np.sum(log_likelihoods)


def optimize(data_slice_info, vectorized_function, num_slices=42):

    final_optimized_params_dict = {}

    for slice_id in tqdm(range(num_slices), desc=f"Optimizing parameters for each slice (total: {num_slices})"):
        all_occurrences = []
        all_outcomes = []
        all_total_samples = 0
        optimized_params_dict = {}

        for _model, _slice_data in data_slice_info.items():
            all_occurrences.extend(_slice_data[str(slice_id)]["occurrences"])
            all_outcomes.extend(_slice_data[str(slice_id)]["answers"])
            all_total_samples += _slice_data[str(slice_id)]["total_samples"]

        all_occurrences = np.array(all_occurrences)
        all_outcomes = np.array(all_outcomes)

        initial_params = [0.07] * len(data_slice_info)  # alphas
        initial_params.append(1)  # x_0
        initial_params.append(0)  # L_0
        initial_params = np.array(initial_params)

        bounds = [(0.037, None)] * len(data_slice_info)  # alphas
        bounds.append((0, None))  # x_0
        bounds.append((0, None))  # L_0

        # Minimize the negative log-likelihood
        result = minimize(
            negative_log_likelihood,
            x0=initial_params,
            args=(all_occurrences, all_outcomes, all_total_samples, len(data_slice_info), vectorized_function),
            bounds=bounds,
            method="L-BFGS-B",
        )
        x_0 = result.x[-2]
        L_0 = result.x[-1]
        print(f"Final x_0: {x_0}")
        print(f"Final L_0: {L_0}")

        for optimized_param, _model in zip(result.x[:-2], data_slice_info.keys()):
            optimized_params_dict[_model] = optimized_param
            print(f"Final optimized alpha for {_model}: {optimized_param}")

        final_optimized_params_dict[str(slice_id)] = {"alphas": optimized_params_dict, "x_0": x_0, "L_0": L_0}

    return final_optimized_params_dict


if __name__ == "__main__":
    abs_path = os.path.abspath(os.path.dirname(__file__)).split("sample_efficiency_evaluation")[0]
    models = [
        "gpt2_124m",
        "gpt2_209m",
        "gpt2_355m",
        "mamba2_172m",
        "mamba2_432m",
        "xlstm_247m",
    ]  # results depend on other models
    bear_sizes = ["big", "small"]

    for bear_size in bear_sizes:
        model_dict = {}
        for model in models:
            path_to_checkpoints_probing_results = f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-big/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/probing_results_on_checkpoints/checkpoint_extracted"
            path_to_increasing_occurrences_in_slices = f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-{bear_size}/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/increasing_occurrences_in_slices.json"

            slice_data = get_slice_data(path_to_checkpoints_probing_results, path_to_increasing_occurrences_in_slices)
            model_dict[model] = slice_data

        optimized_params = optimize(model_dict, vectorized_psf_ext)
        save_dict_as_json(optimized_params, f"./BEAR-{bear_size}_optimized_params.json")
