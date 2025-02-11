import os
import numpy as np
import logging

from utility.utility import save_dict_as_json
from scipy.optimize import minimize

from info_gathering.correct_answer_probability_analysis.probability_function_optimization.util import (
    get_slice_data,
    negative_log_likelihood,
    plot_alphas,
)


def power_scaling_function_ext(alpha, x):
    return 1 - np.power(1 / (1 + x), alpha)


vectorized_psf_ext = np.vectorize(power_scaling_function_ext, excluded=["alpha"])


def optimize_alphas(data_slice_info, vectorized_function, concatenate_all_slices=False):
    # Initial guess for alpha
    initial_params = np.array([0.07])
    bounds = [(0.037, None)]
    _optimized_alphas = []

    if concatenate_all_slices:
        all_occurrences = []
        all_outcomes = []
        all_total_samples = 0

        for slice_id, _slice_data in data_slice_info.items():
            all_occurrences.extend(_slice_data["occurrences"])
            all_outcomes.extend(_slice_data["answers"])
            all_total_samples += _slice_data["total_samples"]

        all_occurrences = np.array(all_occurrences)
        all_outcomes = np.array(all_outcomes)

        # Minimize the negative log-likelihood
        result = minimize(
            negative_log_likelihood,
            x0=initial_params,
            args=(all_occurrences, all_outcomes, all_total_samples, vectorized_function),
            bounds=bounds,
            method="L-BFGS-B",
        )
        optimized_alpha = result.x[0]
        print(f"Final optimized_alpha: {optimized_alpha}")

        return optimized_alpha

    for slice_id, _slice_data in data_slice_info.items():
        occurrences = np.array(_slice_data["occurrences"])
        outcomes = np.array(_slice_data["answers"])
        total_samples = _slice_data["total_samples"]

        # Minimize the negative log-likelihood
        result = minimize(
            negative_log_likelihood,
            x0=initial_params,
            args=(occurrences, outcomes, total_samples, vectorized_function),
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


if __name__ == "__main__":
    abs_path = os.path.abspath(os.path.dirname(__file__)).split("sample_efficiency_evaluation")[0]
    models = ["gpt2_124m", "gpt2_209m", "mamba2_172m", "xlstm_247m"]
    bear_sizes = ["big", "small"]
    optimize_over_all_slices = False

    if optimize_over_all_slices:
        for bear_size in bear_sizes:
            optimized_alphas = []
            for model in models:
                path_to_checkpoints_probing_results = f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-big/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/probing_results_on_checkpoints/checkpoint_extracted"
                path_to_increasing_occurrences_in_slices = f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-{bear_size}/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/increasing_occurrences_in_slices.json"

                slice_data = get_slice_data(
                    path_to_checkpoints_probing_results, path_to_increasing_occurrences_in_slices
                )

                optimized_alphas.append(
                    {"Model": model, "Alpha": optimize_alphas(slice_data, vectorized_psf_ext, optimize_over_all_slices)}
                )

            for model in optimized_alphas:

                _output_path = f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-{bear_size}/{model['Model']}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/correct_answer_probability_optimized_params/optimized_params/"

                if not os.path.exists(_output_path):
                    os.makedirs(_output_path)

                save_dict_as_json(
                    model,
                    f"{_output_path}/psf-ext_optimized_alpha_over_all_slices.json",
                )

    else:

        for bear_size in bear_sizes:
            optimized_alphas = []
            for model in models:
                path_to_checkpoints_probing_results = f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-big/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/probing_results_on_checkpoints/checkpoint_extracted"
                path_to_increasing_occurrences_in_slices = f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-{bear_size}/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/increasing_occurrences_in_slices.json"

                slice_data = get_slice_data(
                    path_to_checkpoints_probing_results, path_to_increasing_occurrences_in_slices
                )

                optimized_alphas.append({"Model": model, "Alphas": optimize_alphas(slice_data, vectorized_psf_ext)})

            for model in optimized_alphas:

                _output_path = f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-{bear_size}/{model['Model']}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/correct_answer_probability_optimized_params/optimized_params/"

                if not os.path.exists(_output_path):
                    os.makedirs(_output_path)

                save_dict_as_json(
                    model,
                    f"{_output_path}/psf-ext_optimized_alphas.json",
                )

            output_path_diagram = f"{abs_path}/sample_efficiency_evaluation_results/correct_answer_probability_analysis_plots/BEAR-{bear_size}/power_scaling_function_extended/"

            if not os.path.exists(output_path_diagram):
                os.makedirs(output_path_diagram)

            plot_alphas(
                optimized_alphas, output_path_diagram, output_diagram_name=f"psf-ext_optimized_alphas_bear_{bear_size}"
            )
