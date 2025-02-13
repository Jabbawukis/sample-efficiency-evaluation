import os
import numpy as np
import logging

from scipy.optimize import minimize

from info_gathering.correct_answer_probability_analysis.probability_function_optimization.util import (
    negative_log_likelihood,
    run_optimization,
)


def power_scaling_function_ext(alpha, x):
    return 1 - np.power(1 / (1 + x), alpha)


vectorized_psf_ext = np.vectorize(power_scaling_function_ext, excluded=["alpha"])


def optimize(data_slice_info, vectorized_function, concatenate_all_slices=False):
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
    models = ["gpt2_124m", "gpt2_209m", "mamba2_172m", "xlstm_247m", "gpt2_355m"] # results depend on other models
    bear_sizes = ["big", "small"]
    optimized_params_for_all_slices_output_file_name = "psf-ext_optimized_alphas.json"
    optimized_params_over_all_slices_output_file_name = "psf-ext_optimized_alpha_over_all_slices.json"
    function_dir_name = "power_scaling_function_extended"
    comparative_plot_output_file_name = "psf-ext_optimized_alphas"
    param_name = "Alphas"
    param_name_key = "alpha"
    optimize_over_all_slices = False
    force_optimization = False

    run_optimization(
        optimize,
        vectorized_psf_ext,
        abs_path,
        models,
        bear_sizes,
        optimized_params_for_all_slices_output_file_name,
        optimized_params_over_all_slices_output_file_name,
        function_dir_name,
        comparative_plot_output_file_name,
        param_name,
        param_name_key,
        optimize_over_all_slices,
        force_optimization,
    )
