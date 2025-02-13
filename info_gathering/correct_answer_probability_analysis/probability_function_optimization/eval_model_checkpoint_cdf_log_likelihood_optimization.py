import os
import numpy as np
import logging

from scipy.optimize import minimize

from info_gathering.correct_answer_probability_analysis.probability_function_optimization.util import (
    run_optimization,
    negative_log_likelihood,
)


def cumulative_distribution_function(lambd, x):
    return 0 if x == 0 else 1 - np.exp(-lambd * x)


# Vectorize the CDF to handle arrays
vectorized_cdf = np.vectorize(cumulative_distribution_function, excluded=["lambd"])


def optimize(data_slice_info, _vectorized_function, concatenate_all_slices=False):
    # Initial guess for lambda
    initial_params = np.array(0.1)
    bounds = [(1e-5, None)]
    _optimized_lambdas = []

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
            args=(all_occurrences, all_outcomes, all_total_samples, _vectorized_function),
            bounds=bounds,
            method="L-BFGS-B",
        )
        optimized_lambda = result.x[0]
        print(f"Final Optimized lambda: {optimized_lambda}")

        return optimized_lambda

    for slice_id, _slice_data in data_slice_info.items():
        occurrences = np.array(_slice_data["occurrences"])
        outcomes = np.array(_slice_data["answers"])
        total_samples = _slice_data["total_samples"]

        # Minimize the negative log-likelihood
        result = minimize(
            negative_log_likelihood,
            x0=initial_params,
            args=(occurrences, outcomes, total_samples, _vectorized_function),
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


if __name__ == "__main__":
    abs_path = os.path.abspath(os.path.dirname(__file__)).split("sample_efficiency_evaluation")[0]
    models = ["gpt2_124m", "gpt2_209m", "mamba2_172m", "xlstm_247m", "gpt2_355m"] # results depend on other models
    bear_sizes = ["big", "small"]
    optimized_params_for_all_slices_output_file_name = "cdf_optimized_lambdas.json"
    optimized_params_over_all_slices_output_file_name = "cdf_optimized_lambda_over_all_slices.json"
    function_dir_name = "cumulative_distribution_function"
    comparative_plot_output_file_name = "cdf_optimized_lambdas"
    param_name = "Lambdas"
    param_name_key = "lambda"
    optimize_over_all_slices = False
    force_optimization = False

    run_optimization(
        optimize,
        vectorized_cdf,
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
