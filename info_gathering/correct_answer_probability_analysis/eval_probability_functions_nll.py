import os
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from utility.utility import load_json_dict, save_dict_as_json
from info_gathering.correct_answer_probability_analysis.eval_model_checkpoint_psf_log_likelihood_optimization import (
    vectorized_psf,
)
from info_gathering.correct_answer_probability_analysis.eval_model_checkpoint_psf_log_likelihood_optimization import (
    get_slice_data as get_slice_data_psf,
)
from info_gathering.correct_answer_probability_analysis.eval_model_checkpoint_cdf_log_likelihood_optimization import (
    vectorized_cdf,
)
from info_gathering.correct_answer_probability_analysis.eval_model_checkpoint_cdf_log_likelihood_optimization import (
    get_slice_data as get_slice_data_cdf,
)


def compute_log_likelihood(t, p_i):
    return t * np.log(p_i) + (1 - t) * np.log(1 - p_i)


def negative_log_likelihood(_param, vectorized_probability_function, _occurrences, _outcomes):
    p_i = vectorized_probability_function(_param, _occurrences)
    # Ensure probabilities are within a valid range to avoid log(0)
    p_i = np.clip(p_i, 1e-10, 1 - 1e-10)
    log_likelihood = compute_log_likelihood(_outcomes, p_i)
    return -np.sum(log_likelihood)


def plot_nll(model_nll_dict, _output_path, output_diagram_name):
    plt.figure(figsize=(24, 18))
    for _model, _functions in model_nll_dict.items():
        for _function in _functions:

            nlls = np.array([value["value"] for value in _function["NLL_values"]])

            slices = np.array([value["slice"] for value in _function["NLL_values"]])

            plt.plot(slices, nlls, marker="o", linestyle="-", label=f"{_model} - {_function['Function']}")

    plt.title("Negative Loss Likelihood over Slices (lower is better)", fontsize=16)
    plt.xlabel("Slice", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(_output_path, f"{output_diagram_name}.png"))
    plt.clf()
    plt.close()


models = ["gpt2_124m", "gpt2_209m", "mamba2_172m", "xlstm_247m"]
bear_sizes = ["big", "small"]
functions = [
    {
        "function_method": vectorized_psf,
        "function_name": "Power Scaling Function",
        "file_name": "psf_optimized_alphas.json",
        "get_slice_data": get_slice_data_psf,
        "Params": "Alphas",
        "Param": "alpha",
    },
    {
        "function_method": vectorized_cdf,
        "function_name": "Cumulative Distribution Function",
        "file_name": "cdf_optimized_lambdas.json",
        "get_slice_data": get_slice_data_cdf,
        "Params": "Lambdas",
        "Param": "lambda",
    },
]

nll_on_slices = []

for bear_size in bear_sizes:
    for model in tqdm(models, desc="Get NLL on slices for models"):
        path_to_checkpoints_probing_results = f"../../sample_efficiency_evaluation_results/probing_results/BEAR-big/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/probing_results_on_checkpoints/checkpoint_extracted"
        path_to_increasing_occurrences_in_slices = f"../../sample_efficiency_evaluation_results/probing_results/BEAR-{bear_size}/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/increasing_occurrences_in_slices.json"

        model_dict = {model: []}
        for function in functions:
            nll_sums = []
            slice_data = function["get_slice_data"](
                path_to_checkpoints_probing_results, path_to_increasing_occurrences_in_slices
            )

            optimized_params_dict = load_json_dict(
                f"../../sample_efficiency_evaluation_results/probing_results/BEAR-{bear_size}/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/correct_answer_probability_optimized_params/{function['file_name']}"
            )
            list_of_optimized_params: list[dict] = optimized_params_dict[function["Params"]]

            available_params_for_slices = {}
            for param in list_of_optimized_params:
                available_params_for_slices[param["slice"]] = param

            for slice_id, _slice_data in slice_data.items():
                occurrences = np.array(_slice_data["occurrences"])
                outcomes = np.array(_slice_data["answers"])

                try:
                    slice_optimized_param = available_params_for_slices[slice_id]
                except KeyError:
                    continue

                assert slice_optimized_param["slice"] == slice_id

                optimized_param = slice_optimized_param[function["Param"]]

                nll_sums.append(
                    {
                        "slice": slice_id,
                        "value": negative_log_likelihood(
                            optimized_param, function["function_method"], occurrences, outcomes
                        ),
                    }
                )

            model_dict[model].append({"Function": function["function_name"], "NLL_values": nll_sums})
        nll_on_slices.append(model_dict)

    for model in nll_on_slices:
        plot_nll(
            model,
            f"../../sample_efficiency_evaluation_results/probing_results/BEAR-{bear_size}/{list(model.keys())[0]}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/correct_answer_probability_optimized_params/",
            f"nll_on_slices_bear_{bear_size}",
        )
        save_dict_as_json(
            model,
            f"../../sample_efficiency_evaluation_results/probing_results/BEAR-{bear_size}/{list(model.keys())[0]}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/correct_answer_probability_optimized_params/nll_on_slices.json",
        )
