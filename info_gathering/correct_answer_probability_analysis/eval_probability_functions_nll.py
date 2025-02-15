import os
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from utility.utility import load_json_dict, save_dict_as_json
from info_gathering.correct_answer_probability_analysis.probability_function_optimization.eval_model_checkpoint_psf_log_likelihood_optimization import (
    vectorized_psf,
)
from info_gathering.correct_answer_probability_analysis.probability_function_optimization.eval_model_checkpoint_psf_ext_log_likelihood_optimization import (
    vectorized_psf_ext,
)
from info_gathering.correct_answer_probability_analysis.probability_function_optimization.eval_model_checkpoint_cdf_log_likelihood_optimization import (
    vectorized_cdf,
)

from info_gathering.correct_answer_probability_analysis.probability_function_optimization.util import (
    get_slice_data,
    negative_log_likelihood,
)


def plot_nll(model_nll_dict, _output_path, output_diagram_name):
    plt.figure(figsize=(24, 18))
    # Ensure all x-axis values are shown
    plt.xticks(range(0, 42))

    for _model, _functions in model_nll_dict.items():
        for _function in _functions:

            nlls = []
            count = 0
            for value in _function["NLL_values"]:
                if value["slice"] != str(count):
                    nlls.append(np.nan)
                    count += 1
                nlls.append(value["value"])
                count += 1

            nlls = np.array(nlls)
            nlls_mask = np.isfinite(nlls)
            xs = np.arange(42)

            plt.plot(
                xs[nlls_mask], nlls[nlls_mask], marker="o", linestyle="-", label=f"{_model} - {_function['Function']}"
            )

    plt.title("Negative Loss Likelihood over Slices (lower is better)", fontsize=16)
    plt.xlabel("Slice", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(_output_path, f"{output_diagram_name}.png"))
    plt.clf()
    plt.close()


if __name__ == "__main__":
    abs_path = os.path.abspath(os.path.dirname(__file__)).split("sample_efficiency_evaluation")[0]
    nll_on_slices = []
    models = []  # results dont depend on other models
    bear_sizes = ["big", "small"]
    functions = [
        {
            "function_method": vectorized_psf,
            "function_name": "Power Scaling Function",
            "file_name": "psf_optimized_alphas.json",
            "get_slice_data": get_slice_data,
            "Params": "Alphas",
            "Param": "alpha",
        },
        {
            "function_method": vectorized_psf_ext,
            "function_name": "Power Scaling Function Extended",
            "file_name": "psf-ext_optimized_alphas.json",
            "get_slice_data": get_slice_data,
            "Params": "Alphas",
            "Param": "alpha",
        },
        {
            "function_method": vectorized_cdf,
            "function_name": "Cumulative Distribution Function",
            "file_name": "cdf_optimized_lambdas.json",
            "get_slice_data": get_slice_data,
            "Params": "Lambdas",
            "Param": "lambda",
        },
    ]

    for bear_size in bear_sizes:
        for model in tqdm(models, desc="Get NLL on slices for models"):
            path_to_checkpoints_probing_results = f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-big/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/probing_results_on_checkpoints/checkpoint_extracted"
            path_to_increasing_occurrences_in_slices = f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-{bear_size}/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/increasing_occurrences_in_slices.json"

            model_dict = {model: []}
            for function in functions:
                nll_sums = []
                slice_data = function["get_slice_data"](
                    path_to_checkpoints_probing_results, path_to_increasing_occurrences_in_slices
                )

                optimized_params_dict = load_json_dict(
                    f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-{bear_size}/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/correct_answer_probability_optimized_params/optimized_params/{function['file_name']}"
                )
                list_of_optimized_params: list[dict] = optimized_params_dict[function["Params"]]

                available_params_for_slices = {}
                for param in list_of_optimized_params:
                    available_params_for_slices[param["slice"]] = param

                for slice_id, _slice_data in slice_data.items():
                    occurrences = np.array(_slice_data["occurrences"])
                    outcomes = np.array(_slice_data["answers"])
                    total_samples = _slice_data["total_samples"]

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
                                optimized_param,
                                occurrences,
                                outcomes,
                                total_samples,
                                function["function_method"],
                            ),
                        }
                    )

                model_dict[model].append({"Function": function["function_name"], "NLL_values": nll_sums})
            nll_on_slices.append(model_dict)

        for model in nll_on_slices:
            plot_nll(
                model,
                f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-{bear_size}/{list(model.keys())[0]}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/correct_answer_probability_optimized_params/",
                f"nll_on_slices_bear_{bear_size}",
            )
            save_dict_as_json(
                model,
                f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-{bear_size}/{list(model.keys())[0]}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/correct_answer_probability_optimized_params/nll_on_slices.json",
            )
