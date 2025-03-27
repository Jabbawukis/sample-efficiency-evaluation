import os
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import info_gathering.paths as paths

from utility.utility import load_json_dict, save_dict_as_json
from info_gathering.correct_answer_probability_analysis.probability_function_optimization.cdf_nll_optimization import (
    vectorized_cdf,
)
from info_gathering.correct_answer_probability_analysis.probability_function_optimization.psf_nll_optimization import (
    vectorized_psf,
)
from info_gathering.correct_answer_probability_analysis.probability_function_optimization.psf_ext_nll_optimization import (
    vectorized_psf_ext,
)
from info_gathering.correct_answer_probability_analysis.probability_function_optimization.psf_ext2_nll_optimization import (
    vectorized_psf_ext2,
)
from info_gathering.correct_answer_probability_analysis.probability_function_optimization.psf_ext3_nll_optimization import (
    vectorized_psf_ext3,
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
    abs_path = os.path.abspath(os.path.dirname(__file__)).split("sample-efficiency-evaluation")[0]
    nll_on_slices = []
    models = [
        "gpt2_124m",
        "gpt2_209m",
        "gpt2_355m",
        "mamba2_172m",
        "mamba2_432m",
        "xlstm_247m",
        "xlstm_406m",
    ]  # results dont depend on other models
    bear_sizes = ["big", "small"]
    num_slices = 42
    functions = [
        {
            "function_method": vectorized_cdf,
            "function_name": "Cumulative Distribution Function",
            "file_name": "cdf_optimized_lambdas.json",
            "get_slice_data": get_slice_data,
            "Params": "Lambdas",
            "Param": "lambda",
        },
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
        # {
        #     "function_method": vectorized_psf_ext2,
        #     "function_name": "Power Scaling Function Extended2",
        #     "file_name": "psf-ext2_optimized_alphas.json",
        #     "get_slice_data": get_slice_data,
        #     "Params": "Alphas",
        #     "Param": "alpha",
        # },
        {
            "function_method": vectorized_psf_ext3,
            "function_name": "Power Scaling Function Extended3",
            "file_name": "psf-ext3_optimized_alphas.json",
            "get_slice_data": get_slice_data,
            "Params": "Alphas",
            "Param": "alpha",
        },
    ]

    for bear_size in bear_sizes:
        for model in tqdm(models, desc="Get NLL on slices for models"):
            path_to_increasing_occurrences_in_slices = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.increasing_occurrences_in_slices_wikipedia_20231101_en}"

            model_dict = {model: []}
            for function in functions:
                nll_sums = []
                slice_data = function["get_slice_data"](num_slices, path_to_increasing_occurrences_in_slices)

                optimized_params_dict = load_json_dict(
                    f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/correct_answer_probability_optimized_params/optimized_params/{function['file_name']}"
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

                    additional_args = slice_optimized_param["args"] if "args" in slice_optimized_param else None

                    nll_sums.append(
                        {
                            "slice": slice_id,
                            "value": (
                                negative_log_likelihood(
                                    optimized_param,
                                    occurrences,
                                    outcomes,
                                    total_samples,
                                    function["function_method"],
                                )
                                if additional_args is None
                                else negative_log_likelihood(
                                    optimized_param,
                                    occurrences,
                                    outcomes,
                                    total_samples,
                                    function["function_method"],
                                    *additional_args,
                                )
                            ),
                        }
                    )

                model_dict[model].append({"Function": function["function_name"], "NLL_values": nll_sums})
            nll_on_slices.append(model_dict)

        for model in nll_on_slices:
            plot_nll(
                model,
                f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{list(model.keys())[0]}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/correct_answer_probability_optimized_params/",
                f"nll_on_slices_bear_{bear_size}",
            )
            save_dict_as_json(
                model,
                f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{list(model.keys())[0]}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/correct_answer_probability_optimized_params/nll_on_slices.json",
            )
