import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utility.utility import load_json_dict, save_dict_as_json
from src.experiments.answer_probability_analysis.optimization_utils import (
    negative_log_likelihood,
)
from src.experiments.answer_probability_analysis.config import (
    ABS_PATH,
    MODELS,
    BEAR_SIZES,
    NUM_SLICES,
    FUNCTIONS,
)
import experiments.paths as paths


def plot_nll(model_nll_dict, _output_path, output_diagram_name):
    plt.figure(figsize=(16, 12))
    plt.xticks(range(0, NUM_SLICES))

    for _model, _functions in model_nll_dict.items():
        for _function in _functions:
            nlls = [
                next(
                    (item["value"] for item in _function["NLL_values"] if item["slice"] == str(i)),
                    np.nan,
                )
                for i in range(NUM_SLICES)
            ]

            nlls = np.array(nlls)
            nlls_mask = np.isfinite(nlls)
            xs = np.arange(NUM_SLICES)

            plt.plot(
                xs[nlls_mask],
                nlls[nlls_mask],
                marker="o",
                linestyle="-",
                label=f"{_model} - {_function['Function']}",
            )

    plt.title("Negative Loss Likelihood over Slices", fontsize=16)
    plt.xlabel("Slice", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(_output_path, f"{output_diagram_name}.png"))
    plt.savefig(os.path.join(_output_path, f"{output_diagram_name}.pdf"))
    plt.clf()
    plt.close()


def calculate_nll_for_model(model, bear_size):
    path_to_increasing_occurrences_in_slices = f"{ABS_PATH}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.increasing_occurrences_in_slices_wikipedia_20231101_en}"
    model_dict = {model: []}

    for function in FUNCTIONS:
        nll_sums = []
        slice_data = function["get_slice_data"](NUM_SLICES, path_to_increasing_occurrences_in_slices)
        optimized_params_dict = load_json_dict(
            f"{ABS_PATH}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/correct_answer_probability_optimized_params/optimized_params/{function['file_name']}"
        )
        list_of_optimized_params: list[dict] = optimized_params_dict[function["Params"]]
        available_params_for_slices = {param["slice"]: param for param in list_of_optimized_params}

        for slice_id, _slice_data in slice_data.items():
            if slice_id not in available_params_for_slices:
                continue

            occurrences = np.array(_slice_data["occurrences"])
            outcomes = np.array(_slice_data["answers"])
            total_samples = _slice_data["total_samples"]
            slice_optimized_param = available_params_for_slices[slice_id]
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
    return model_dict


def main():
    for bear_size in BEAR_SIZES:
        nll_on_slices = [
            calculate_nll_for_model(model, bear_size) for model in tqdm(MODELS, desc="Get NLL on slices for models")
        ]

        for model_nll in nll_on_slices:
            model_name = list(model_nll.keys())[0]
            output_path = f"{ABS_PATH}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model_name}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/correct_answer_probability_optimized_params/"
            plot_nll(
                model_nll,
                output_path,
                f"nll_on_slices_bear_{bear_size}",
            )
            save_dict_as_json(
                model_nll,
                f"{output_path}/nll_on_slices.json",
            )


if __name__ == "__main__":
    main()
