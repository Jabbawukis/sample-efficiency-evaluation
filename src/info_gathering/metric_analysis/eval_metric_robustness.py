import os
import random

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import info_gathering.paths as paths

from utility.utility import save_dict_as_json, load_json_dict, split_relation_occurrences_info_json_on_occurrences
from lm_pub_quiz import DatasetResults

from info_gathering.model_accuracy.util import (
    get_checkpoint_accuracy_overall,
    get_checkpoint_occurrence_weighted_accuracy,
    get_checkpoint_occurrence_weighted_accuracy_overall,
)
from info_gathering.model_accuracy.eval_model_checkpoint_weighted_accuracy_on_slices import (
    weighting_function,
)
from info_gathering.correct_answer_probability_analysis.probability_function_optimization.psf_nll_optimization import (
    optimize,
)
from info_gathering.correct_answer_probability_analysis.probability_function_optimization.util import (
    get_slice_data,
)


def power_scaling_function(alpha, x, L_0, x_0):
    return 1 - (0 + 0.88 / (np.power((1 + x), alpha)))


vectorized_psf = np.vectorize(power_scaling_function, excluded=["alpha", "L_0", "x_0"])


def plot_scores(data: dict, output_path: str, num_samples: int):
    plt.figure(figsize=(10, 4))
    metrics = ["Accuracy", "WASB", "WAF", "α"]
    colors = {"Accuracy": "tab:blue", "α": "tab:green", "WASB": "tab:red", "WAF": "tab:orange"}

    for split in data["Accuracy"].keys():
        models = list(data["Accuracy"][split].keys())
        x = np.arange(len(models))  # Model positions on x-axis
        width = 0.11  # Reduce width to fit bars properly without overlap

        fig, ax = plt.subplots(figsize=(7, 5))

        bar_containers = []
        labels = []

        for i, metric in enumerate(metrics):
            on_split_values = [data[metric][split][model]["on_split"] for model in models]
            total_values = [data[metric][split][model]["total"] for model in models]

            bar1 = ax.bar(
                x + (i - 1) * 2 * width,
                on_split_values,
                width,
                label=f"{metric} (on split)",
                color=colors[metric],
                alpha=0.6,
            )
            bar2 = ax.bar(
                x + (i - 1) * 2 * width + width,
                total_values,
                width,
                label=f"{metric} (on all data)",
                color=colors[metric],
                alpha=1.0,
                hatch="//",
            )

            bar_containers.append(bar1)
            bar_containers.append(bar2)
            labels.append(f"{metric} (on split)")
            labels.append(f"{metric} (on all data)")

            # Add value labels above bars
            for b in bar1:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height(),
                    f"{b.get_height():.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=50,
                )
            for b in bar2:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height(),
                    f"{b.get_height():.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=50,
                )

        ax.set_ylabel("Scores")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.set_ylim(0, 1.1)
        # Creating legend with metric colors for both on_split and total
        legend_labels = [plt.Rectangle((0, 0), 0.5, 0.5, color=colors[m], alpha=0.6) for m in metrics] + [
            plt.Rectangle((0, 0), 1, 1, fc=colors[m], alpha=1.0, hatch="//") for m in metrics
        ]
        legend_texts = [f"{m} (on split)" for m in metrics] + [f"{m} (on all data)" for m in metrics]
        ax.legend(legend_labels, legend_texts, title="Metrics", loc="upper left")
        # ax.set_xlim(-0.45, len(models) - 0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{split}.png"))
        plt.savefig(os.path.join(output_path, f"{split}.pdf"))
        plt.clf()
        plt.close()


def get_model_answer_for_occurrences_in_data(
    path_to_probing_results: str, path_to_relation_info: str, split_info: dict
):

    occurrences = load_json_dict(path_to_relation_info)
    bear_results = DatasetResults.from_path(path_to_probing_results)
    fact_info_dict = {}

    for relation_id, entity_dict in occurrences.items():
        relation_instance_table = bear_results[relation_id].instance_table
        relation_instance_table["correctly_predicted"] = relation_instance_table.apply(
            lambda row: row.answer_idx == np.argmax(row.pll_scores), axis=1
        )

        fact_info_dict[relation_id] = {}

        for answer_row in relation_instance_table.itertuples():
            fact_info = {
                "occurrences_increase": [
                    {
                        "total": entity_dict[answer_row.sub_id]["occurrences"],
                        "correct": answer_row.correctly_predicted,
                    }
                ],
                "obj_id": entity_dict[answer_row.sub_id]["obj_id"],
            }
            fact_info_dict[relation_id][answer_row.sub_id] = fact_info

    final_output = {}
    for split, fact_list in split_info.items():
        relation_info = {}
        split_a_percent, split_b_percent = split

        for fact in fact_list["list"]:
            relation_id, entity_id = fact

            if relation_id not in relation_info:
                relation_info[relation_id] = {}
            relation_info[relation_id][entity_id] = fact_info_dict[relation_id][entity_id]

        final_output[f"{split_a_percent}_{fact_list['threshold']}_{split_b_percent}"] = relation_info
    return final_output


if __name__ == "__main__":
    abs_path = os.path.abspath(os.path.dirname(__file__)).split("sample-efficiency-evaluation")[0]
    models = [
        "gpt2_209m",
        "gpt2_355m",
        "mamba2_172m",
        "mamba2_432m",
        "xlstm_247m",
        "xlstm_406m",
        "llama_208m",
        "llama_360m",
    ]  # results depend on other models
    num_buckets = 14
    bear_sizes = ["big", "small"]
    subset_percentage = {
        "big": {
            "threshold": 64,
            "total_num_samples": 1000,
            "splits": [(0.8, 0.2), (0.2, 0.8)],
        },
        "small": {
            "threshold": 64,
            "total_num_samples": 500,
            "splits": [(0.8, 0.2), (0.2, 0.8)],
        },
    }

    relation_occurrence_buckets = []
    for i in range(num_buckets):
        if i == num_buckets - 1:
            relation_occurrence_buckets.append((2**i, float("inf")))
            break
        relation_occurrence_buckets.append((2**i, 2 ** (i + 1)))

    # seed = random.randint(0, 1000)
    seed = 93
    for bear_size in bear_sizes:

        splits_file_appendix = (
            f"{subset_percentage[bear_size]['splits'][0][0]}_"
            f"{subset_percentage[bear_size]['splits'][0][1]}_"
            f"{subset_percentage[bear_size]['threshold']}_"
            f"{subset_percentage[bear_size]['splits'][1][0]}_"
            f"{subset_percentage[bear_size]['splits'][1][1]}_seed_{seed}"
        )
        output_path = f"{abs_path}/sample-efficiency-evaluation-results/sample_efficiency_measures/metric_robustness/wikimedia_wikipedia_20231101_en/BEAR-{bear_size}/{splits_file_appendix}/"
        os.makedirs(output_path, exist_ok=True)
        random.seed(seed)
        splits = split_relation_occurrences_info_json_on_occurrences(
            f"{abs_path}/sample-efficiency-evaluation-results/fact_matching_results/BEAR-{bear_size}/{paths.relation_occurrence_info_wikipedia_20231101_en}",
            **subset_percentage[bear_size],
        )

        model_scores = {"Accuracy": {}, "WASB": {}, "WAF": {}, "α": {}}
        model_alphas_dict = {}
        for split in subset_percentage[bear_size]["splits"]:
            model_scores["Accuracy"][f"{split[0]}_{subset_percentage[bear_size]['threshold']}_{split[1]}"] = {}
            model_scores["WASB"][f"{split[0]}_{subset_percentage[bear_size]['threshold']}_{split[1]}"] = {}
            model_scores["WAF"][f"{split[0]}_{subset_percentage[bear_size]['threshold']}_{split[1]}"] = {}
            model_scores["α"][f"{split[0]}_{subset_percentage[bear_size]['threshold']}_{split[1]}"] = {}
            model_alphas_dict[f"{split[0]}_{subset_percentage[bear_size]['threshold']}_{split[1]}"] = {}

        for model in tqdm(models, desc=f"Evaluating Probe results in BEAR-{bear_size}"):
            probing_results_final_model = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.final_model_probing_scores_wikipedia_20231101_en}"
            result = get_model_answer_for_occurrences_in_data(
                path_to_probing_results=probing_results_final_model,
                path_to_relation_info=f"{abs_path}/sample-efficiency-evaluation-results/fact_matching_results/BEAR-{bear_size}/{paths.relation_occurrence_info_wikipedia_20231101_en}",
                split_info=splits,
            )
            samples_path = f"{output_path}/samples"
            os.makedirs(samples_path, exist_ok=True)
            for split, fact_dict in result.items():
                save_dict_as_json(fact_dict, f"{samples_path}/{model}_{split}_bear_{bear_size}.json")
                model_scores["Accuracy"][split][model] = {
                    "on_split": get_checkpoint_accuracy_overall(
                        1, f"{samples_path}/{model}_{split}_bear_{bear_size}.json"
                    )[0],
                    "total": get_checkpoint_accuracy_overall(
                        42,
                        f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.increasing_occurrences_in_slices_wikipedia_20231101_en}",
                    )[41],
                }
                model_scores["WASB"][split][model] = {
                    "on_split": get_checkpoint_occurrence_weighted_accuracy(
                        1,
                        f"{samples_path}/{model}_{split}_bear_{bear_size}.json",
                        weighting_function,
                        relation_occurrence_buckets,
                    )[0],
                    "total": get_checkpoint_occurrence_weighted_accuracy(
                        42,
                        f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.increasing_occurrences_in_slices_wikipedia_20231101_en}",
                        weighting_function,
                        relation_occurrence_buckets,
                    )[41],
                }
                model_scores["WAF"][split][model] = {
                    "on_split": get_checkpoint_occurrence_weighted_accuracy_overall(
                        1, f"{samples_path}/{model}_{split}_bear_{bear_size}.json", weighting_function
                    )[0],
                    "total": get_checkpoint_occurrence_weighted_accuracy_overall(
                        42,
                        f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.increasing_occurrences_in_slices_wikipedia_20231101_en}",
                        weighting_function,
                    )[41],
                }
                model_alphas_dict[split][model] = get_slice_data(
                    1,
                    f"{samples_path}/{model}_{split}_bear_{bear_size}.json",
                )
        for split, model_dict in model_alphas_dict.items():
            optimized_params = optimize(
                model_dict,
                vectorized_psf,
                1,
            )
            for model, optimized_param in optimized_params.items():
                path_to_total_op_alpha = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.model_optimized_params_wikipedia_20231101_en}/psf_optimized_alphas.json"
                total_optimized_alpha = load_json_dict(path_to_total_op_alpha)["Alphas"][41]["alpha"]
                model_scores["α"][split][model] = {
                    "on_split": optimized_param["Alphas"][0]["alpha"],
                    "total": total_optimized_alpha,
                }
        save_dict_as_json(model_scores, f"{output_path}/samples/model_scores.json")
        final_diagram_output_path = f"{output_path}/dias/{bear_size}/{splits_file_appendix}/"
        os.makedirs(final_diagram_output_path, exist_ok=True)
        plot_scores(
            model_scores,
            final_diagram_output_path,
            num_samples=subset_percentage[bear_size]["total_num_samples"],
        )
