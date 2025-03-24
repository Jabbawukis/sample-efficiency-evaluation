import os
import random

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import info_gathering.paths as paths

from utility.utility import save_dict_as_json, load_json_dict, split_relation_occurrences_info_json_on_occurrences
from lm_pub_quiz import DatasetResults

from info_gathering.model_performance_analysis.util import (
    get_checkpoint_accuracy_overall,
    get_checkpoint_occurrence_weighted_accuracy,
)
from info_gathering.model_performance_analysis.eval_model_checkpoint_weighted_accuracy_on_slices import (
    weighting_function,
)


def plot_scores(scores_models: dict, output: str, output_diagram_name: str, num_samples: int):
    plt.figure(figsize=(16, 10))
    bar_width = 0.2  # Adjust bar width for slimmer bars

    x_labels = list(scores_models.keys())
    scores = [score[0] for score in scores_models.values()]

    colors = ["#1f77b4" if m % 2 == 0 else "#ff7f0e" for m in range(len(x_labels))]

    plt.bar(x_labels, scores, width=bar_width, color=colors)

    # Display the score on top of each bar
    for x, y in zip(x_labels, scores):
        plt.text(x, y + 0.01, f"{y:.4f}", fontsize=12, color="black", ha="center", va="bottom")

    # Add titles, labels, and legend
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Accuracy Scores Over Facts ({num_samples} Samples)", fontsize=16)
    plt.xlabel("Slices", fontsize=14)
    plt.ylabel("Accuracy Score", fontsize=14)
    plt.grid(axis="y", alpha=0.5)
    plt.tight_layout()
    # plt.savefig(os.path.join(output, f"{output_diagram_name}.pdf"))
    plt.savefig(os.path.join(output, f"{output_diagram_name}.png"))
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
                        "obj_id": entity_dict[answer_row.sub_id]["obj_id"],
                    }
                ]
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
        "gpt2_124m",
        "gpt2_209m",
        "gpt2_355m",
        "mamba2_172m",
        "mamba2_432m",
        "xlstm_247m",
    ]  # results dont depend on other models
    num_buckets = 14
    num_slices = 1
    bear_sizes = ["big", "small"]
    subset_percentage = {
        "big": {
            "threshold": 1024,
            "total_num_samples": 500,
            "splits": [(0.9, 0.1), (0.1, 0.9)],
        },
        "small": {
            "threshold": 1024,
            "total_num_samples": 100,
            "splits": [(0.9, 0.1), (0.1, 0.9)],
        },
    }

    relation_occurrence_buckets = []
    for i in range(num_buckets):
        if i == num_buckets - 1:
            relation_occurrence_buckets.append((2**i, float("inf")))
            break
        relation_occurrence_buckets.append((2**i, 2 ** (i + 1)))

    for bear_size in bear_sizes:
        splits_file_appendix = (
            f"{subset_percentage[bear_size]['splits'][0][0]}_"
            f"{subset_percentage[bear_size]['splits'][0][1]}_"
            f"{subset_percentage[bear_size]['threshold']}_"
            f"{subset_percentage[bear_size]['splits'][1][0]}_"
            f"{subset_percentage[bear_size]['splits'][1][1]}"
        )

        random.seed(42)
        splits = split_relation_occurrences_info_json_on_occurrences(
            f"{abs_path}/sample-efficiency-evaluation-results/fact_matching_results/BEAR-{bear_size}/{paths.relation_occurrence_info_wikipedia_20231101_en}",
            **subset_percentage[bear_size],
        )

        model_accuracy_on_split = {}
        model_weighted_accuracy_on_split = {}

        for model in tqdm(models, desc=f"Evaluating Probe results in BEAR-{bear_size}"):
            probing_results_final_model = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.final_model_probing_scores_wikipedia_20231101_en}"

            # output_path = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/wikimedia_wikipedia_20231101_en/occurrence_splits/"
            #####################
            output_path = f"./test/samples/{bear_size}/{splits_file_appendix}/"
            #####################
            os.makedirs(output_path, exist_ok=True)

            result = get_model_answer_for_occurrences_in_data(
                path_to_probing_results=probing_results_final_model,
                path_to_relation_info=f"{abs_path}/sample-efficiency-evaluation-results/fact_matching_results/BEAR-{bear_size}/{paths.relation_occurrence_info_wikipedia_20231101_en}",
                split_info=splits,
            )
            for split, fact_dict in result.items():
                save_dict_as_json(fact_dict, f"{output_path}/{model}_{split}_bear_{bear_size}.json")
                model_accuracy_on_split[f"{model}_{split}"] = get_checkpoint_accuracy_overall(
                    num_slices, f"{output_path}/{model}_{split}_bear_{bear_size}.json"
                )
                model_weighted_accuracy_on_split[f"{model}_{split}"] = get_checkpoint_occurrence_weighted_accuracy(
                    num_slices,
                    f"{output_path}/{model}_{split}_bear_{bear_size}.json",
                    weighting_function,
                    relation_occurrence_buckets,
                )

        # final_diagram_output_path = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/accuracy_over_slices/wikimedia_wikipedia_20231101_en/BEAR-{bear_size}/split_occurrences/"
        #####################
        final_diagram_output_path = f"./test/dias/{bear_size}/{splits_file_appendix}/"
        #####################

        os.makedirs(final_diagram_output_path, exist_ok=True)
        plot_scores(
            model_accuracy_on_split,
            final_diagram_output_path,
            f"accuracy_on_splits_bear_{bear_size}",
            num_samples=subset_percentage[bear_size]["total_num_samples"],
        )
        plot_scores(
            model_weighted_accuracy_on_split,
            final_diagram_output_path,
            f"weighted_accuracy_on_splits_bear_{bear_size}",
            num_samples=subset_percentage[bear_size]["total_num_samples"],
        )
