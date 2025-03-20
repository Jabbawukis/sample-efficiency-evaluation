import os
import random

import numpy as np
from tqdm import tqdm

import info_gathering.paths as paths

from utility.utility import save_dict_as_json, load_json_dict
from lm_pub_quiz import DatasetResults


def get_splits(
    path_to_relation_info: str,
    split_info: dict):

    occurrences = load_json_dict(path_to_relation_info)
    split_a_list = []
    split_b_list = []
    split_threshold = split_info["threshold"]
    total_num_samples = split_info["total_num_samples"]

    for relation_id, entity_dict in occurrences.items():
        for entity, fact_info in entity_dict.items():
            if fact_info["occurrences"] < split_threshold:
                split_a_list.append((relation_id, entity))
            else:
                split_b_list.append((relation_id, entity))
    split_dict = {}
    for split in split_info["splits"]:
        split_a_percent, split_b_percent = split

        split_a = random.sample(split_a_list, round(total_num_samples*split_a_percent))
        split_b = random.sample(split_b_list, round(total_num_samples*split_b_percent))

        split_dict[split] = {"list": split_a + split_b,
                             "threshold": split_threshold}

    return split_dict



def get_model_answer_for_occurrences_in_data(
    path_to_probing_results: str,
    output: str,
    path_to_relation_info: str,
    split_info: dict
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
            fact_info = {"occurrences_increase": [
                {
                    "total": entity_dict[answer_row.sub_id]["occurrences"],
                    "correct": answer_row.correctly_predicted,
                    "obj_id": entity_dict[answer_row.sub_id]["obj_id"]
                }
            ]}
            fact_info_dict[relation_id][answer_row.sub_id] = fact_info

    for split, fact_list in split_info.items():
        final_output = {}
        split_a_percent, split_b_percent = split

        for fact in fact_list["list"]:
            relation_id, entity_id = fact

            if relation_id not in final_output:
                final_output[relation_id] = {}
            final_output[relation_id][entity_id] = fact_info_dict[relation_id][entity_id]

        save_dict_as_json(final_output,
                          f"{output}/occurrence_info_{split_a_percent}_{fact_list['threshold']}_{split_b_percent}.json")


if __name__ == "__main__":
    models = [
        "gpt2_124m",
        "gpt2_209m",
        "gpt2_355m",
        "mamba2_172m",
        "mamba2_432m",
        "xlstm_247m",
    ]  # results dont depend on other models
    bear_sizes = ["big", "small"]
    # bear_sizes = ["small"]
    abs_path = os.path.abspath(os.path.dirname(__file__)).split("sample-efficiency-evaluation")[0]
    subset_percentage = {
        "big": {
            "threshold": 1024,
            "total_num_samples": 500,
            "splits": [(0.8, 0.2), (0.2, 0.8)],
        },
        "small": {
            "threshold": 1024,
            "total_num_samples": 100,
            "splits": [(0.8, 0.2), (0.2, 0.8)],
        }
    }

    for bear_size in bear_sizes:

        splits = get_splits(
            f"{abs_path}/sample-efficiency-evaluation-results/fact_matching_results/BEAR-{bear_size}/{paths.relation_occurrence_info_wikipedia_20231101_en}",
            subset_percentage[bear_size])

        for model in tqdm(models, desc=f"Evaluating Probe results in BEAR-{bear_size}"):
            probing_results = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.final_model_probing_scores_wikipedia_20231101_en}"
            output_path = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/wikimedia_wikipedia_20231101_en"

            get_model_answer_for_occurrences_in_data(
                path_to_probing_results=probing_results,
                output=output_path,
                path_to_relation_info=f"{abs_path}/sample-efficiency-evaluation-results/fact_matching_results/BEAR-{bear_size}/{paths.relation_occurrence_info_wikipedia_20231101_en}",
                split_info=splits
            )
