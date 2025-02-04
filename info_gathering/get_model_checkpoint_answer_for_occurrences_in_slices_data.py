import os
from typing import Optional

import numpy as np

from tqdm import tqdm

from utility.utility import save_dict_as_json, count_increasing_occurrences_in_slices, load_json_dict
from lm_pub_quiz import DatasetResults


def get_num(x: str) -> int:
    number = x.split("-")[-1]
    return int(number)


def get_model_checkpoint_answer_for_occurrences_in_slices_data(
    path_checkpoints_probing_results: str,
    output_path_increasing_occurrences_in_slices: str,
    path_to_relation_info_on_slices: str,
    subset_indices: Optional[str] = None,
):
    checkpoints: list = os.listdir(path_checkpoints_probing_results)

    sorted_checkpoints = sorted(checkpoints, key=get_num)
    increasing_occurrences = count_increasing_occurrences_in_slices(path_to_relation_info_on_slices)

    for idx, checkpoint in enumerate(tqdm(sorted_checkpoints, desc="Evaluating Probe results in slices")):

        if subset_indices:
            subset_indices_dict: dict = load_json_dict(subset_indices)
            bear_results = DatasetResults.from_path(f"{path_checkpoints_probing_results}/{checkpoint}").filter_subset(
                subset_indices_dict
            )
        else:
            bear_results = DatasetResults.from_path(f"{path_checkpoints_probing_results}/{checkpoint}")

        for relation_id, entity_dict in increasing_occurrences.items():
            relation_instance_table = bear_results[relation_id].instance_table
            relation_instance_table["correctly_predicted"] = relation_instance_table.apply(
                lambda row: row.answer_idx == np.argmax(row.pll_scores), axis=1
            )
            for answer_row in relation_instance_table.itertuples():
                entity_dict[answer_row.sub_id]["occurrences_increase"][idx]["correct"] = answer_row.correctly_predicted
                entity_dict[answer_row.sub_id]["occurrences_increase"][idx]["checkpoint"] = checkpoint
                assert entity_dict[answer_row.sub_id]["occurrences_increase"][idx]["Slice"] == idx

    save_dict_as_json(increasing_occurrences, output_path_increasing_occurrences_in_slices)


models = ["gpt2_124m", "mamba2_172m", "xlstm_247m", "gpt2_209m"]

slice_info_small = "../../sample_efficiency_evaluation_results/fact_matching_results/BEAR-small/wikimedia_wikipedia_20231101_en/evaluation_on_slices/relation_info_on_slices"
slice_info_big = "../../sample_efficiency_evaluation_results/fact_matching_results/BEAR-big/wikimedia_wikipedia_20231101_en/evaluation_on_slices/relation_info_on_slices"


for model in models:
    path_to_checkpoints_probing_results = f"../../sample_efficiency_evaluation_results/probing_results/BEAR-big/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/probing_results_on_checkpoints/checkpoint_extracted"
    output_path_to_increasing_occurrences_in_slices = f"../../sample_efficiency_evaluation_results/probing_results/BEAR-big/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/increasing_occurrences_in_slices.json"
    get_model_checkpoint_answer_for_occurrences_in_slices_data(
        path_to_checkpoints_probing_results, output_path_to_increasing_occurrences_in_slices, slice_info_big
    )

for model in models:
    path_to_checkpoints_probing_results = f"../../sample_efficiency_evaluation_results/probing_results/BEAR-big/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/probing_results_on_checkpoints/checkpoint_extracted"
    output_path_to_increasing_occurrences_in_slices = f"../../sample_efficiency_evaluation_results/probing_results/BEAR-small/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/increasing_occurrences_in_slices.json"

    get_model_checkpoint_answer_for_occurrences_in_slices_data(
        path_to_checkpoints_probing_results,
        output_path_to_increasing_occurrences_in_slices,
        slice_info_small,
        "../BEAR/bear_lite_indices.json",
    )
