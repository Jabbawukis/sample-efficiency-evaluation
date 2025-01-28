import os

import numpy as np

from tqdm import tqdm

from utility.utility import save_dict_as_json, count_increasing_occurrences_in_slices
from lm_pub_quiz import DatasetResults


def get_num(x: str) -> int:
    number = x.split("-")[-1]
    return int(number)


def get_model_checkpoint_answer_for_occurrences_in_slices_data(
    path_checkpoints_probing_results: str, output_path_increasing_occurrences_in_slices: str
):
    checkpoints: list = os.listdir(path_checkpoints_probing_results)

    sorted_checkpoints = sorted(checkpoints, key=get_num)
    increasing_occurrences = count_increasing_occurrences_in_slices(
        "../../sample_efficiency_evaluation_results/fact_matching_results/BEAR-big/wikimedia_wikipedia_20231101_en/evaluation_on_slices/relation_info_on_slices"
    )

    for idx, checkpoint in enumerate(tqdm(sorted_checkpoints, desc="Evaluating Probe results in slices")):
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


models = ["gpt2", "mamba2", "xlstm"]

for model in models:
    path_to_checkpoints_probing_results = f"../../sample_efficiency_evaluation_results/probing_results/BEAR-big/{model}_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices/probing_results_on_checkpoints/checkpoint_extracted"
    output_path_to_increasing_occurrences_in_slices = f"../../sample_efficiency_evaluation_results/probing_results/BEAR-big/{model}_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices/increasing_occurrences_in_slices.json"
    get_model_checkpoint_answer_for_occurrences_in_slices_data(
        path_to_checkpoints_probing_results, output_path_to_increasing_occurrences_in_slices
    )
