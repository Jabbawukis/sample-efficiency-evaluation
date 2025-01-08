import os

import numpy as np

from tqdm import tqdm

from utility.utility import load_json_dict, save_dict_as_json, count_increasing_occurrences_in_slices
from lm_pub_quiz import DatasetResults

path_to_checkpoints = ""
path_to_increasing_occurrences_in_slices = "" # count_increasing_occurrences_in_slices output

def get_num(x: str) -> int:
    number = x.split("-")[-1]
    return int(number)

checkpoints: list = os.listdir(path_to_checkpoints)

sorted_checkpoints = sorted(checkpoints, key=get_num)
increasing_occurrences = load_json_dict(path_to_increasing_occurrences_in_slices)

for idx, checkpoint in enumerate(tqdm(sorted_checkpoints, desc="Counting increasing occurrences in slices")):
    bear_results = DatasetResults.from_path(f"{path_to_checkpoints}/{checkpoint}")
    for relation_id, entity_dict in increasing_occurrences.items():
        relation_instance_table = bear_results[relation_id].instance_table
        relation_instance_table["correctly_predicted"] = relation_instance_table.apply(
            lambda row: row.answer_idx == np.argmax(row.pll_scores), axis=1
        )
        for answer_row in relation_instance_table.itertuples():
            entity_dict[answer_row.sub_id]["occurrences_increase"][idx]["correct"] = answer_row.correctly_predicted
            entity_dict[answer_row.sub_id]["occurrences_increase"][idx]["checkpoint"] = checkpoint
            assert entity_dict[answer_row.sub_id]["occurrences_increase"][idx]["Slice"] == idx

save_dict_as_json(increasing_occurrences, path_to_increasing_occurrences_in_slices)