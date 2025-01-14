import os
import math

from mpmath import mp
from tqdm import tqdm

from utility.utility import load_json_dict

path_to_checkpoints_probing_results = "../../sample_efficiency_evaluation_results/probing_results/BEAR-big/gpt2_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices/probing_results_on_checkpoints/checkpoint_extracted"
path_to_increasing_occurrences_in_slices = "../../sample_efficiency_evaluation_results/fact_matching_results/BEAR-big/wikimedia_wikipedia_20231101_en/evaluation_on_slices/increasing_occurrences_in_slices.json"

def tanh(z):
    return (mp.exp(z) - mp.exp(-z)) / (mp.exp(z) + mp.exp(-z))

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def get_num(x: str) -> int:
    number = x.split("-")[-1]
    return int(number)

checkpoints: list = os.listdir(path_to_checkpoints_probing_results)
sorted_checkpoints = sorted(checkpoints, key=get_num)
increasing_occurrences = load_json_dict(path_to_increasing_occurrences_in_slices)
scores_on_slices = {}

for idx, checkpoint in enumerate(tqdm(sorted_checkpoints, desc="IRT Score results in slices")):
    metadata = load_json_dict(f"{path_to_checkpoints_probing_results}/{checkpoint}/metadata_results.json")
    irt_sum = []
    for relation_id, entity_dict in increasing_occurrences.items():
        possible_answers: int = len(metadata[relation_id]["answer_space_labels"])
        for entity_id, occurrences_increase in entity_dict.items():
            slice_info = occurrences_increase["occurrences_increase"][idx]
            assert slice_info["Slice"] == idx
            assert slice_info["checkpoint"] == checkpoint
            T = 1 if slice_info["correct"] else 0
            b = slice_info["total"]
            k = 16
            p = 1 / (1 + pow(10, -b/400))
            score = k * (T - p)
            irt_sum.append(score)
    scores_on_slices[f"{idx}"] = sum(irt_sum)
print(scores_on_slices)


