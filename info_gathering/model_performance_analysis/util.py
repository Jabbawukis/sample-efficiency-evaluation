import os
from typing import Union

import numpy as np
from tqdm import tqdm
from utility.utility import load_json_dict


def get_num(x: str) -> int:
    number = x.split("-")[-1]
    return int(number)


def get_checkpoint_weighted_accuracy(
    path_to_checkpoints: str, path_to_increasing_occurrences_in_slices: str, weighting_function: callable
):
    checkpoints: list = os.listdir(path_to_checkpoints)
    sorted_checkpoints = sorted(checkpoints, key=get_num)
    increasing_occurrences = load_json_dict(path_to_increasing_occurrences_in_slices)
    final_output = {}

    for idx, _checkpoint in enumerate(tqdm(sorted_checkpoints, desc="Evaluating Probe results in slices")):
        relation_accuracy_scores_dict = {}
        sum_of_wights = []
        for relation_id, entity_dict in increasing_occurrences.items():
            for entity_id, fact in entity_dict.items():
                assert fact["occurrences_increase"][idx]["Slice"] == idx
                assert fact["occurrences_increase"][idx]["checkpoint"] == _checkpoint
                occurrences = fact["occurrences_increase"][idx]["total"]

                if occurrences not in relation_accuracy_scores_dict:
                    relation_accuracy_scores_dict[occurrences] = {"correct": 0, "total": 0}

                relation_accuracy_scores_dict[occurrences]["total"] += 1
                if fact["occurrences_increase"][idx]["correct"]:
                    relation_accuracy_scores_dict[occurrences]["correct"] += 1

        accuracy_scores_output = {}
        for occurrence, stats in relation_accuracy_scores_dict.items():
            if stats["total"] == 0:
                continue
            weight = weighting_function(occurrence)
            sum_of_wights.append(weight)
            accuracy_scores_output[occurrence] = {"accuracy": (stats["correct"] / stats["total"]) * weight}
        sum_of_wights = np.sum(np.array(sum_of_wights))
        sum_of_accuracy_scores = np.sum(np.array([stats["accuracy"] for stats in accuracy_scores_output.values()]))
        final_output[idx] = (1/sum_of_wights) * sum_of_accuracy_scores

    return final_output


def get_checkpoint_occurrence_bucket_accuracy(
    path_to_checkpoints: str,
    path_to_increasing_occurrences_in_slices: str,
    relation_occurrence_buckets: list[tuple[int, Union[int, float]]],
):
    checkpoints: list = os.listdir(path_to_checkpoints)
    sorted_checkpoints = sorted(checkpoints, key=get_num)
    increasing_occurrences = load_json_dict(path_to_increasing_occurrences_in_slices)
    final_output = {}

    for idx, _checkpoint in enumerate(tqdm(sorted_checkpoints, desc="Evaluating Probe results in slices")):

        relation_accuracy_scores_dict = {"0": {"correct": 0, "total": 0}}
        for occurrence in relation_occurrence_buckets:
            relation_accuracy_scores_dict[f"{occurrence[0]}-{occurrence[1]}"] = {
                "correct": 0,
                "total": 0,
            }

        for relation_id, entity_dict in increasing_occurrences.items():
            for entity_id, fact in entity_dict.items():
                assert fact["occurrences_increase"][idx]["Slice"] == idx
                assert fact["occurrences_increase"][idx]["checkpoint"] == _checkpoint
                occurrences = fact["occurrences_increase"][idx]["total"]
                if occurrences == 0:
                    relation_accuracy_scores_dict["0"]["total"] += 1
                    if fact["occurrences_increase"][idx]["correct"]:
                        relation_accuracy_scores_dict["0"]["correct"] += 1
                    continue
                for bucket in relation_occurrence_buckets:
                    if bucket[0] <= occurrences <= bucket[1]:
                        relation_accuracy_scores_dict[f"{bucket[0]}-{bucket[1]}"]["total"] += 1
                        if fact["occurrences_increase"][idx]["correct"]:
                            relation_accuracy_scores_dict[f"{bucket[0]}-{bucket[1]}"]["correct"] += 1
        accuracy_scores_output = {}
        for key, bucket in relation_accuracy_scores_dict.items():
            if bucket["total"] == 0:
                continue
            accuracy_scores_output[key] = {
                "accuracy": bucket["correct"] / bucket["total"],
                "correct": bucket["correct"],
                "total": bucket["total"],
            }
        final_output[_checkpoint] = accuracy_scores_output

    # Convert final_output to a DataFrame
    out_put_data = []
    for _checkpoint, buckets in final_output.items():
        for bucket, stats in buckets.items():
            out_put_data.append(
                {
                    "Checkpoint": _checkpoint,
                    "Occurrence Buckets": bucket,
                    "Accuracy": stats["accuracy"],
                    "Total Occurrences": stats["total"],
                }
            )
    return out_put_data
