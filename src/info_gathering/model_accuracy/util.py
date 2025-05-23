import logging
import os
from typing import Union

import numpy as np
from tqdm import tqdm
from utility.utility import load_json_dict


def get_num(x: str) -> int:
    number = x.split("-")[-1]
    return int(number)


def get_checkpoint_occurrence_weighted_accuracy(
    num_slices: int,
    path_to_increasing_occurrences_in_slices: str,
    weighting_function: callable,
    relation_occurrence_buckets: list[tuple[int, Union[int, float]]],
):
    increasing_occurrences = load_json_dict(path_to_increasing_occurrences_in_slices)
    final_output = {}

    for idx in tqdm(range(num_slices), desc="Evaluating Probe results in slices"):
        sum_of_wights = []
        # remove the 0 bucket if filtering out facts with less than 1024 occurrences
        # relation_accuracy_scores_dict = {}
        relation_accuracy_scores_dict = {"0": {"correct": 0, "total": 0, "lower_bound": 0}}
        ########################################################################################
        for occurrence in relation_occurrence_buckets:
            relation_accuracy_scores_dict[f"{occurrence[0]}-{occurrence[1]}"] = {
                "correct": 0,
                "total": 0,
                "lower_bound": occurrence[0],
            }

        for relation_id, entity_dict in increasing_occurrences.items():
            for entity_id, fact in entity_dict.items():
                try:
                    assert fact["occurrences_increase"][idx]["Slice"] == idx
                except (AssertionError, KeyError):
                    logging.warning(f"Warning: slice in dict is not {idx}")
                occurrences = fact["occurrences_increase"][idx]["total"]
                ########################################################################################
                if occurrences == 0:
                    relation_accuracy_scores_dict["0"]["total"] += 1
                    if fact["occurrences_increase"][idx]["correct"]:
                        relation_accuracy_scores_dict["0"]["correct"] += 1
                    continue
                ########################################################################################
                for bucket in relation_occurrence_buckets:
                    bucket_start = bucket[0]
                    bucket_end = bucket[1]
                    if bucket_start <= occurrences < bucket_end:
                        relation_accuracy_scores_dict[f"{bucket_start}-{bucket_end}"]["total"] += 1
                        if fact["occurrences_increase"][idx]["correct"]:
                            relation_accuracy_scores_dict[f"{bucket_start}-{bucket_end}"]["correct"] += 1
                        break

        accuracy_scores_output = {}
        for occurrence, stats in relation_accuracy_scores_dict.items():
            if stats["total"] == 0:
                continue
            weight = weighting_function(stats["lower_bound"])
            sum_of_wights.append(weight)
            accuracy_scores_output[occurrence] = (stats["correct"] / stats["total"]) * weight
        sum_of_wights = np.sum(np.array(sum_of_wights))
        sum_of_accuracy_scores = np.sum(np.array([stats for stats in accuracy_scores_output.values()]))
        final_output[idx] = (1 / sum_of_wights) * sum_of_accuracy_scores
    return final_output


def get_checkpoint_occurrence_weighted_accuracy_overall(
    num_slices: int, path_to_increasing_occurrences_in_slices: str, weighting_function: callable
):
    increasing_occurrences = load_json_dict(path_to_increasing_occurrences_in_slices)
    final_output = {}

    for idx in tqdm(range(num_slices), desc="Evaluating Probe results in slices"):
        sum_of_wights = []
        sum_of_weights_total = []
        for relation_id, entity_dict in increasing_occurrences.items():
            for entity_id, fact in entity_dict.items():
                try:
                    assert fact["occurrences_increase"][idx]["Slice"] == idx
                except (AssertionError, KeyError):
                    logging.warning(f"Warning: slice in dict is not {idx}")
                occurrences = fact["occurrences_increase"][idx]["total"]
                weight = weighting_function(occurrences)
                sum_of_weights_total.append(weight)
                if fact["occurrences_increase"][idx]["correct"]:
                    sum_of_wights.append(weight)
        sum_of_wights = np.sum(np.array(sum_of_wights))
        sum_of_weights_total = np.sum(np.array(sum_of_weights_total))
        final_output[idx] = sum_of_wights / sum_of_weights_total
    return final_output


def get_checkpoint_accuracy_overall(num_slices: int, path_to_increasing_occurrences_in_slices: str):
    increasing_occurrences = load_json_dict(path_to_increasing_occurrences_in_slices)
    final_output = {}

    for idx in tqdm(range(num_slices), desc="Evaluating Probe results in slices"):
        correct = 0
        total = 0
        for relation_id, entity_dict in increasing_occurrences.items():
            for entity_id, fact in entity_dict.items():

                # filter out facts with less than 1024 occurrences
                # if fact["occurrences_increase"][idx]["total"] >= 1024:
                #     pass
                # else:
                #     continue

                # filter out facts with less than 1024 occurrences
                # if fact["occurrences_increase"][idx]["total"] < 1024:
                #     pass
                # else:
                #     continue

                try:
                    assert fact["occurrences_increase"][idx]["Slice"] == idx
                except (AssertionError, KeyError):
                    logging.warning(f"Warning: slice in dict is not {idx}")
                if fact["occurrences_increase"][idx]["correct"]:
                    correct += 1
                total += 1
        final_output[idx] = correct / total
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
                    bucket_start = bucket[0]
                    bucket_end = bucket[1]
                    if bucket_start <= occurrences < bucket_end:
                        relation_accuracy_scores_dict[f"{bucket_start}-{bucket_end}"]["total"] += 1
                        if fact["occurrences_increase"][idx]["correct"]:
                            relation_accuracy_scores_dict[f"{bucket_start}-{bucket_end}"]["correct"] += 1
                        break
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
                    "Frequency Buckets": bucket,
                    "Accuracy": stats["accuracy"],
                    "Frequency": stats["total"],
                }
            )
    return out_put_data
