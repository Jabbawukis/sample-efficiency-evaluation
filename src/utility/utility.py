import json
import logging
import os
import random
import re
from typing import Optional

from tqdm import tqdm
import matplotlib.pyplot as plt


class SetEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, set):
            return list(o)
        return json.JSONEncoder.default(self, o)


def load_json_dict(json_file_path: str) -> dict:
    """
    Load json file.
    :param json_file_path: Path to json file
    :return: Dictionary containing information
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_dict = json.load(f)
    return json_dict


def load_json_line_dict(json_line_file_path: str) -> list[dict]:
    """
    Load json line file.
    :param json_line_file_path: Path to json file
    :return: List of dictionaries containing information
    """
    json_list = []
    with open(json_line_file_path, "r", encoding="utf-8") as f:
        for line in f:
            json_list.append(json.loads(line))
    return json_list


def save_dict_as_json(dictionary: dict, json_output_file_path: str):
    """
    Save dictionary as json file.
    :param dictionary: Dictionary containing information
    :param json_output_file_path: Path to json file
    """
    with open(json_output_file_path, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, indent=4, ensure_ascii=False, cls=SetEncoder)


def clean_string(text: str) -> str:
    """
    Clean string.
    :param text: Text to clean
    :return: Cleaned text
    """
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")
    return text


def word_in_sentence(word: str, sentence: str, ignore_case: bool = True) -> bool:
    """
    Check if word is in sentence.

    :param word: word to check
    :param sentence: sentence to check
    :param ignore_case: ignore case
    :return: True if word is in sentence, False otherwise
    """
    if ignore_case:
        pattern = re.compile(r"(?<!\w)({0})(?!\w)".format(re.escape(word)), flags=re.IGNORECASE)
    else:
        pattern = re.compile(r"(?<!\w)({0})(?!\w)".format(re.escape(word)))
    return bool(pattern.search(sentence))


def create_fact_occurrence_histogram(
    path_to_rel_info_file: str,
    num_buckets: int = 14,
    output_diagram_name: str = "occurrence_statistics",
    output_path: Optional[str] = None,
) -> None:
    """
    Create fact occurrence statistics and plot a histogram.
    The bucket end is exclusive.

    :param output_diagram_name: Name of the output diagram.
    :param num_buckets: Number of buckets to divide the relation occurrences into.
        The default is 14. Each bucket is a power of two starting from 1.
        e.g. (1, 2), (2, 4), (4, 8), ... ending with (8192, inf) for 14 buckets.
    :param path_to_rel_info_file: Path to relation info file.
    :param output_path: Path to save the diagram.
    :return:
    """
    out = output_path
    if output_path is None:
        out = os.path.dirname(path_to_rel_info_file)
    relation_info_dict: dict = load_json_dict(path_to_rel_info_file)

    occurrence_buckets = []
    for i in range(num_buckets):
        if i == num_buckets - 1:
            occurrence_buckets.append((2**i, float("inf")))
            break
        occurrence_buckets.append((2**i, 2 ** (i + 1)))
    relation_occurrence_info_dict = {}
    for occurrence in occurrence_buckets:
        relation_occurrence_info_dict[f"{occurrence[0]}-{occurrence[1]}"] = {
            "total_occurrence": 0,
        }
    relation_occurrence_info_dict["0"] = {"total_occurrence": 0}

    for relations in relation_info_dict.values():
        for fact in relations.values():
            occurrences = fact["occurrences"]
            if occurrences == 0:
                relation_occurrence_info_dict["0"]["total_occurrence"] += 1
                continue
            for bucket in occurrence_buckets:
                bucket_start = bucket[0]
                bucket_end = bucket[1]
                if bucket_start <= occurrences < bucket_end:
                    relation_occurrence_info_dict[f"{bucket_start}-{bucket_end}"]["total_occurrence"] += 1
                    break

    def get_num(x: str) -> int:
        number = x.split("-")[0]
        return int(number)

    x_labels = sorted(list(relation_occurrence_info_dict.keys()), key=get_num)
    occurrences = [relation_occurrence_info_dict[x_label]["total_occurrence"] for x_label in x_labels]
    plt.figure(figsize=(6, 5))
    plt.bar(x_labels, occurrences)
    for i, count in enumerate(occurrences):
        plt.text(i, count, str(count), ha="center", va="bottom")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Frequency Buckets")
    plt.ylabel("Frequency of Subj/Obj Pairs")
    plt.title("Entity Pair Frequency Histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"{output_diagram_name}.png"))
    plt.savefig(os.path.join(out, f"{output_diagram_name}.pdf"))
    plt.clf()
    plt.close()


def count_increasing_occurrences_in_slices(path_to_files: str) -> dict:
    """
    Count increasing occurrences in relation occurrences info files
    :param path_to_files: Path to relation info occurrences files.
    :return: Dictionary containing increasing occurrences in for each slice
    """
    files: list = os.listdir(path_to_files)
    files.sort()
    increasing_occurrences_in_slices = {}
    for file in tqdm(files, desc="Counting increasing occurrences in slices"):
        if file.endswith(".json"):
            for relation_id, entities in load_json_dict(f"{path_to_files}/{file}").items():
                for entity_id, fact in entities.items():
                    if relation_id not in increasing_occurrences_in_slices:
                        increasing_occurrences_in_slices[relation_id] = {}
                    if entity_id not in increasing_occurrences_in_slices[relation_id]:
                        increasing_occurrences_in_slices[relation_id][entity_id] = {
                            "occurrences_increase": [],
                            "obj_id": fact["obj_id"],
                        }
                    try:
                        total_old = increasing_occurrences_in_slices[relation_id][entity_id]["occurrences_increase"][
                            -1
                        ]["total"]
                        total = total_old + fact["occurrences"]
                    except IndexError:
                        total = fact["occurrences"]
                    increasing_occurrences_in_slices[relation_id][entity_id]["occurrences_increase"].append(
                        {"Slice": files.index(file), "occurrences": fact["occurrences"], "total": total}
                    )
    return increasing_occurrences_in_slices


def join_relation_occurrences_info_json_files(path_to_files: str) -> None:
    """
    Join relation occurrences info files
    :param path_to_files: Path to relation info files.
    :return:
    """
    files: list = os.listdir(path_to_files)
    files.sort()
    first_file = load_json_dict(f"{path_to_files}/{files[0]}")
    for file in tqdm(files[1:], desc="Joining relation info files"):
        if file.endswith(".json"):
            for relation_id, entities in load_json_dict(f"{path_to_files}/{file}").items():
                for entity_id, fact in entities.items():
                    first_file[relation_id][entity_id]["occurrences"] += fact["occurrences"]
    for _, entities in first_file.items():
        for _, fact in entities.items():
            fact.pop("sentences")
    save_dict_as_json(first_file, f"{path_to_files}/joined_relation_occurrence_info.json")
    logging.info("Joined relation info files.")


def split_relation_occurrences_info_json_on_occurrences(path_to_relation_info: str, **kwargs) -> dict:
    """
    Split relation occurrences info json file based on occurrences.

    :param path_to_relation_info: Path to relation info file.
    :param kwargs: Information for splitting the relation occurrences info file.
    - threshold: int = 100 (The threshold occurrence for splitting the relation occurrences info file)
    - total_num_samples: int = 1000 (Total number of samples to draw from the relation occurrences info file)
    - splits: list[tuples] = [(0.5, 0.5), (0.7, 0.3)] (The splits to divide the total number of samples)
    :return: Dictionary containing a dictionary with the split list (list of tuples) and threshold for each split.
    """
    occurrences = load_json_dict(path_to_relation_info)
    split_a_list = []
    split_b_list = []
    split_threshold = kwargs.get("threshold")
    total_num_samples = kwargs.get("total_num_samples")

    for relation_id, entity_dict in occurrences.items():
        for entity, fact_info in entity_dict.items():
            if fact_info["occurrences"] < split_threshold:
                split_a_list.append((relation_id, entity))
            else:
                split_b_list.append((relation_id, entity))
    split_dict = {}
    for split in kwargs.get("splits"):
        split_a_percent, split_b_percent = split

        num_a_samples = round(total_num_samples * split_a_percent)
        num_b_samples = round(total_num_samples * split_b_percent)
        try:
            split_a = random.sample(split_a_list, num_a_samples)
            split_b = random.sample(split_b_list, num_b_samples)
        except ValueError:
            logging.error(
                "Total number of samples %s is less than the sum of the number of samples for splits.",
                total_num_samples,
            )
            continue

        split_dict[split] = {"list": split_a + split_b, "threshold": split_threshold}

    return split_dict
