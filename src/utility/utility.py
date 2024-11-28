import json
import logging
import os
import re
from typing import Union, Optional

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


def word_in_sentence(word: str, sentence: str) -> bool:
    """
    Check if word is in sentence.

    :param word: word to check
    :param sentence: sentence to check
    :return: True if word is in sentence, False otherwise
    """
    pattern = re.compile(r"(?<!\w)({0})(?!\w)".format(re.escape(word)), flags=re.IGNORECASE)
    return bool(pattern.search(sentence))


def create_fact_occurrence_histogram(
    path_to_rel_info_file: str,
    output_diagram_name: str = "occurrence_statistics",
    output_path: Optional[str] = None,
) -> None:
    """
    Create fact occurrence statistics and plot a histogram.

    :param output_diagram_name: Name of the output diagram.
    :param path_to_rel_info_file: Path to relation info file.
    :param output_path: Path to save the diagram.
    :return:
    """
    out = output_path
    if output_path is None:
        out = os.path.dirname(path_to_rel_info_file)
    relation_info_dict: dict = load_json_dict(path_to_rel_info_file)
    occurrence_buckets = [
        (1, 99),
        (100, 299),
        (300, 499),
        (500, 699),
        (700, 899),
        (900, 999),
        (1000, float("inf")),
    ]
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
                if bucket[0] <= occurrences <= bucket[1]:
                    relation_occurrence_info_dict[f"{bucket[0]}-{bucket[1]}"]["total_occurrence"] += 1
                    break

    x_labels = relation_occurrence_info_dict.keys()
    occurrences = [relation_occurrence_info_dict[x_label]["total_occurrence"] for x_label in x_labels]
    plt.bar(x_labels, occurrences)
    for i, count in enumerate(occurrences):
        plt.text(i, count, str(count), ha="center", va="bottom")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Occurrence Buckets")
    plt.ylabel("Number of Subj/Obj Pairs")
    plt.title("Entity Pair Occurrence Histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"{output_diagram_name}.png"))


def join_relation_info_json_files(
    path_to_files: str,
    correct_possible_duplicates: Union[str, bool] = False,
    remove_sentences: Union[str, bool] = False,
) -> None:
    """
    Join relation info files
    :param path_to_files: Path to relation info files.
    This is useful when final-joined json is too large to store in memory.
    :param correct_possible_duplicates: Correct possible duplicates in the relation info files.
     Requires the sentences to be present in the relation info files.
    :param remove_sentences: Remove sentences from the joined relation info file.
    :return:
    """
    if isinstance(remove_sentences, str):
        remove_sentences = remove_sentences.lower() == "true"
    if isinstance(correct_possible_duplicates, str):
        correct_possible_duplicates = correct_possible_duplicates.lower() == "true"
    files: list = os.listdir(path_to_files)
    files.sort()
    first_file = load_json_dict(f"{path_to_files}/{files[0]}")
    for file in tqdm(files[1:], desc="Joining relation info files"):
        if file.endswith(".json"):
            for relation_id, entities in load_json_dict(f"{path_to_files}/{file}").items():
                for entity_id, fact in entities.items():
                    first_file[relation_id][entity_id]["occurrences"] += fact["occurrences"]
                    sentences = set(first_file[relation_id][entity_id]["sentences"] + fact["sentences"])
                    first_file[relation_id][entity_id]["sentences"] = list(sentences)
                    if (
                        0
                        < len(first_file[relation_id][entity_id]["sentences"])
                        < first_file[relation_id][entity_id]["occurrences"]
                    ) and correct_possible_duplicates:
                        first_file[relation_id][entity_id]["occurrences"] = len(
                            first_file[relation_id][entity_id]["sentences"]
                        )
                        logging.warning(
                            "Mismatch in occurrences and sentences for %s (%s), may contain duplicate occurrences. Correcting occurrences set to (%s)!",
                            entity_id,
                            relation_id,
                            correct_possible_duplicates,
                        )
    if remove_sentences:
        for _, entities in first_file.items():
            for _, fact in entities.items():
                fact.pop("sentences")
    save_dict_as_json(first_file, f"{path_to_files}/joined_relation_occurrence_info.json")
    logging.info("Joined relation info files.")
