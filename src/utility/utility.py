import json
import logging
import os
import re
from typing import Union

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


def save_json_dict(json_dict: dict, json_file_path: str):
    """
    Save json file.
    :param json_dict: Dictionary containing information
    :param json_file_path: Path to json file
    """
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=4, ensure_ascii=False, cls=SetEncoder)


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


def decorate_sentence_with_ids(sentence: str, linked_entities) -> str:
    """
    Decorate entities with IDs.
    :param sentence: List of entities
    :param linked_entities: List of linked entities
    :return: Dictionary containing entities with IDs
    """
    entity_ids = [f"Q{str(linked_entity.get_id())}" for linked_entity in linked_entities]
    return f"{sentence} [{' '.join(entity_ids)}]"


def word_in_sentence(word: str, sentence: str) -> bool:
    """
    Check if word is in sentence.

    :param word: word to check
    :param sentence: sentence to check
    :return: True if word is in sentence, False otherwise
    """
    pattern = re.compile(r"(?<!\w)({0})(?!\w)".format(re.escape(word)), flags=re.IGNORECASE)
    return bool(pattern.search(sentence))


def extract_entity_information(bear_data_path: str, bear_relation_info_path: str) -> dict:
    """
    Extract entity information from bear data.
    :param bear_data_path: Path to bear facts directory.
    :param bear_relation_info_path: Path to the BEAR relation info file.
    :return: Relation dictionary
    """
    relation_dict: dict = {}
    bear_relation_info_dict: dict = load_json_dict(bear_relation_info_path)
    for relation_key, _ in bear_relation_info_dict.items():
        try:
            fact_list: list[dict] = load_json_line_dict(f"{bear_data_path}/{relation_key}.jsonl")
            relation_dict.update({relation_key: {}})
        except FileNotFoundError:
            logging.error("File not found: %s/%s.jsonl", bear_data_path, relation_key)
            continue
        for fact_dict in fact_list:
            logging.info("Extracting entity information for %s", relation_key)
            relation_dict[relation_key][fact_dict["sub_id"]] = {
                "subj_label": fact_dict["sub_label"],
                "subj_aliases": set(fact_dict["sub_aliases"]),
                "obj_id": fact_dict["obj_id"],
                "obj_label": fact_dict["obj_label"],
                "obj_aliases": set(),
                "occurrences": 0,
                "sentences": set(),
            }
    for _, relations in relation_dict.items():
        for _, fact in relations.items():
            for _, relations_ in relation_dict.items():
                try:
                    fact["obj_aliases"].update(relations_[fact["obj_id"]]["subj_aliases"])
                except KeyError:
                    continue
    return relation_dict


def get_tokens_from_sentence(sentence: str, tokenizer) -> list[str]:
    """
    Get tokens from sentence.
    :param sentence: Sentence
    :param tokenizer: Tokenizer
    :return: List of tokens
    """
    return [token.orth_ for token in tokenizer(sentence.lower())]


def create_fact_occurrence_histogram(
    path_to_rel_info_file: str, output_diagram_name: str = "occurrence_statistics"
) -> None:
    """
    Create fact occurrence statistics and plot a histogram.
    :param output_diagram_name: Name of the output diagram.
    :param path_to_rel_info_file: Path to relation info file.
    :return:
    """
    relation_info_dict: dict = load_json_dict(path_to_rel_info_file)
    occurrence_buckets = [0, 0, 0, 0, 0, 0]

    for relations in relation_info_dict.values():
        for fact in relations.values():
            occurrences = fact["occurrences"]
            if occurrences == 0:
                occurrence_buckets[0] += 1
            elif 1 <= occurrences <= 99:
                occurrence_buckets[1] += 1
            elif 100 <= occurrences <= 299:
                occurrence_buckets[2] += 1
            elif 300 <= occurrences <= 499:
                occurrence_buckets[3] += 1
            elif 500 <= occurrences <= 1000:
                occurrence_buckets[4] += 1
            else:
                occurrence_buckets[5] += 1

    x_labels = ["0", "1-99", "100-299", "300-499", "500-1000", ">1000"]
    plt.bar(x_labels, occurrence_buckets)
    for i, count in enumerate(occurrence_buckets):
        plt.text(i, count, str(count), ha="center", va="bottom")
    plt.xlabel("Occurrences")
    plt.ylabel("Number of Entities")
    plt.title("Entity Occurrence Histogram")
    plt.savefig(os.path.join(os.path.dirname(path_to_rel_info_file), f"{output_diagram_name}.png"))


def join_relation_info_json_files(
    path_to_files: str,
    correct_possible_duplicates: Union[str, bool] = False,
    remove_sentences: Union[str, bool] = False,
) -> None:
    """
    Join relation info files
    :param path_to_files: Path to relation info files.
    This is useful when final-joined json is too large to store in memory.
    :param correct_possible_duplicates: Correct possible duplicates in relation info files.
    :param remove_sentences: Remove sentences from relation info files.
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
    save_json_dict(first_file, f"{path_to_files}/joined_relation_info.json")
    logging.info("Joined relation info files.")
