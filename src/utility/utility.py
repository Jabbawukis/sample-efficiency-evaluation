import json


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
    for linked_entity in linked_entities:
        identifier = linked_entity.get_id()
        label = linked_entity.get_label()
        sentence = sentence.replace(label, f"({label}) [Q{identifier}]")
    return sentence
