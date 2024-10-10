import json


def load_json_dict(json_file_path: str) -> dict:
    """
    Load json file.
    :param json_file_path: Path to json file
    :return: Dictionary containing information
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_dict = json.load(f)
    return json_dict


def load_json_line_dict(json_line_file_path: str) -> list:
    """
    Load json line file.
    :param json_line_file_path: Path to json file
    :return: Dictionary containing information
    """
    with open(json_line_file_path, "r", encoding="utf-8") as f:
        json_list = list(f)
    return json_list


def load_json_str(json_str: str) -> dict:
    """
    Load json string.
    :param json_str: Json string
    :return: Dictionary containing information
    """
    json_dict = json.loads(json_str)
    return json_dict
