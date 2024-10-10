"""
FactMatcher
"""

import logging

from tqdm import tqdm
from utility import utility


class FactMatcher:
    """
    FactMatcher
    """

    def __init__(self, **kwargs):
        self.bear_relation_info_dict: dict = utility.load_json_dict(kwargs.get("bear_relation_info_path"))
        self.entity_relation_info_dict: dict = self.extract_entity_information(kwargs.get("bear_data_path"))

    def extract_entity_information(self, bear_data_path: str) -> dict:
        """
        Extract entity information from bear data.
        :return:
        """
        relation_dict: dict = {}
        for relation_key, _ in self.bear_relation_info_dict.items():
            try:
                fact_list: list[str] = utility.load_json_line_dict(f"{bear_data_path}/{relation_key}.jsonl")
                relation_dict.update({relation_key: {}})
            except FileNotFoundError:
                logging.error("File not found: %s/%s.jsonl", bear_data_path, relation_key)
                continue
            for fact in tqdm(fact_list, desc=f"Extracting entity information for {relation_key}"):
                fact_dict = utility.load_json_str(fact)
                relation_dict[relation_key][fact_dict["sub_label"]] = {
                    "aliases": fact_dict["sub_aliases"],
                    "obj_label": fact_dict["obj_label"],
                }
        return relation_dict
