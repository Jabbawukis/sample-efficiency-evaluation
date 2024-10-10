import os
import unittest
from unittest.mock import patch

from sample_efficiency_evaluation.fact_matcher import FactMatcherSimpleHeuristic
from utility import utility


class FactMatcherTest(unittest.TestCase):

    def setUp(self) -> None:
        self.test_relation_info_dict = {
            "P6": {"domains": ["Political", "Biographical", "Historical"]},
            "P19": {"domains": ["Biographical"]},
        }
        self.test_entity_relation_info_dict = {
            "P6": {
                "Abu Dhabi": {
                    "aliases": ["Abū Dhabi", "Abudhabi"],
                    "obj_label": "Khalifa bin Zayed Al Nahyan",
                },
                "Armenia": {
                    "aliases": ["Republic of Armenia", "🇦🇲", "ARM", "AM"],
                    "obj_label": "Nikol Pashinyan",
                },
                "Free State of Fiume": {
                    "aliases": [],
                    "obj_label": "Gabriele D'Annunzio",
                },
                "Nepal": {
                    "aliases": [
                        "NPL",
                        "Federal Democratic Republic of Nepal",
                        "NEP",
                        "NP",
                        "🇳🇵",
                    ],
                    "obj_label": "Khadga Prasad Sharma Oli",
                },
            }
        }
        self.maxDiff = None
        self.test_resources_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_resources"))

    def test_extract_entity_information(self):
        with patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict) as mock_load_json_dict:

            fact_matcher = FactMatcherSimpleHeuristic(
                bear_relation_info_path=f"{self.test_resources_abs_path}/relation_info.json",
                bear_data_path=f"{self.test_resources_abs_path}/BEAR",
            )

            self.assertEqual(fact_matcher.bear_relation_info_dict, self.test_relation_info_dict)
            self.assertEqual(fact_matcher.entity_relation_info_dict, self.test_entity_relation_info_dict)
            mock_load_json_dict.assert_called_once_with(f"{self.test_resources_abs_path}/relation_info.json")
