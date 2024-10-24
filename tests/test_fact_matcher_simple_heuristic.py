import json
import logging
import os
import unittest
import shutil
from unittest.mock import patch, MagicMock

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
                "Q1519": {
                    "subj_label": "Abu Dhabi",
                    "subj_aliases": {"Abū Dhabi", "Abudhabi"},
                    "obj_id": "Q1059948",
                    "obj_label": "Khalifa bin Zayed Al Nahyan",
                    "obj_aliases": set(),
                    "occurrences": 0,
                },
                "Q399": {
                    "subj_label": "Armenia",
                    "subj_aliases": {"Republic of Armenia", "🇦🇲", "ARM", "AM"},
                    "obj_id": "Q7035479",
                    "obj_label": "Nikol Pashinyan",
                    "obj_aliases": set(),
                    "occurrences": 0,
                },
                "Q548114": {
                    "subj_label": "Free State of Fiume",
                    "subj_aliases": set(),
                    "obj_id": "Q193236",
                    "obj_label": "Gabriele D'Annunzio",
                    "obj_aliases": set(),
                    "occurrences": 0},
                "Q5626824": {
                    "subj_label": "Gülcemal Sultan",
                    "subj_aliases": set(),
                    "obj_id": "Q222",
                    "obj_label": "Albania",
                    "obj_aliases": set(),
                    "occurrences": 0},
                "Q837": {
                    "subj_label": "Nepal",
                    "subj_aliases": {"NPL", "Federal Democratic Republic of Nepal", "NEP", "NP", "🇳🇵"},
                    "obj_id": "Q3195923",
                    "obj_label": "Khadga Prasad Sharma Oli",
                    "obj_aliases": set(),
                    "occurrences": 0,
                },
            }
        }
        self.test_relation_info_dict_obj_aliases = {
            "P_00": {"domains": ["stuff"]},
            "P_01": {"domains": ["hi"]},
        }
        self.test_entity_relation_info_dict_filled_obj_aliases = {
            "P_00": {
                "Q30": {
                    "subj_label": "United States of America",
                    "subj_aliases": {"the United States of America","America","U.S.A.","USA","U.S.","US"},
                    "obj_id": "Q61",
                    "obj_label": "Washington, D.C.",
                    "obj_aliases": set(),
                    "occurrences": 0,
                },
                "Q178903": {
                    "subj_label": "Alexander Hamilton",
                    "subj_aliases": {"Publius","Hamilton","Alexander Hamilton, US Treasury secretary","A. Ham"},
                    "obj_id": "Q30",
                    "obj_label": "United States of America",
                    "obj_aliases": {"the United States of America","America","U.S.A.","USA","U.S.","US"},
                    "occurrences": 0,
                }
            },
            "P_01": {
                "Q2127993": {
                    "subj_label": "Rainer Bernhardt",
                    "subj_aliases": {"Rainer Herbert Georg Bernhardt"},
                    "obj_id": "Q30",
                    "obj_label": "United States of America",
                    "obj_aliases": {"the United States of America","America","U.S.A.","USA","U.S.","US"},
                    "occurrences": 0,
                }
            }
        }
        self.maxDiff = None
        self.test_resources_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_resources"))
        self.indexer_mocked = MagicMock()
        self.writer_mocked = MagicMock()
        self.test_index_dir = f"{self.test_resources_abs_path}/test_index_dir"
        if os.path.exists(self.test_index_dir):
            shutil.rmtree(self.test_index_dir)

    def test_extract_entity_information_good(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict) as mock_load_json_dict,
            patch.object(logging, "error") as mock_error,
            patch.object(
                FactMatcherSimpleHeuristic, "_initialize_index", return_value=(self.writer_mocked, self.indexer_mocked)
            ),
        ):

            fact_matcher = FactMatcherSimpleHeuristic(
                bear_relation_info_path=f"{self.test_resources_abs_path}/relation_info.json",
                bear_facts_path=f"{self.test_resources_abs_path}/BEAR",
            )

            self.assertEqual(fact_matcher.entity_relation_info_dict, self.test_entity_relation_info_dict)
            mock_error.assert_called_once()
            mock_load_json_dict.assert_called_once_with(f"{self.test_resources_abs_path}/relation_info.json")

    def test_extract_entity_information_good2(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict) as mock_load_json_dict,
            patch.object(logging, "error") as mock_error,
            patch.object(
                FactMatcherSimpleHeuristic, "_initialize_index", return_value=(self.writer_mocked, self.indexer_mocked)
            ),
        ):

            fact_matcher = FactMatcherSimpleHeuristic(bear_data_path=f"{self.test_resources_abs_path}")

            self.assertEqual(fact_matcher.entity_relation_info_dict, self.test_entity_relation_info_dict)
            mock_error.assert_called_once()
            mock_load_json_dict.assert_called_once_with(f"{self.test_resources_abs_path}/relation_info.json")

    def test_extract_entity_information_good_filled_obj_aliases(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict_obj_aliases) as mock_load_json_dict,
            patch.object(
                FactMatcherSimpleHeuristic, "_initialize_index", return_value=(self.writer_mocked, self.indexer_mocked)
            ),
        ):

            fact_matcher = FactMatcherSimpleHeuristic(bear_data_path=f"{self.test_resources_abs_path}")

            self.assertEqual(fact_matcher.entity_relation_info_dict, self.test_entity_relation_info_dict_filled_obj_aliases)
            mock_load_json_dict.assert_called_once_with(f"{self.test_resources_abs_path}/relation_info.json")

    def test_convert_relation_info_dict_to_json(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict_obj_aliases),
            patch.object(
                FactMatcherSimpleHeuristic, "_initialize_index", return_value=(self.writer_mocked, self.indexer_mocked)
            ),
            patch.object(
                FactMatcherSimpleHeuristic,
                "_extract_entity_information",
                return_value=self.test_entity_relation_info_dict_filled_obj_aliases,
            ),
        ):
            fact_matcher = FactMatcherSimpleHeuristic(bear_data_path=f"{self.test_resources_abs_path}")
            fact_matcher.convert_relation_info_dict_to_json(f"{self.test_resources_abs_path}/test_relation_info.json")

    def test_index_dataset_1(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict),
            patch.object(
                FactMatcherSimpleHeuristic, "_initialize_index", return_value=(self.writer_mocked, self.indexer_mocked)
            ),
            patch.object(
                FactMatcherSimpleHeuristic,
                "_extract_entity_information",
                return_value=self.test_entity_relation_info_dict,
            ),
            patch.object(
                FactMatcherSimpleHeuristic,
                "index_file",
            ) as mock_index_file,
            patch.object(
                FactMatcherSimpleHeuristic,
                "close",
            ),
        ):

            fact_matcher = FactMatcherSimpleHeuristic(
                bear_data_path=f"{self.test_resources_abs_path}",
                file_index_dir=self.test_index_dir,
            )

            fact_matcher.index_dataset(
                [
                    {"text": "Boeing is a company. Boeing 747 is a plane."},
                    {"text": "Boeing 747 is a plane"},
                ],
                text_key="text",
                split_contents_into_sentences=False,
            )

            mock_index_file.assert_any_call("Boeing is a company. Boeing 747 is a plane.")
            mock_index_file.assert_any_call("Boeing 747 is a plane")
            self.assertEqual(mock_index_file.call_count, 2)

    def test_index_dataset_2(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict),
            patch.object(
                FactMatcherSimpleHeuristic,
                "_extract_entity_information",
                return_value=self.test_entity_relation_info_dict,
            ),
            patch.object(
                FactMatcherSimpleHeuristic, "_initialize_index", return_value=(self.writer_mocked, self.indexer_mocked)
            ),
            patch.object(
                FactMatcherSimpleHeuristic,
                "index_file",
            ) as mock_index_file,
            patch.object(
                FactMatcherSimpleHeuristic,
                "close",
            ),
        ):

            fact_matcher = FactMatcherSimpleHeuristic(
                bear_data_path=f"{self.test_resources_abs_path}",
                file_index_dir=self.test_index_dir,
            )

            fact_matcher.index_dataset(
                [
                    {"text": "Boeing is a company. Boeing 747 is a plane."},
                    {"text": "Boeing 747 is a cool plane."},
                ],
                text_key="text",
                split_contents_into_sentences=True,
            )

            mock_index_file.assert_any_call("Boeing is a company.")
            mock_index_file.assert_any_call("Boeing 747 is a plane.")
            mock_index_file.assert_any_call("Boeing 747 is a cool plane.")
            self.assertEqual(mock_index_file.call_count, 3)

    def test_search_index(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict),
            patch.object(
                FactMatcherSimpleHeuristic,
                "_extract_entity_information",
                return_value=self.test_entity_relation_info_dict,
            ),
        ):

            fact_matcher = FactMatcherSimpleHeuristic(
                bear_data_path=f"{self.test_resources_abs_path}",
                file_index_dir=self.test_index_dir,
            )

            fact_matcher.index_file("Boeing is a company")
            fact_matcher.index_file("Boeing 747 is a plane")
            results = fact_matcher.search_index("Boeing")
            fact_matcher.close()
            self.assertEqual(len(results), 2)
            self.assertEqual(
                results,
                [
                    {
                        "path": "/ddda5959a6a4f994ee6a55c0e60b6137ea776e79846fc5a35d58ef0840005905",
                        "title": "ddda5959a6a4f994ee6a55c0e60b6137ea776e79846fc5a35d58ef0840005905",
                        "text": "Boeing is a company",
                    },
                    {
                        "path": "/1b4c34a604c95618ceb558da613bd8655d0a6a21ccaf0480dc150eff44d30047",
                        "title": "1b4c34a604c95618ceb558da613bd8655d0a6a21ccaf0480dc150eff44d30047",
                        "text": "Boeing 747 is a plane",
                    },
                ],
            )

    def test_search_index_sub_query_1(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict),
            patch.object(
                FactMatcherSimpleHeuristic,
                "_extract_entity_information",
                return_value=self.test_entity_relation_info_dict,
            ),
        ):

            fact_matcher = FactMatcherSimpleHeuristic(
                bear_data_path=f"{self.test_resources_abs_path}",
                file_index_dir=self.test_index_dir,
            )

            fact_matcher.index_file("Boeing is a company")
            fact_matcher.index_file("Boeing 747 is a plane")
            results = fact_matcher.search_index("Boeing", sub_query="747")
            fact_matcher.close()
            self.assertEqual(len(results), 1)
            self.assertEqual(
                results,
                [
                    {
                        "path": "/1b4c34a604c95618ceb558da613bd8655d0a6a21ccaf0480dc150eff44d30047",
                        "title": "1b4c34a604c95618ceb558da613bd8655d0a6a21ccaf0480dc150eff44d30047",
                        "text": "Boeing 747 is a plane",
                    }
                ],
            )

    def test_search_index_sub_query_2(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict),
            patch.object(
                FactMatcherSimpleHeuristic,
                "_extract_entity_information",
                return_value=self.test_entity_relation_info_dict,
            ),
        ):

            fact_matcher = FactMatcherSimpleHeuristic(
                bear_data_path=f"{self.test_resources_abs_path}",
                file_index_dir=self.test_index_dir,
            )

            fact_matcher.index_file("Angela Merkel is the chancellor of Germany")
            fact_matcher.index_file("Boeing 747 is a plane")
            results = fact_matcher.search_index("Angela Merkel", sub_query="chancellor Germany")
            fact_matcher.close()
            self.assertEqual(len(results), 1)
            self.assertEqual(
                results,
                [
                    {
                        "path": "/4640d50a3d19c3d61223ee8bee3f4615164524b78bbf06bb2f7c70a6e4ccc6d4",
                        "title": "4640d50a3d19c3d61223ee8bee3f4615164524b78bbf06bb2f7c70a6e4ccc6d4",
                        "text": "Angela Merkel is the chancellor of Germany",
                    }
                ],
            )

    def test_create_fact_statistics_good(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict),
            patch.object(
                FactMatcherSimpleHeuristic,
                "_extract_entity_information",
                return_value=self.test_entity_relation_info_dict,
            ),
        ):

            fact_matcher = FactMatcherSimpleHeuristic(
                bear_data_path=f"{self.test_resources_abs_path}",
                file_index_dir=self.test_index_dir,
            )
            fact_matcher.index_file("Abu Dhabi blah blah blah Khalifa bin Zayed Al Nahyan")
            fact_matcher.index_file("Abudhabi blah blah blah Khalifa bin Zayed Al Nahyan")
            fact_matcher.index_file("Armenia blah blah blah Nikol Pashinyan")
            fact_matcher.index_file("Free State of Fiume blah ducks blah Nikol Pashinyan Gabriele D'Annunzio")
            fact_matcher.index_file("Nepal NPL is cool Khadga Prasad Sharma Oli")
            fact_matcher.create_fact_statistics()
            self.assertEqual(fact_matcher.entity_relation_info_dict, {
            "P6": {
                "Q1519": {
                    "subj_label": "Abu Dhabi",
                    "subj_aliases": {"Abū Dhabi", "Abudhabi"},
                    "obj_id": "Q1059948",
                    "obj_label": "Khalifa bin Zayed Al Nahyan",
                    "obj_aliases": set(),
                    "occurrences": 2,
                },
                "Q399": {
                    "subj_label": "Armenia",
                    "subj_aliases": {"Republic of Armenia", "🇦🇲", "ARM", "AM"},
                    "obj_id": "Q7035479",
                    "obj_label": "Nikol Pashinyan",
                    "obj_aliases": set(),
                    "occurrences": 1,
                },
                "Q548114": {
                    "subj_label": "Free State of Fiume",
                    "subj_aliases": set(),
                    "obj_id": "Q193236",
                    "obj_label": "Gabriele D'Annunzio",
                    "obj_aliases": set(),
                    "occurrences": 1},
                "Q5626824": {
                    "subj_label": "Gülcemal Sultan",
                    "subj_aliases": set(),
                    "obj_id": "Q222",
                    "obj_label": "Albania",
                    "obj_aliases": set(),
                    "occurrences": 0},
                "Q837": {
                    "subj_label": "Nepal",
                    "subj_aliases": {"NPL", "Federal Democratic Republic of Nepal", "NEP", "NP", "🇳🇵"},
                    "obj_id": "Q3195923",
                    "obj_label": "Khadga Prasad Sharma Oli",
                    "obj_aliases": set(),
                    "occurrences": 1,
                },
            }
        })

    def test_create_fact_statistics_good2(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict_obj_aliases),
            patch.object(
                FactMatcherSimpleHeuristic,
                "_extract_entity_information",
                return_value=self.test_entity_relation_info_dict_filled_obj_aliases,
            ),
        ):

            fact_matcher = FactMatcherSimpleHeuristic(
                bear_data_path=f"{self.test_resources_abs_path}",
                file_index_dir=self.test_index_dir,
            )
            fact_matcher.index_file("United States of America blah blah blah Washington, D.C.")
            fact_matcher.index_file("United States of America (U.S.A.) blah blah blah Washington, D.C.")
            fact_matcher.index_file("United of America (U.S.A.) blah blah blah Washington, D.C.")
            fact_matcher.index_file("Alexander Hamilton blah blah blah the United States of America")
            fact_matcher.index_file("Publius blah blah blah the USA based in Washington, D.C.")
            fact_matcher.index_file("Hamilton blah blah blah United States of America")
            fact_matcher.index_file("US blah blah blah A. Ham")
            fact_matcher.create_fact_statistics()
            self.assertEqual(fact_matcher.entity_relation_info_dict, {
            "P_00": {
                "Q30": {
                    "subj_label": "United States of America",
                    "subj_aliases": {"the United States of America","America","U.S.A.","USA","U.S.","US"},
                    "obj_id": "Q61",
                    "obj_label": "Washington, D.C.",
                    "obj_aliases": set(),
                    "occurrences": 4,
                },
                "Q178903": {
                    "subj_label": "Alexander Hamilton",
                    "subj_aliases": {"Publius","Hamilton","Alexander Hamilton, US Treasury secretary","A. Ham"},
                    "obj_id": "Q30",
                    "obj_label": "United States of America",
                    "obj_aliases": {"the United States of America","America","U.S.A.","USA","U.S.","US"},
                    "occurrences": 3,
                }
            },
            "P_01": {
                "Q2127993": {
                    "subj_label": "Rainer Bernhardt",
                    "subj_aliases": {"Rainer Herbert Georg Bernhardt"},
                    "obj_id": "Q30",
                    "obj_label": "United States of America",
                    "obj_aliases": {"the United States of America","America","U.S.A.","USA","U.S.","US"},
                    "occurrences": 0,
                }
            }
        })