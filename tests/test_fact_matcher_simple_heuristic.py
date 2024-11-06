import logging
import os
import unittest
import shutil
from unittest.mock import patch, MagicMock

from sample_efficiency_evaluation import FactMatcherSimple
from utility import utility


class FactMatcherTestSimpleHeuristic(unittest.TestCase):

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
                    "sentences": set(),
                },
                "Q399": {
                    "subj_label": "Armenia",
                    "subj_aliases": {"Republic of Armenia", "🇦🇲", "ARM", "AM"},
                    "obj_id": "Q7035479",
                    "obj_label": "Nikol Pashinyan",
                    "obj_aliases": set(),
                    "occurrences": 0,
                    "sentences": set(),
                },
                "Q548114": {
                    "subj_label": "Free State of Fiume",
                    "subj_aliases": set(),
                    "obj_id": "Q193236",
                    "obj_label": "Gabriele D'Annunzio",
                    "obj_aliases": set(),
                    "occurrences": 0,
                    "sentences": set(),
                },
                "Q5626824": {
                    "subj_label": "Gülcemal Sultan",
                    "subj_aliases": set(),
                    "obj_id": "Q222",
                    "obj_label": "Albania",
                    "obj_aliases": set(),
                    "occurrences": 0,
                    "sentences": set(),
                },
                "Q837": {
                    "subj_label": "Nepal",
                    "subj_aliases": {"NPL", "Federal Democratic Republic of Nepal", "NEP", "NP", "🇳🇵"},
                    "obj_id": "Q3195923",
                    "obj_label": "Khadga Prasad Sharma Oli",
                    "obj_aliases": set(),
                    "occurrences": 0,
                    "sentences": set(),
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
                    "sentences": set(),
                },
                "Q178903": {
                    "subj_label": "Alexander Hamilton",
                    "subj_aliases": {"Publius","Hamilton","Alexander Hamilton, US Treasury secretary","A. Ham"},
                    "obj_id": "Q30",
                    "obj_label": "United States of America",
                    "obj_aliases": {"the United States of America","America","U.S.A.","USA","U.S.","US"},
                    "occurrences": 0,
                    "sentences": set(),
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
                    "sentences": set(),
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
                FactMatcherSimple, "_initialize_index", return_value=(self.writer_mocked, self.indexer_mocked)
            ),
        ):

            fact_matcher = FactMatcherSimple(
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
                FactMatcherSimple, "_initialize_index", return_value=(self.writer_mocked, self.indexer_mocked)
            ),
        ):

            fact_matcher = FactMatcherSimple(bear_data_path=f"{self.test_resources_abs_path}")

            self.assertEqual(fact_matcher.entity_relation_info_dict, self.test_entity_relation_info_dict)
            mock_error.assert_called_once()
            mock_load_json_dict.assert_called_once_with(f"{self.test_resources_abs_path}/relation_info.json")

    def test_extract_entity_information_good_filled_obj_aliases(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict_obj_aliases) as mock_load_json_dict,
            patch.object(
                FactMatcherSimple, "_initialize_index", return_value=(self.writer_mocked, self.indexer_mocked)
            ),
        ):

            fact_matcher = FactMatcherSimple(bear_data_path=f"{self.test_resources_abs_path}")

            self.assertEqual(fact_matcher.entity_relation_info_dict, self.test_entity_relation_info_dict_filled_obj_aliases)
            mock_load_json_dict.assert_called_once_with(f"{self.test_resources_abs_path}/relation_info.json")

    def test_convert_relation_info_dict_to_json(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict_obj_aliases),
            patch.object(
                FactMatcherSimple, "_initialize_index", return_value=(self.writer_mocked, self.indexer_mocked)
            ),
            patch.object(
                FactMatcherSimple,
                "_extract_entity_information",
                return_value=self.test_entity_relation_info_dict_filled_obj_aliases,
            ),
        ):
            fact_matcher = FactMatcherSimple(bear_data_path=f"{self.test_resources_abs_path}")
            fact_matcher.convert_relation_info_dict_to_json(f"{self.test_resources_abs_path}/test_relation_info.json")

    def test_index_dataset_1(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict),
            patch.object(
                FactMatcherSimple, "_initialize_index", return_value=(self.writer_mocked, self.indexer_mocked)
            ),
            patch.object(
                FactMatcherSimple,
                "_extract_entity_information",
                return_value=self.test_entity_relation_info_dict,
            ),
            patch.object(
                FactMatcherSimple,
                "_index_file",
            ) as mock_index_file,
        ):

            fact_matcher = FactMatcherSimple(
                bear_data_path=f"{self.test_resources_abs_path}",
                file_index_dir=self.test_index_dir,
            )

            fact_matcher.index_dataset(
                [
                    {"text": "Boeing is a company. Boeing 747 is a plane."},
                    {"text": "Boeing 747 is a plane"},
                ],
                text_key="text",
                
            )

            mock_index_file.assert_any_call("Boeing is a company. Boeing 747 is a plane.")
            mock_index_file.assert_any_call("Boeing 747 is a plane")
            self.assertEqual(mock_index_file.call_count, 2)

    def test_index_dataset_2(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict),
            patch.object(
                FactMatcherSimple,
                "_extract_entity_information",
                return_value=self.test_entity_relation_info_dict,
            ),
            patch.object(
                FactMatcherSimple, "_initialize_index", return_value=(self.writer_mocked, self.indexer_mocked)
            ),
            patch.object(
                FactMatcherSimple,
                "_index_file",
            ) as mock_index_file,
        ):

            fact_matcher = FactMatcherSimple(
                bear_data_path=f"{self.test_resources_abs_path}",
                file_index_dir=self.test_index_dir,
            )

            fact_matcher.index_dataset(
                [
                    {"text": "Boeing is a company. Boeing 747 is a plane."},
                    {"text": "Boeing 747 is a cool plane."},
                ],
                text_key="text",
                
            )

            mock_index_file.assert_any_call("Boeing is a company. Boeing 747 is a plane.")
            mock_index_file.assert_any_call("Boeing 747 is a cool plane.")
            self.assertEqual(mock_index_file.call_count, 2)

    def test_search_index(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict),
            patch.object(
                FactMatcherSimple,
                "_extract_entity_information",
                return_value=self.test_entity_relation_info_dict,
            ),
        ):

            fact_matcher = FactMatcherSimple(
                bear_data_path=f"{self.test_resources_abs_path}",
                file_index_dir=self.test_index_dir,
            )

            fact_matcher.index_dataset(
                [{"text": "Boeing is a company. Boeing 747 is a plane."}],
                text_key="text",
                
            )
            results = fact_matcher.search_index("Boeing")
            fact_matcher.close()
            self.assertEqual(len(results), 1)
            self.assertEqual(
                results,
                [{'path': '/bca71e8376fccc36d8a182990017664c59740de27860c6ae67829670bcb690df',
                  'text': 'Boeing is a company. Boeing 747 is a plane.',
                  'title': 'bca71e8376fccc36d8a182990017664c59740de27860c6ae67829670bcb690df'}]
            )

    def test_search_index_sub_query_1(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict),
            patch.object(
                FactMatcherSimple,
                "_extract_entity_information",
                return_value=self.test_entity_relation_info_dict,
            ),
        ):

            fact_matcher = FactMatcherSimple(
                bear_data_path=f"{self.test_resources_abs_path}",
                file_index_dir=self.test_index_dir,
            )

            fact_matcher.index_dataset(
                [{"text": "Boeing 747 is a plane."}],
                text_key="text",
                
            )


            results = fact_matcher.search_index("Boeing", sub_query="747")
            fact_matcher.close()
            self.assertEqual(len(results), 1)
            self.assertEqual(
                results,
                [
                    {
                        "path": "/c0f690980267574d47032f7a21259e83bdd29b3fb5a5c0dd48da21c40a3b3a10",
                        "title": "c0f690980267574d47032f7a21259e83bdd29b3fb5a5c0dd48da21c40a3b3a10",
                        "text": "Boeing 747 is a plane.",
                    }
                ],
            )

    def test_search_index_sub_query_2(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict),
            patch.object(
                FactMatcherSimple,
                "_extract_entity_information",
                return_value=self.test_entity_relation_info_dict,
            ),
        ):

            fact_matcher = FactMatcherSimple(
                bear_data_path=f"{self.test_resources_abs_path}",
                file_index_dir=self.test_index_dir,
            )

            fact_matcher.index_dataset(
                [
                    {"text": "Boeing 747 is a plane."},
                 {"text": "Angela Merkel is the chancellor of Germany"}],
                text_key="text",
                
            )
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
                FactMatcherSimple,
                "_extract_entity_information",
                return_value=self.test_entity_relation_info_dict,
            ),
        ):

            fact_matcher = FactMatcherSimple(
                bear_data_path=f"{self.test_resources_abs_path}",
                file_index_dir=self.test_index_dir,
                save_file_content=True,
            )

            fact_matcher.index_dataset(
                [
                    {"text": "Abu Dhabi blah blah blah Khalifa bin Zayed Al Nahyan"},
                    {"text": "Abudhabi blah blah blah Khalifa bin Zayed Al Nahyan"},
                    {"text": "Armenia blah blah blah Nikol Pashinyan"},
                    {"text": "Free State of Fiume blah ducks blah Nikol Pashinyan Gabriele D'Annunzio"},
                    {"text": "Nepal NPL is cool Khadga Prasad Sharma Oli"},
                ],
                text_key="text",
                
            )
            fact_matcher.close()
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
                    "sentences": {"Abu Dhabi blah blah blah Khalifa bin Zayed Al Nahyan", "Abudhabi blah blah blah Khalifa bin Zayed Al Nahyan"},
                },
                "Q399": {
                    "subj_label": "Armenia",
                    "subj_aliases": {"Republic of Armenia", "🇦🇲", "ARM", "AM"},
                    "obj_id": "Q7035479",
                    "obj_label": "Nikol Pashinyan",
                    "obj_aliases": set(),
                    "occurrences": 1,
                    "sentences": {"Armenia blah blah blah Nikol Pashinyan"},
                },
                "Q548114": {
                    "subj_label": "Free State of Fiume",
                    "subj_aliases": set(),
                    "obj_id": "Q193236",
                    "obj_label": "Gabriele D'Annunzio",
                    "obj_aliases": set(),
                    "occurrences": 1,
                    "sentences": {"Free State of Fiume blah ducks blah Nikol Pashinyan Gabriele D'Annunzio"},
                },
                "Q5626824": {
                    "subj_label": "Gülcemal Sultan",
                    "subj_aliases": set(),
                    "obj_id": "Q222",
                    "obj_label": "Albania",
                    "obj_aliases": set(),
                    "occurrences": 0,
                    "sentences": set(),
                },
                "Q837": {
                    "subj_label": "Nepal",
                    "subj_aliases": {"NPL", "Federal Democratic Republic of Nepal", "NEP", "NP", "🇳🇵"},
                    "obj_id": "Q3195923",
                    "obj_label": "Khadga Prasad Sharma Oli",
                    "obj_aliases": set(),
                    "occurrences": 1,
                    "sentences": {"Nepal NPL is cool Khadga Prasad Sharma Oli"},
                },
            }
        })

    def test_create_fact_statistics_good3(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict_obj_aliases),
            patch.object(
                FactMatcherSimple,
                "_extract_entity_information",
                return_value=self.test_entity_relation_info_dict_filled_obj_aliases,
            ),
        ):

            fact_matcher = FactMatcherSimple(
                bear_data_path=f"{self.test_resources_abs_path}",
                file_index_dir=self.test_index_dir,
            )

            fact_matcher.index_dataset(
                [
                    {"text": "United States of America blah blah blah Washington, D.C."},
                    {"text": "United States of America blah Alexander blah blah Washington, D.C."},
                    {"text": "United States of America (U.S.A.) blah blah blah Washington, D.C."},
                    {"text": "United of America (U.S.A.) blah blah blah Washington, D.C."},
                    {"text": "Alexander Hamilton blah blah blah the United States of America"},
                    {"text": "Publius blah blah blah the USA based in Washington, D.C."},
                    {"text": "Hamilton blah blah blah United States of America"},
                    {"text": "US blah blah blah A. Ham"},
                ],
                text_key="text",
                
            )
            fact_matcher.close()
            fact_matcher.create_fact_statistics()
            self.assertEqual(fact_matcher.entity_relation_info_dict, {
            "P_00": {
                "Q30": {
                    "subj_label": "United States of America",
                    "subj_aliases": {"the United States of America","America","U.S.A.","USA","U.S.","US"},
                    "obj_id": "Q61",
                    "obj_label": "Washington, D.C.",
                    "obj_aliases": set(),
                    "occurrences": 5,
                    "sentences": set(),
                },
                "Q178903": {
                    "subj_label": "Alexander Hamilton",
                    "subj_aliases": {"Publius","Hamilton","Alexander Hamilton, US Treasury secretary","A. Ham"},
                    "obj_id": "Q30",
                    "obj_label": "United States of America",
                    "obj_aliases": {"the United States of America","America","U.S.A.","USA","U.S.","US"},
                    "occurrences": 3,
                    "sentences": set(),
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
                    "sentences": set(),
                }
            }
        })