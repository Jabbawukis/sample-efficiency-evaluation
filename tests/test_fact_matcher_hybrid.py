import os
import unittest
import shutil
from unittest.mock import patch, MagicMock

from sample_efficiency_evaluation import FactMatcherHybrid
from utility import utility


class FactMatcherTestHybrid(unittest.TestCase):

    def setUp(self) -> None:
        self.test_relation_info_dict_obj_aliases = {
            "P_00": {"domains": ["stuff"]},
            "P_01": {"domains": ["hi"]},
        }
        self.test_entity_relation_info_dict_filled_obj_aliases = {
            "P_00": {
                "Q30": {
                    "subj_label": "United States of America",
                    "subj_aliases": {"the United States of America", "America", "U.S.A.", "USA", "U.S.", "US"},
                    "obj_id": "Q61",
                    "obj_label": "Washington, D.C.",
                    "obj_aliases": set(),
                    "occurrences": 0,
                    "sentences": set(),
                },
                "Q178903": {
                    "subj_label": "Alexander Hamilton",
                    "subj_aliases": {"Publius", "Hamilton", "Alexander Hamilton, US Treasury secretary", "A. Ham"},
                    "obj_id": "Q30",
                    "obj_label": "United States of America",
                    "obj_aliases": {"the United States of America", "America", "U.S.A.", "USA", "U.S.", "US"},
                    "occurrences": 0,
                    "sentences": set(),
                }
            },
            "P_01": {
                "Q2127993": {
                    "subj_label": "Rainer Bernhardt",
                    "subj_aliases": {"Rainer Herbert Georg Bernhardt", "Bernhardt"},
                    "obj_id": "Q30",
                    "obj_label": "United States of America",
                    "obj_aliases": {"the United States of America", "America", "U.S.A.", "USA", "U.S.", "US"},
                    "occurrences": 0,
                    "sentences": set(),
                },
                "Q62085": {
                    "subj_label": "Joachim Sauer",
                    "subj_aliases": {"J. Sauer", "Sauer"},
                    "obj_id": "Q567",
                    "obj_label": "Angela Merkel",
                    "obj_aliases": {"Merkel"},
                    "occurrences": 0,
                    "sentences": set(),
                }
            }
        }
        self.maxDiff = None
        self.test_resources_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_resources"))
        self.indexer_mocked = MagicMock()
        self.writer_mocked = MagicMock()
        self.test_index_dir = f"{self.test_resources_abs_path}/test_index_dir_entity_linking"
        if os.path.exists(self.test_index_dir):
            shutil.rmtree(self.test_index_dir)

    def test_index_dataset_1(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict_obj_aliases),
            patch.object(
                FactMatcherHybrid, "_initialize_index", return_value=(self.writer_mocked, self.indexer_mocked)
            ),
            patch.object(
                FactMatcherHybrid,
                "_extract_entity_information",
                return_value=self.test_entity_relation_info_dict_filled_obj_aliases,
            ),
            patch.object(
                FactMatcherHybrid,
                "_index_file",
            ) as mock_index_file
        ):

            fact_matcher = FactMatcherHybrid(
                bear_data_path=f"{self.test_resources_abs_path}",
                file_index_dir=self.test_index_dir,
            )
            fact_matcher.index_dataset(
                [
                    {"text": "I watched the Pirates of the Caribbean last silvester."},
                    {"text": "I follow the New England Patriots"},
                ],
                text_key="text",
                split_contents_into_sentences=False,
            )

            mock_index_file.assert_any_call("I watched the Pirates of the Caribbean last silvester.")
            mock_index_file.assert_any_call("I follow the New England Patriots")
            self.assertEqual(mock_index_file.call_count, 2)

    def test_index_dataset_2(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict_obj_aliases),
            patch.object(
                FactMatcherHybrid,
                "_extract_entity_information",
                return_value=self.test_entity_relation_info_dict_filled_obj_aliases,
            ),
            patch.object(
                FactMatcherHybrid, "_initialize_index", return_value=(self.writer_mocked, self.indexer_mocked)
            ),
            patch.object(
                FactMatcherHybrid,
                "_index_file",
            ) as mock_index_file
        ):

            fact_matcher = FactMatcherHybrid(
                bear_data_path=f"{self.test_resources_abs_path}",
                file_index_dir=self.test_index_dir,
            )
            fact_matcher.index_dataset(
                [
                    {"text": "I watched the Pirates of the Caribbean last silvester. I follow the New England Patriots."},
                    {"text": "Boeing 747 is a cool plane."},
                ],
                text_key="text",
                split_contents_into_sentences=True,
            )

            mock_index_file.assert_any_call("I watched the Pirates of the Caribbean last silvester. I follow the New England Patriots.")
            mock_index_file.assert_any_call("Boeing 747 is a cool plane.")
            self.assertEqual(mock_index_file.call_count, 2)

    def test_create_fact_statistics_good(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict_obj_aliases),
            patch.object(
                FactMatcherHybrid,
                "_extract_entity_information",
                return_value=self.test_entity_relation_info_dict_filled_obj_aliases,
            ),
        ):

            fact_matcher = FactMatcherHybrid(
                bear_data_path=f"{self.test_resources_abs_path}",
                file_index_dir=self.test_index_dir,
                save_file_content=True,
            )

            fact_matcher.index_dataset(
                [
                    {"text": "United States of America blah blah blah Washington, D.C. blah."
                             " United States of America blah Alexander blah blah Washington, D.C. blah."
                             " United States of America (U.S.A.) blah blah blah Washington, D.C. blah."},
                    {"text": "United of America (U.S.A.) blah blah blah Washington, D.C. blah."
                             " Alexander Hamilton blah blah blah the United States of America."},
                    {"text": "Publius blah blah blah the USA based in Washington, D.C. blah."
                             " Hamilton blah blah blah United States of America."
                             " US blah blah blah A. Ham"},
                    {"text": "Rainer Herbert Georg Bernhardt blah blah blah the USA blah."
                             " Bernhardt blah blah blah United States of America."},
                    {"text": "Joachim Sauer and Merkel."
                             " A. Merkel blah blah blah Joachim Sauer."}
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
                    "sentences": {"United States of America blah blah blah Washington, D.C. blah.",
                                  "United States of America blah Alexander blah blah Washington, D.C. blah.",
                                  "United States of America (U.S.A.) blah blah blah Washington, D.C. blah.",
                                  "United of America (U.S.A.) blah blah blah Washington, D.C. blah.",
                                  "Publius blah blah blah the USA based in Washington, D.C. blah."}
                },
                "Q178903": {
                    "subj_label": "Alexander Hamilton",
                    "subj_aliases": {"Publius","Hamilton","Alexander Hamilton, US Treasury secretary","A. Ham"},
                    "obj_id": "Q30",
                    "obj_label": "United States of America",
                    "obj_aliases": {"the United States of America","America","U.S.A.","USA","U.S.","US"},
                    "occurrences": 3, # "A. Ham" does not get recognized by the entity linker.
                    "sentences": {"Alexander Hamilton blah blah blah the United States of America.",
                                  "Hamilton blah blah blah United States of America.",
                                  "Publius blah blah blah the USA based in Washington, D.C. blah."},
                }
            },
            "P_01": {
                "Q2127993": {
                    "subj_label": "Rainer Bernhardt",
                    "subj_aliases": {"Rainer Herbert Georg Bernhardt", "Bernhardt"},
                    "obj_id": "Q30",
                    "obj_label": "United States of America",
                    "obj_aliases": {"the United States of America","America","U.S.A.","USA","U.S.","US"},
                    "occurrences": 2, # "Rainer Bernhardt" does not get recognized by the entity linker.
                    "sentences": {"Rainer Herbert Georg Bernhardt blah blah blah the USA blah.",
                                  "Bernhardt blah blah blah United States of America."}
                },
                "Q62085": {
                    "subj_label": "Joachim Sauer",
                    "subj_aliases": {"J. Sauer", "Sauer"},
                    "obj_id": "Q567",
                    "obj_label": "Angela Merkel",
                    "obj_aliases": {"Merkel"},
                    "occurrences": 2, # "A. Merkel" by entity linker, "Merkel" by string matching.
                    "sentences": {"Joachim Sauer and Merkel.",
                                  "A. Merkel blah blah blah Joachim Sauer."}
                }
            }
        })