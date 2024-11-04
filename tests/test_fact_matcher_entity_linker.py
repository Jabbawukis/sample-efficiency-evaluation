import os
import unittest
import shutil
from unittest.mock import patch, MagicMock

from sample_efficiency_evaluation import FactMatcherEntityLinking
from utility import utility


class FactMatcherTestEntityLinking(unittest.TestCase):

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
        self.maxDiff = None
        self.test_resources_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_resources"))
        self.indexer_mocked = MagicMock()
        self.writer_mocked = MagicMock()
        self.test_index_dir = f"{self.test_resources_abs_path}/test_index_dir_entity_linking"
        if os.path.exists(self.test_index_dir):
            shutil.rmtree(self.test_index_dir)

    def test_index_dataset_1(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict),
            patch.object(
                FactMatcherEntityLinking, "_initialize_index", return_value=(self.writer_mocked, self.indexer_mocked)
            ),
            patch.object(
                FactMatcherEntityLinking,
                "_extract_entity_information",
                return_value=self.test_entity_relation_info_dict,
            ),
            patch.object(
                FactMatcherEntityLinking,
                "_index_file",
            ) as mock_index_file,
            patch.object(
                FactMatcherEntityLinking,
                "_get_entity_ids",
            ) as get_entity_ids_mock
        ):

            fact_matcher = FactMatcherEntityLinking(
                bear_data_path=f"{self.test_resources_abs_path}",
                file_index_dir=self.test_index_dir,
            )
            fact_matcher.entity_linker = MagicMock()
            entity_1 = MagicMock()
            entity_1.get_id.return_value = 12525597
            entity_2 = MagicMock()
            entity_2.get_id.return_value = 194318
            entity_3 = MagicMock()
            entity_3.get_id.return_value = 664609
            entity_4 = MagicMock()
            entity_4.get_id.return_value = 193390
            entity_5 = MagicMock()
            entity_5.get_id.return_value = 197
            entity_6 = MagicMock()
            entity_6.get_id.return_value = 66
            get_entity_ids_mock.side_effect = [[entity_1, entity_2, entity_3], [entity_4]]
            fact_matcher.index_dataset(
                [
                    {"text": "I watched the Pirates of the Caribbean last silvester."},
                    {"text": "I follow the New England Patriots"},
                ],
                text_key="text",
                split_contents_into_sentences=False,
            )

            mock_index_file.assert_any_call("I watched the Pirates of the Caribbean last silvester. [Q12525597 Q194318 Q664609]")
            mock_index_file.assert_any_call("I follow the New England Patriots [Q193390]")
            self.assertEqual(mock_index_file.call_count, 2)

    def test_index_dataset_2(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict),
            patch.object(
                FactMatcherEntityLinking,
                "_extract_entity_information",
                return_value=self.test_entity_relation_info_dict,
            ),
            patch.object(
                FactMatcherEntityLinking, "_initialize_index", return_value=(self.writer_mocked, self.indexer_mocked)
            ),
            patch.object(
                FactMatcherEntityLinking,
                "_index_file",
            ) as mock_index_file,
            patch.object(
                FactMatcherEntityLinking,
                "_get_entity_ids",
            ) as get_entity_ids_mock,
        ):

            fact_matcher = FactMatcherEntityLinking(
                bear_data_path=f"{self.test_resources_abs_path}",
                file_index_dir=self.test_index_dir,
            )
            fact_matcher.entity_linker = MagicMock()
            entity_1 = MagicMock()
            entity_1.get_id.return_value = 12525597
            entity_2 = MagicMock()
            entity_2.get_id.return_value = 194318
            entity_3 = MagicMock()
            entity_3.get_id.return_value = 664609
            entity_4 = MagicMock()
            entity_4.get_id.return_value = 193390
            entity_5 = MagicMock()
            entity_5.get_id.return_value = 197
            entity_6 = MagicMock()
            entity_6.get_id.return_value = 66
            get_entity_ids_mock.side_effect = [[entity_1, entity_2, entity_3], [entity_4], [entity_5, entity_6]]

            fact_matcher.index_dataset(
                [
                    {"text": "I watched the Pirates of the Caribbean last silvester. I follow the New England Patriots."},
                    {"text": "Boeing 747 is a cool plane."},
                ],
                text_key="text",
                split_contents_into_sentences=True,
            )

            mock_index_file.assert_any_call("I watched the Pirates of the Caribbean last silvester. [Q12525597 Q194318 Q664609]")
            mock_index_file.assert_any_call("I follow the New England Patriots. [Q193390]")
            mock_index_file.assert_any_call("Boeing 747 is a cool plane. [Q197 Q66]")
            self.assertEqual(mock_index_file.call_count, 3)

    def test_create_fact_statistics_good(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict),
            patch.object(
                FactMatcherEntityLinking,
                "_extract_entity_information",
                return_value=self.test_entity_relation_info_dict,
            ),
        ):

            fact_matcher = FactMatcherEntityLinking(
                bear_data_path=f"{self.test_resources_abs_path}",
                file_index_dir=self.test_index_dir,
            )

            fact_matcher.index_dataset(
                [{"text": "Abu Dhabi blah blah blah Khalifa bin Zayed Al Nahyan"},
                    {"text": "Abudhabi blah blah blah Khalifa bin Zayed Al Nahyan"},
                    {"text": "Armenia blah blah blah Nikol Pashinyan"},
                    {"text": "Free State of Fiume blah ducks blah Nikol Pashinyan Gabriele D'Annunzio"},
                    {"text": "Nepal NPL is cool Khadga Prasad Sharma Oli"}],
                text_key="text",
                split_contents_into_sentences=True,
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
                    "occurrences": 1,
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
        })