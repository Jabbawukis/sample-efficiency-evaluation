import os
import unittest
from unittest.mock import patch, MagicMock

from sample_efficiency_evaluation import FactMatcherEntityLinking
from utility import utility


class FactMatcherEntityLinkingTest(unittest.TestCase):

    def setUp(self) -> None:
        self.test_relation_info_dict = {
            "P6": {"domains": ["Political", "Biographical", "Historical"]},
            "P19": {"domains": ["Biographical"]},
        }
        self.test_entity_relation_occurrence_info_dict = {
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
        self.maxDiff = None
        self.test_resources_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_resources"))

    def test_create_fact_statistics_good(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict),
            patch.object(
                utility,
                "extract_entity_information",
                return_value=self.test_entity_relation_occurrence_info_dict,
            ),
            patch.object(
                FactMatcherEntityLinking,
                "_get_entity_ids",
            ) as get_entity_ids_mock,
        ):

            fact_matcher = FactMatcherEntityLinking(
                bear_data_path=f"{self.test_resources_abs_path}",
                save_file_content=True,
            )

            fact_matcher.entity_linker = MagicMock()
            entity_1 = MagicMock()
            entity_1.get_id.return_value = 1519
            entity_2 = MagicMock()
            entity_2.get_id.return_value = 1059948
            entity_3 = MagicMock()
            entity_3.get_id.return_value = 399
            entity_4 = MagicMock()
            entity_4.get_id.return_value = 7035479
            entity_5 = MagicMock()
            entity_5.get_id.return_value = 837
            entity_6 = MagicMock()
            entity_6.get_id.return_value = 3195923
            get_entity_ids_mock.side_effect = [
                [entity_1, entity_2],
                [entity_2],
                [entity_3, entity_4],
                [],
                [entity_5, entity_6],
            ]

            fact_matcher.create_fact_statistics(
                [
                    {
                        "text": "Abu Dhabi blah blah blah Khalifa bin Zayed Al Nahyan. Abudhabi blah blah blah Khalifa bin Zayed Al Nahyan."
                    },
                    {"text": "Armenia blah blah blah Nikol Pashinyan"},
                    {
                        "text": "Free State of Fiume blah ducks blah Nikol Pashinyan Gabriele D'Annunzio. Nepal NPL is cool Khadga Prasad Sharma Oli"
                    },
                ],
                text_key="text",
            )
            self.assertEqual(
                fact_matcher.entity_relation_occurrence_info_dict,
                {
                    "P6": {
                        "Q1519": {
                            "obj_aliases": set(),
                            "obj_id": "Q1059948",
                            "obj_label": "Khalifa bin Zayed Al Nahyan",
                            "occurrences": 1,
                            "sentences": {"Abu Dhabi blah blah blah Khalifa bin Zayed Al " "Nahyan."},
                            "subj_aliases": {"Abudhabi", "Abū Dhabi"},
                            "subj_label": "Abu Dhabi",
                        },
                        "Q399": {
                            "obj_aliases": set(),
                            "obj_id": "Q7035479",
                            "obj_label": "Nikol Pashinyan",
                            "occurrences": 1,
                            "sentences": {"Armenia blah blah blah Nikol Pashinyan"},
                            "subj_aliases": {"🇦🇲", "AM", "Republic of Armenia", "ARM"},
                            "subj_label": "Armenia",
                        },
                        "Q548114": {
                            "obj_aliases": set(),
                            "obj_id": "Q193236",
                            "obj_label": "Gabriele D'Annunzio",
                            "occurrences": 0,
                            "sentences": set(),
                            "subj_aliases": set(),
                            "subj_label": "Free State of Fiume",
                        },
                        "Q5626824": {
                            "obj_aliases": set(),
                            "obj_id": "Q222",
                            "obj_label": "Albania",
                            "occurrences": 0,
                            "sentences": set(),
                            "subj_aliases": set(),
                            "subj_label": "Gülcemal Sultan",
                        },
                        "Q837": {
                            "obj_aliases": set(),
                            "obj_id": "Q3195923",
                            "obj_label": "Khadga Prasad Sharma Oli",
                            "occurrences": 1,
                            "sentences": {"Nepal NPL is cool Khadga Prasad Sharma Oli"},
                            "subj_aliases": {"Federal Democratic Republic of Nepal", "NEP", "NP", "NPL", "🇳🇵"},
                            "subj_label": "Nepal",
                        },
                    }
                },
            )
