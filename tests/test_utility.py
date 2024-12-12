import os
import unittest
from unittest.mock import patch

from utility import utility


class UtilityTest(unittest.TestCase):

    def setUp(self) -> None:
        self.maxDiff = None
        self.entity_relation_result_info_dict_1 = {
            "P6": {
                "Q1519": {
                    "obj_aliases": [],
                    "obj_id": "Q1059948",
                    "obj_label": "Khalifa bin Zayed Al Nahyan",
                    "occurrences": 1,
                    "sentences": {"Abu Dhabi blah blah blah Khalifa bin Zayed Al " "Nahyan.": 1},
                    "subj_aliases": ["Abudhabi", "Abū Dhabi"],
                    "subj_label": "Abu Dhabi",
                },
                "Q399": {
                    "obj_aliases": [],
                    "obj_id": "Q7035479",
                    "obj_label": "Nikol Pashinyan",
                    "occurrences": 1,
                    "sentences": {"Armenia blah blah blah Nikol Pashinyan": 1},
                    "subj_aliases": ["🇦🇲", "AM", "Republic of Armenia", "ARM"],
                    "subj_label": "Armenia",
                },
            },
            "P2": {
                "Q548114": {
                    "obj_aliases": [],
                    "obj_id": "Q193236",
                    "obj_label": "Gabriele D'Annunzio",
                    "occurrences": 0,
                    "sentences": {},
                    "subj_aliases": [],
                    "subj_label": "Free State of Fiume",
                },
                "Q5626824": {
                    "obj_aliases": [],
                    "obj_id": "Q222",
                    "obj_label": "Albania",
                    "occurrences": 0,
                    "sentences": {},
                    "subj_aliases": [],
                    "subj_label": "Gülcemal Sultan",
                },
                "Q837": {
                    "obj_aliases": [],
                    "obj_id": "Q3195923",
                    "obj_label": "Khadga Prasad Sharma Oli",
                    "occurrences": 1,
                    "sentences": {"Nepal NPL is cool Khadga Prasad Sharma Oli": 1},
                    "subj_aliases": ["Federal Democratic Republic of Nepal", "NEP", "NP", "NPL", "🇳🇵"],
                    "subj_label": "Nepal",
                },
            },
        }
        self.entity_relation_result_info_dict_2 = {
            "P6": {
                "Q1519": {
                    "obj_aliases": [],
                    "obj_id": "Q1059948",
                    "obj_label": "Khalifa bin Zayed Al Nahyan",
                    "occurrences": 1,
                    "sentences": {"Abu Dhabi blah blah blah Khalifa bin Zayed Al " "Nahyan.": 1},
                    "subj_aliases": ["Abudhabi", "Abū Dhabi"],
                    "subj_label": "Abu Dhabi",
                },
                "Q399": {
                    "obj_aliases": [],
                    "obj_id": "Q7035479",
                    "obj_label": "Nikol Pashinyan",
                    "occurrences": 1,
                    "sentences": {"Armenia blah blah blah Nikol Pashinyan blub": 1},
                    "subj_aliases": ["🇦🇲", "AM", "Republic of Armenia", "ARM"],
                    "subj_label": "Armenia",
                },
            },
            "P2": {
                "Q548114": {
                    "obj_aliases": [],
                    "obj_id": "Q193236",
                    "obj_label": "Gabriele D'Annunzio",
                    "occurrences": 0,
                    "sentences": {},
                    "subj_aliases": [],
                    "subj_label": "Free State of Fiume",
                },
                "Q5626824": {
                    "obj_aliases": [],
                    "obj_id": "Q222",
                    "obj_label": "Albania",
                    "occurrences": 2,
                    "sentences": {"sentence 1": 1, "sentence 2": 1},
                    "subj_aliases": [],
                    "subj_label": "Gülcemal Sultan",
                },
                "Q837": {
                    "obj_aliases": [],
                    "obj_id": "Q3195923",
                    "obj_label": "Khadga Prasad Sharma Oli",
                    "occurrences": 1,
                    "sentences": {"Nepal NPL is cool Khadga Prasad Sharma Oli": 1},
                    "subj_aliases": ["Federal Democratic Republic of Nepal", "NEP", "NP", "NPL", "🇳🇵"],
                    "subj_label": "Nepal",
                },
            },
        }

    def test_word_in_sentence(self):
        self.assertTrue(
            utility.word_in_sentence("zayed Al NAHYAn", "Abu Dhabi blah blah blah Khalifa bin Zayed Al Nahyan.")
        )
        self.assertTrue(utility.word_in_sentence("khalifa", "Abu Dhabi blah blah blah Khalifa bin Zayed Al Nahyan."))
        self.assertTrue(utility.word_in_sentence("Armenia", "Armenia blah blah blah Nikol Pashinyan"))
        self.assertTrue(utility.word_in_sentence("armenia", "Armenia blah blah blah Nikol Pashinyan"))
        self.assertTrue(utility.word_in_sentence("Yerevan T.c.", "Yerevan t.c.! blah blah blah Nikol Pashinyan"))
        self.assertTrue(utility.word_in_sentence("U.S.A.", "Yerevan t.c.! U.S.A. blah blah blah Nikol Pashinyan"))
        self.assertTrue(utility.word_in_sentence("Nepal", "Nepal NPL is cool Khadga Prasad Sharma Oli"))
        self.assertTrue(utility.word_in_sentence("nepal", "Nepal NPL is cool Khadga Prasad Sharma Oli"))
        self.assertTrue(
            utility.word_in_sentence(
                "Washington, D.C.", "United States of America blah blah blah Washington, D.C.! blah."
            )
        )

        self.assertFalse(utility.word_in_sentence("Kathmandu", "Nepal NPL is cool Khadga Prasad Sharma Oli"))
        self.assertFalse(utility.word_in_sentence("Yerevan T.c.", "Armenia blah Yerevan T.c blah blah Nikol Pashinyan"))
        self.assertFalse(utility.word_in_sentence("Dubai", "Abu Dhabi blah blah blah Khalifa bin Zayed Al Nahyan."))

    def test_join_relation_info_json_files_good_1(self):
        with (
            patch.object(utility, "load_json_dict") as load_json_dict,
            patch.object(utility, "save_dict_as_json") as save_dict_as_json,
            patch.object(
                os,
                "listdir",
                return_value=["0_relation_info.json", "1_relation_info.json"],
            ),
        ):
            load_json_dict.side_effect = [
                self.entity_relation_result_info_dict_1,
                self.entity_relation_result_info_dict_2,
            ]
            utility.join_relation_occurrences_info_json_files("output")
            save_dict_as_json.assert_called_once_with(
                {
                    "P6": {
                        "Q1519": {
                            "obj_aliases": [],
                            "obj_id": "Q1059948",
                            "obj_label": "Khalifa bin Zayed Al Nahyan",
                            "occurrences": 2,
                            "subj_aliases": ["Abudhabi", "Abū Dhabi"],
                            "subj_label": "Abu Dhabi",
                        },
                        "Q399": {
                            "obj_aliases": [],
                            "obj_id": "Q7035479",
                            "obj_label": "Nikol Pashinyan",
                            "occurrences": 2,
                            "subj_aliases": ["🇦🇲", "AM", "Republic of Armenia", "ARM"],
                            "subj_label": "Armenia",
                        },
                    },
                    "P2": {
                        "Q548114": {
                            "obj_aliases": [],
                            "obj_id": "Q193236",
                            "obj_label": "Gabriele D'Annunzio",
                            "occurrences": 0,
                            "subj_aliases": [],
                            "subj_label": "Free State of Fiume",
                        },
                        "Q5626824": {
                            "obj_aliases": [],
                            "obj_id": "Q222",
                            "obj_label": "Albania",
                            "occurrences": 2,
                            "subj_aliases": [],
                            "subj_label": "Gülcemal Sultan",
                        },
                        "Q837": {
                            "obj_aliases": [],
                            "obj_id": "Q3195923",
                            "obj_label": "Khadga Prasad Sharma Oli",
                            "occurrences": 2,
                            "subj_aliases": ["Federal Democratic Republic of Nepal", "NEP", "NP", "NPL", "🇳🇵"],
                            "subj_label": "Nepal",
                        },
                    },
                },
                "output/joined_relation_occurrence_info.json",
            )
