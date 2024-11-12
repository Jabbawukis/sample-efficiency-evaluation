import logging
import os
import unittest
from unittest.mock import patch

from utility import utility


class UtilityTest(unittest.TestCase):

    def setUp(self) -> None:
        self.maxDiff = None
        self.test_resources_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_resources"))

        self.test_bear_relation_info_dict = {
            "P6": {"domains": ["Political", "Biographical", "Historical"]},
            "P19": {"domains": ["Biographical"]},
        }
        self.test_bear_relation_info_dict_obj_aliases = {
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
                },
            },
            "P_01": {
                "Q2127993": {
                    "subj_label": "Rainer Bernhardt",
                    "subj_aliases": {"Rainer Herbert Georg Bernhardt"},
                    "obj_id": "Q30",
                    "obj_label": "United States of America",
                    "obj_aliases": {"the United States of America", "America", "U.S.A.", "USA", "U.S.", "US"},
                    "occurrences": 0,
                    "sentences": set(),
                }
            },
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
        self.entity_relation_result_info_dict_1 = {
            "P6": {
                "Q1519": {
                    "obj_aliases": [],
                    "obj_id": "Q1059948",
                    "obj_label": "Khalifa bin Zayed Al Nahyan",
                    "occurrences": 1,
                    "sentences": ["Abu Dhabi blah blah blah Khalifa bin Zayed Al " "Nahyan."],
                    "subj_aliases": ["Abudhabi", "Abū Dhabi"],
                    "subj_label": "Abu Dhabi",
                },
                "Q399": {
                    "obj_aliases": [],
                    "obj_id": "Q7035479",
                    "obj_label": "Nikol Pashinyan",
                    "occurrences": 1,
                    "sentences": ["Armenia blah blah blah Nikol Pashinyan"],
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
                    "sentences": [],
                    "subj_aliases": [],
                    "subj_label": "Free State of Fiume",
                },
                "Q5626824": {
                    "obj_aliases": [],
                    "obj_id": "Q222",
                    "obj_label": "Albania",
                    "occurrences": 0,
                    "sentences": [],
                    "subj_aliases": [],
                    "subj_label": "Gülcemal Sultan",
                },
                "Q837": {
                    "obj_aliases": [],
                    "obj_id": "Q3195923",
                    "obj_label": "Khadga Prasad Sharma Oli",
                    "occurrences": 1,
                    "sentences": ["Nepal NPL is cool Khadga Prasad Sharma Oli"],
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
                    "sentences": ["Abu Dhabi blah blah blah Khalifa bin Zayed Al " "Nahyan."],
                    "subj_aliases": ["Abudhabi", "Abū Dhabi"],
                    "subj_label": "Abu Dhabi",
                },
                "Q399": {
                    "obj_aliases": [],
                    "obj_id": "Q7035479",
                    "obj_label": "Nikol Pashinyan",
                    "occurrences": 1,
                    "sentences": ["Armenia blah blah blah Nikol Pashinyan blub"],
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
                    "sentences": [],
                    "subj_aliases": [],
                    "subj_label": "Free State of Fiume",
                },
                "Q5626824": {
                    "obj_aliases": [],
                    "obj_id": "Q222",
                    "obj_label": "Albania",
                    "occurrences": 2,
                    "sentences": ["sentence 1", "sentence 2"],
                    "subj_aliases": [],
                    "subj_label": "Gülcemal Sultan",
                },
                "Q837": {
                    "obj_aliases": [],
                    "obj_id": "Q3195923",
                    "obj_label": "Khadga Prasad Sharma Oli",
                    "occurrences": 1,
                    "sentences": ["Nepal NPL is cool Khadga Prasad Sharma Oli"],
                    "subj_aliases": ["Federal Democratic Republic of Nepal", "NEP", "NP", "NPL", "🇳🇵"],
                    "subj_label": "Nepal",
                },
            },
        }

    def test_extract_entity_information_good(self):
        with (
            patch.object(
                utility, "load_json_dict", return_value=self.test_bear_relation_info_dict
            ) as mock_load_json_dict,
            patch.object(logging, "error") as mock_error,
        ):

            result = utility.extract_entity_information(
                f"{self.test_resources_abs_path}/BEAR", f"{self.test_resources_abs_path}/relation_info.json"
            )

            self.assertEqual(result, self.test_entity_relation_info_dict)
            mock_error.assert_called_once()
            mock_load_json_dict.assert_called_once_with(f"{self.test_resources_abs_path}/relation_info.json")

    def test_extract_entity_information_good_filled_obj_aliases(self):
        with patch.object(
            utility, "load_json_dict", return_value=self.test_bear_relation_info_dict_obj_aliases
        ) as mock_load_json_dict:
            result = utility.extract_entity_information(
                f"{self.test_resources_abs_path}/BEAR", f"{self.test_resources_abs_path}/relation_info.json"
            )

            self.assertEqual(result, self.test_entity_relation_info_dict_filled_obj_aliases)
            mock_load_json_dict.assert_called_once_with(f"{self.test_resources_abs_path}/relation_info.json")

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
            utility.join_relation_info_json_files("output", correct_possible_duplicates=True, remove_sentences=True)
            save_dict_as_json.assert_called_once_with(
                {
                    "P6": {
                        "Q1519": {
                            "obj_aliases": [],
                            "obj_id": "Q1059948",
                            "obj_label": "Khalifa bin Zayed Al Nahyan",
                            "occurrences": 1,
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
                            "occurrences": 1,
                            "subj_aliases": ["Federal Democratic Republic of Nepal", "NEP", "NP", "NPL", "🇳🇵"],
                            "subj_label": "Nepal",
                        },
                    },
                },
                "output/joined_relation_info.json",
            )

    def test_join_relation_info_json_files_good_2(self):
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
            utility.join_relation_info_json_files("output", correct_possible_duplicates=True, remove_sentences="True")
            save_dict_as_json.assert_called_once_with(
                {
                    "P6": {
                        "Q1519": {
                            "obj_aliases": [],
                            "obj_id": "Q1059948",
                            "obj_label": "Khalifa bin Zayed Al Nahyan",
                            "occurrences": 1,
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
                            "occurrences": 1,
                            "subj_aliases": ["Federal Democratic Republic of Nepal", "NEP", "NP", "NPL", "🇳🇵"],
                            "subj_label": "Nepal",
                        },
                    },
                },
                "output/joined_relation_info.json",
            )

    def test_save_fact_statistics_dict_as_json_good(self):
        with patch.object(utility, "save_dict_as_json") as save_dict_as_json:
            utility.save_fact_statistics_dict_as_json(self.test_entity_relation_info_dict, "output.json")
            save_dict_as_json.assert_called_once_with(
                {
                    "P6": {
                        "Q1519": {
                            "obj_id": "Q1059948",
                            "occurrences": 0,
                        },
                        "Q399": {
                            "obj_id": "Q7035479",
                            "occurrences": 0,
                        },
                        "Q548114": {
                            "obj_id": "Q193236",
                            "occurrences": 0,
                        },
                        "Q5626824": {
                            "obj_id": "Q222",
                            "occurrences": 0,
                        },
                        "Q837": {
                            "obj_id": "Q3195923",
                            "occurrences": 0,
                        },
                    }
                },
                "output.json",
            )
