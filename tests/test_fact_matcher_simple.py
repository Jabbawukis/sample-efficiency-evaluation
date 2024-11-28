import logging
import os
import unittest
from unittest.mock import patch, MagicMock

from sample_efficiency_evaluation import FactMatcherSimple


class FactMatcherSimpleTest(unittest.TestCase):

    def setUp(self) -> None:
        os.environ["PYTHONHASHSEED"] = "0"
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
        self.test_entity_relation_occurrence_info_dict_obj_aliases_extended = {
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
                    "subj_aliases": {
                        "Publius",
                        "Hamilton",
                        "Alexander Hamilton, US Treasury secretary",
                        "A. Ham",
                        "RB",
                    },
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
                    "subj_aliases": {"Rainer Herbert Georg Bernhardt", "Bernhardt", "RB"},
                    "obj_id": "Q30",
                    "obj_label": "United States of America",
                    "obj_aliases": {"the United States of America", "America", "U.S.A.", "USA", "U.S.", "US"},
                    "occurrences": 0,
                    "sentences": set(),
                },
                "Q38": {
                    "subj_label": "Italy",
                    "subj_aliases": {"IT", "ITA", "Italia", "Italian Republic"},
                    "obj_id": "Q652",
                    "obj_label": "Italian",
                    "obj_aliases": set(),
                    "occurrences": 0,
                    "sentences": set(),
                },
            },
        }
        self.test_relation_mapping_dict = {
            "italy": {"relations": {("P_01", "Q38")}},
            "IT": {"relations": {("P_01", "Q38")}},
            "ITA": {"relations": {("P_01", "Q38")}},
            "italia": {"relations": {("P_01", "Q38")}},
            "italian republic": {"relations": {("P_01", "Q38")}},
            "a. ham": {"relations": {("P_00", "Q178903")}},
            "alexander hamilton": {"relations": {("P_00", "Q178903")}},
            "alexander hamilton , us treasury secretary": {"relations": {("P_00", "Q178903")}},
            "america": {"relations": {("P_00", "Q30")}},
            "bernhardt": {"relations": {("P_01", "Q2127993")}},
            "hamilton": {"relations": {("P_00", "Q178903")}},
            "publius": {"relations": {("P_00", "Q178903")}},
            "rainer bernhardt": {"relations": {("P_01", "Q2127993")}},
            "rainer herbert georg bernhardt": {"relations": {("P_01", "Q2127993")}},
            "RB": {"relations": {("P_00", "Q178903"), ("P_01", "Q2127993")}},
            "the united states of america": {"relations": {("P_00", "Q30")}},
            "u.s .": {"relations": {("P_00", "Q30")}},
            "u.s.a .": {"relations": {("P_00", "Q30")}},
            "united states of america": {"relations": {("P_00", "Q30")}},
            "US": {"relations": {("P_00", "Q30")}},
            "USA": {"relations": {("P_00", "Q30")}},
        }
        self.test_entity_relation_occurrence_info_dict_small = {
            "P_00": {
                "Q173017": {
                    "subj_label": "Limpopo River",
                    "subj_aliases": {"Limpopo"},
                    "obj_id": "Q15",
                    "obj_label": "Africa",
                    "obj_aliases": set(),
                    "occurrences": 0,
                    "sentences": set(),
                }
            }
        }
        self.test_relation_mapping_dict_small = {
            "limpopo river": {"relations": {("P_00", "Q173017")}},
            "limpopo": {"relations": {("P_00", "Q173017")}},
        }

        self.maxDiff = None
        self.test_resources_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_resources"))

    def test_extract_entity_information_good(self):
        with (patch.object(logging, "error") as mock_error,):
            fact_matcher = FactMatcherSimple(
                bear_data_path=f"{self.test_resources_abs_path}",
            )

            self.assertEqual(fact_matcher.entity_relation_occurrence_info_dict, self.test_entity_relation_info_dict)
            mock_error.assert_called_once()

    def test_extract_entity_information_good_alias_extension_1(self):
        test_entity_relation_info_dict_alias_extended = self.test_entity_relation_info_dict.copy()
        test_entity_relation_info_dict_alias_extended["P6"]["Q5626824"]["subj_aliases"].update(["G. Sultan", "Sultan"])
        test_entity_relation_info_dict_alias_extended["P6"]["Q548114"]["obj_aliases"].update(
            ["G. D'Annunzio", "D'Annunzio"]
        )
        with (patch.object(logging, "error") as mock_error,):
            fact_matcher = FactMatcherSimple(
                bear_facts_path=f"{self.test_resources_abs_path}/BEAR",
                bear_relation_info_path=f"{self.test_resources_abs_path}/relation_info.json",
                path_to_alias_extensions=f"{self.test_resources_abs_path}/aliases_extension_test.json",
            )

            self.assertEqual(
                fact_matcher.entity_relation_occurrence_info_dict, test_entity_relation_info_dict_alias_extended
            )
            mock_error.assert_called_once()

    def test_extract_entity_information_good_filled_obj_aliases(self):
        with (patch.object(logging, "error") as mock_error,):
            fact_matcher = FactMatcherSimple(
                bear_facts_path=f"{self.test_resources_abs_path}/BEAR",
                bear_relation_info_path=f"{self.test_resources_abs_path}/relation_info_obj_aliases.json",
            )

            self.assertEqual(
                fact_matcher.entity_relation_occurrence_info_dict,
                self.test_entity_relation_info_dict_filled_obj_aliases,
            )
            mock_error.assert_not_called()

    def test_create_mapped_relations_good(self):
        with (
            patch.object(
                FactMatcherSimple,
                "extract_entity_information",
                return_value=self.test_entity_relation_occurrence_info_dict_obj_aliases_extended,
            ),
        ):

            fact_matcher = FactMatcherSimple(
                bear_data_path=f"{self.test_resources_abs_path}",
            )
            self.assertEqual(fact_matcher.relation_mapping_dict, self.test_relation_mapping_dict)
            self.assertEqual(fact_matcher.max_ngram, 6)

    def test_create_fact_statistics_good(self):
        with (
            patch.object(
                FactMatcherSimple,
                "extract_entity_information",
                return_value=self.test_entity_relation_occurrence_info_dict_obj_aliases_extended,
            ),
            patch.object(
                FactMatcherSimple,
                "_create_mapped_relations",
                return_value=self.test_relation_mapping_dict,
            ),
        ):

            fact_matcher = FactMatcherSimple(bear_data_path=f"{self.test_resources_abs_path}")
            fact_matcher.max_ngram = 6

            data = [
                {
                    "text": "United States of America blah blah blah Washington, D.C. blah."
                    " United States of America blah Alexander Hamilton, US Treasury secretary blah blah Washington, D.C. blah."
                    " United States of America (U.S.A.) blah blah blah Washington, D.C. blah."
                },
                {
                    "text": "United of America (U.S.A.) blah blah blah Washington, D.C. blah."
                    " Alexander Hamilton blah blah blah the United States of America."
                },
                {
                    "text": "Publius blah blah blah the USA based in Washington, D.C. blah."
                    " Hamilton blah blah blah United States of America."
                    " US blah blah blah A. Ham."
                    " United States of America (U.S.A.) blah blah blah Washington, D.C. blah."
                },
                {
                    "text": "Rainer Herbert Georg Bernhardt blah blah blah the USA blah."
                    " Bernhardt blah blah blah United States of America."
                    " RB blah blah blah the USA."
                    " The Italian (IT) blah blah blah team."
                },
                {
                    "text": "In Austrian schools, French has already been overtaken by Italian as the second mostchildren at home, as it is the common language of the local labour markets.",
                },
            ]

            fact_matcher.create_fact_statistics(data, text_key="text", save_file_content=True)

            self.assertEqual(
                fact_matcher.entity_relation_occurrence_info_dict,
                {
                    "P_00": {
                        "Q30": {
                            "subj_label": "United States of America",
                            "subj_aliases": {"the United States of America", "America", "U.S.A.", "USA", "U.S.", "US"},
                            "obj_id": "Q61",
                            "obj_label": "Washington, D.C.",
                            "obj_aliases": set(),
                            "occurrences": 5,
                            "sentences": {
                                "United States of America blah blah blah Washington, D.C. blah.",
                                "United States of America blah Alexander Hamilton, US Treasury secretary blah blah Washington, D.C. blah.",
                                "United States of America (U.S.A.) blah blah blah Washington, D.C. blah.",
                                "United of America (U.S.A.) blah blah blah Washington, D.C. blah.",
                                "Publius blah blah blah the USA based in Washington, D.C. blah.",
                            },
                        },
                        "Q178903": {
                            "subj_label": "Alexander Hamilton",
                            "subj_aliases": {
                                "Publius",
                                "Hamilton",
                                "Alexander Hamilton, US Treasury secretary",
                                "A. Ham",
                                "RB",
                            },
                            "obj_id": "Q30",
                            "obj_label": "United States of America",
                            "obj_aliases": {"the United States of America", "America", "U.S.A.", "USA", "U.S.", "US"},
                            "occurrences": 6,
                            "sentences": {
                                "Alexander Hamilton blah blah blah the United States of America.",
                                "Publius blah blah blah the USA based in Washington, D.C. blah.",
                                "Hamilton blah blah blah United States of America.",
                                "US blah blah blah A. Ham.",
                                "United States of America blah Alexander Hamilton, US Treasury secretary blah blah Washington, D.C. blah.",
                                "RB blah blah blah the USA.",
                            },
                        },
                    },
                    "P_01": {
                        "Q2127993": {
                            "subj_label": "Rainer Bernhardt",
                            "subj_aliases": {"Rainer Herbert Georg Bernhardt", "Bernhardt", "RB"},
                            "obj_id": "Q30",
                            "obj_label": "United States of America",
                            "obj_aliases": {"the United States of America", "America", "U.S.A.", "USA", "U.S.", "US"},
                            "occurrences": 3,
                            "sentences": {
                                "Rainer Herbert Georg Bernhardt blah blah blah the USA blah.",
                                "Bernhardt blah blah blah United States of America.",
                                "RB blah blah blah the USA.",
                            },
                        },
                        "Q38": {
                            "subj_label": "Italy",
                            "subj_aliases": {"IT", "ITA", "Italia", "Italian Republic"},
                            "obj_id": "Q652",
                            "obj_label": "Italian",
                            "obj_aliases": set(),
                            "occurrences": 1,
                            "sentences": {"The Italian (IT) blah blah blah team."},
                        },
                    },
                },
            )

    def test_create_fact_statistics_good_2(self):
        with (
            patch.object(
                FactMatcherSimple,
                "extract_entity_information",
                return_value=self.test_entity_relation_occurrence_info_dict_small,
            ),
            patch.object(
                FactMatcherSimple,
                "_create_mapped_relations",
                return_value=self.test_relation_mapping_dict_small,
            ),
        ):
            fact_matcher = FactMatcherSimple(bear_data_path=f"{self.test_resources_abs_path}")
            fact_matcher.max_ngram = 2

            data = [
                {
                    "text": "kilometres (7,580 sq mi) in the provinces of Limpopo and Mpumalanga in northeastern South Africa, and extends 360 kilometres (220 mi) from north to south and 65 kilometres (40 mi) from east to west."
                },
                {
                    "text": "For two thousand years Arab merchants plied East Africa’s Indian Ocean shores, from Mogadishu (Somalia) to the mouth of the Limpopo River (Mozambique), arriving with the north easterly Kaskazi, departing on the south easterly Kusi.",
                },
                {"text": "Phalaborwa, Limpopo is the only town in South Africa that borders the Kruger National Park."},
                {
                    "text": "The park lies in the north-east of South Africa, in the eastern parts of Limpopo and Mpumalanga provinces."
                },
            ]

            fact_matcher.create_fact_statistics(data, text_key="text", save_file_content=True)

            self.assertEqual(
                fact_matcher.entity_relation_occurrence_info_dict,
                {
                    "P_00": {
                        "Q173017": {
                            "subj_label": "Limpopo River",
                            "subj_aliases": {"Limpopo"},
                            "obj_id": "Q15",
                            "obj_label": "Africa",
                            "obj_aliases": set(),
                            "occurrences": 4,
                            "sentences": {
                                "kilometres (7,580 sq mi) in the provinces of Limpopo and Mpumalanga in northeastern South Africa, and extends 360 kilometres (220 mi) from north to south and 65 kilometres (40 mi) from east to west.",
                                "For two thousand years Arab merchants plied East Africa’s Indian Ocean shores, from Mogadishu (Somalia) to the mouth of the Limpopo River (Mozambique), arriving with the north easterly Kaskazi, departing on the south easterly Kusi.",
                                "Phalaborwa, Limpopo is the only town in South Africa that borders the Kruger National Park.",
                                "The park lies in the north-east of South Africa, in the eastern parts of Limpopo and Mpumalanga provinces.",
                            },
                        }
                    }
                },
            )
