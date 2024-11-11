import logging
import os
import unittest
from unittest.mock import patch, MagicMock

from sample_efficiency_evaluation import FactMatcherSimple
from utility import utility


class FactMatcherSimpleTest(unittest.TestCase):

    def setUp(self) -> None:
        os.environ["PYTHONHASHSEED"] = "0"
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
            },
        }
        self.test_relation_mapping_dict = {
            "a. ham": {"relations": {("P_00", "Q178903")}},
            "alexander hamilton": {"relations": {("P_00", "Q178903")}},
            "alexander hamilton , us treasury secretary": {"relations": {("P_00", "Q178903")}},
            "america": {"relations": {("P_00", "Q30")}},
            "bernhardt": {"relations": {("P_01", "Q2127993")}},
            "hamilton": {"relations": {("P_00", "Q178903")}},
            "publius": {"relations": {("P_00", "Q178903")}},
            "rainer bernhardt": {"relations": {("P_01", "Q2127993")}},
            "rainer herbert georg bernhardt": {"relations": {("P_01", "Q2127993")}},
            "rb": {"relations": {("P_00", "Q178903"), ("P_01", "Q2127993")}},
            "the united states of america": {"relations": {("P_00", "Q30")}},
            "u.s .": {"relations": {("P_00", "Q30")}},
            "u.s.a .": {"relations": {("P_00", "Q30")}},
            "united states of america": {"relations": {("P_00", "Q30")}},
            "us": {"relations": {("P_00", "Q30")}},
            "usa": {"relations": {("P_00", "Q30")}},
        }
        self.test_entity_relation_info_dict_small = {
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

    def test_create_mapped_relations_good(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict_obj_aliases),
            patch.object(
                utility,
                "extract_entity_information",
                return_value=self.test_entity_relation_info_dict_filled_obj_aliases,
            ),
        ):

            fact_matcher = FactMatcherSimple(
                bear_data_path=f"{self.test_resources_abs_path}",
                save_file_content=True,
            )
            self.assertEqual(fact_matcher.relation_mapping_dict, self.test_relation_mapping_dict)
            self.assertEqual(fact_matcher.max_ngram, 6)

    def test_create_fact_statistics_good(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict_obj_aliases),
            patch.object(
                utility,
                "extract_entity_information",
                return_value=self.test_entity_relation_info_dict_filled_obj_aliases,
            ),
            patch.object(
                FactMatcherSimple,
                "_create_mapped_relations",
                return_value=self.test_relation_mapping_dict,
            ),
        ):

            fact_matcher = FactMatcherSimple(
                bear_data_path=f"{self.test_resources_abs_path}",
                save_file_content=True,
            )
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
                },
                {"text": "Joachim Sauer and Merkel." " A. Merkel blah blah blah Joachim Sauer."},
            ]

            fact_matcher.create_fact_statistics(data, text_key="text")

            self.assertEqual(
                fact_matcher.relation_sentence_dict,
                {
                    "Q178903": {
                        "P_00": {
                            "318ba27127e06e15dadb45df6c3f59c97f7c92a0b164c638cdd3ae4cd2bcc891",
                            "6d12c67e036217d1a56e8ff48221745ca5911fd69dde4500ed66fbce8d32e1fc",
                            "b64eaa32020333c76be1e83b584d32c33a7f250d0ecfe98f46dfc91bd2509fb6",
                            "c973711103f7a50890ef1e3789133b954fc126dc757133dd829a83cd4145a913",
                            "dcc1d27d07414d3daf438f46bb461d804f33bdf6be5e365130476119e5f83761",
                            "f8a98d33b32b0c785b4114a9e2147c26f5cd3b4f921e40a725ff7f58eafde200",
                        }
                    },
                    "Q2127993": {
                        "P_01": {
                            "318ba27127e06e15dadb45df6c3f59c97f7c92a0b164c638cdd3ae4cd2bcc891",
                            "404c6a14471986e2daf51c268da06a0534361c72814896837592c29da31aa946",
                            "fe072923fdc442f126819371b4e387d2785c26c016bf73f5e2dffeefe5b64c8f",
                        }
                    },
                    "Q30": {
                        "P_00": {
                            "167a648132ac233eba91881d133d6a60a573a1ebccd1eb3744b3dce48da2183f",
                            "6ad48005f0c58493ff1f1c11f66b6aea78a2f5b096c16b2e174f42aa6842649b",
                            "6d12c67e036217d1a56e8ff48221745ca5911fd69dde4500ed66fbce8d32e1fc",
                            "dcc1d27d07414d3daf438f46bb461d804f33bdf6be5e365130476119e5f83761",
                            "ed5404cbbbf89aa1d4f13a263fd192e075b95c560b9f08209c2ff2a92afb5442",
                        }
                    },
                },
            )

            self.assertEqual(
                fact_matcher.entity_relation_info_dict,
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
                        }
                    },
                },
            )

    def test_create_fact_statistics_good_2(self):
        with (
            patch.object(utility, "load_json_dict", return_value=self.test_relation_info_dict_obj_aliases),
            patch.object(
                utility,
                "extract_entity_information",
                return_value=self.test_entity_relation_info_dict_small,
            ),
            patch.object(
                FactMatcherSimple,
                "_create_mapped_relations",
                return_value=self.test_relation_mapping_dict_small,
            ),
        ):
            fact_matcher = FactMatcherSimple(
                bear_data_path=f"{self.test_resources_abs_path}",
                save_file_content=True,
            )
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

            fact_matcher.create_fact_statistics(data, text_key="text")

            self.assertEqual(
                fact_matcher.relation_sentence_dict,
                {
                    "Q173017": {
                        "P_00": {
                            "08d8c790c82dbb48a1707ba13f410bbde8e91b5db2a83ed35cebec0064305373",
                            "38012880db7aecd82513360bf492bea368f0ec2bda0ff343ca3d4d40696f5152",
                            "71547cd430be3dc81d52f02dad956f08b96770b76f5ac0f9a293ed3872fb20e2",
                            "b6431d6dca3c86ef3c8e2e72d54b8bdec2ca791ae69956e70729a156646330ec",
                        }
                    }
                },
            )

            self.assertEqual(
                fact_matcher.entity_relation_info_dict,
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
