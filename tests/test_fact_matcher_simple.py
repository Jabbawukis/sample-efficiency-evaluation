import logging
import os
import unittest
from unittest.mock import patch, MagicMock

from sample_efficiency_evaluation import FactMatcherSimple
from utility import utility


class FactMatcherSimpleTest(unittest.TestCase):

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
                    "obj_label": "Washington, D.C",
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
                "Q178903": {
                    "subj_label": "Alexander Hamilton",
                    "subj_aliases": {
                        "Publius",
                        "Hamilton",
                        "Alexander Hamilton, US Treasury secretary",
                        "A. Ham",
                        "RB",
                    },
                    "obj_id": "Q2127993",
                    "obj_label": "Rainer Bernhardt",
                    "obj_aliases": {"Rainer Herbert Georg Bernhardt", "Bernhardt", "RB"},
                    "occurrences": 0,
                    "sentences": set(),
                },
            },
        }
        self.test_relation_mapping_dict = {
            "a. ham": {"relations": {("P_00", "Q178903"), ("P_01", "Q178903")}},
            "alexander hamilton": {"relations": {("P_00", "Q178903"), ("P_01", "Q178903")}},
            "alexander hamilton, us treasury secretary": {"relations": {("P_00", "Q178903"), ("P_01", "Q178903")}},
            "america": {"relations": {("P_00", "Q30")}},
            "hamilton": {"relations": {("P_00", "Q178903"), ("P_01", "Q178903")}},
            "publius": {"relations": {("P_00", "Q178903"), ("P_01", "Q178903")}},
            "rainer bernhardt": {"relations": {("P_01", "Q2127993")}},
            "bernhardt": {"relations": {("P_01", "Q2127993")}},
            "rainer herbert georg bernhardt": {"relations": {("P_01", "Q2127993")}},
            "the united states of america": {"relations": {("P_00", "Q30")}},
            "u.s.": {"relations": {("P_00", "Q30")}},
            "u.s.a.": {"relations": {("P_00", "Q30")}},
            "united states of america": {"relations": {("P_00", "Q30")}},
            "us": {"relations": {("P_00", "Q30")}},
            "usa": {"relations": {("P_00", "Q30")}},
            "rb": {"relations": {("P_00", "Q178903"), ("P_01", "Q178903"), ("P_01", "Q2127993")}},
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

            data = [
                {
                    "text": "United States of America blah blah blah Washington, D.C blah."
                    " United States of America blah Alexander blah blah Washington, D.C blah."
                    " United States of America (U.S.A.) blah blah blah Washington, D.C blah."
                },
                {
                    "text": "United of America (U.S.A.) blah blah blah Washington, D.C blah."
                    " Alexander Hamilton blah blah blah the United States of America."
                },
                {
                    "text": "Publius blah blah blah the USA based in Washington, D.C blah."
                    " Hamilton blah blah blah United States of America."
                    " US blah blah blah A. Ham"
                },
                {
                    "text": "Rainer Herbert Georg Bernhardt blah blah blah the USA blah."
                    " Bernhardt blah blah blah United States of America."
                },
                {"text": "Joachim Sauer and Merkel." " A. Merkel blah blah blah Joachim Sauer."},
            ]

            fact_matcher.create_fact_statistics(data, text_key="text")

            self.assertEqual(
                fact_matcher.entity_relation_info_dict,
                {
                    "P_00": {
                        "Q30": {
                            "subj_label": "United States of America",
                            "subj_aliases": {"the United States of America", "America", "U.S.A.", "USA", "U.S.", "US"},
                            "obj_id": "Q61",
                            "obj_label": "Washington, D.C",
                            "obj_aliases": set(),
                            "occurrences": 5,
                            "sentences": {
                                "United States of America blah blah blah Washington, D.C blah.",
                                "United States of America blah Alexander blah blah Washington, D.C blah.",
                                "United States of America (U.S.A.) blah blah blah Washington, D.C blah.",
                                "United of America (U.S.A.) blah blah blah Washington, D.C blah.",
                                "Publius blah blah blah the USA based in Washington, D.C blah.",
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
                            "occurrences": 4,
                            "sentences": {
                                "Alexander Hamilton blah blah blah the United States of America.",
                                "Publius blah blah blah the USA based in Washington, D.C blah.",
                                "Hamilton blah blah blah United States of America.",
                                "US blah blah blah A. Ham",
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
                            "occurrences": 2,
                            "sentences": {
                                "Rainer Herbert Georg Bernhardt blah blah blah the USA blah.",
                                "Bernhardt blah blah blah United States of America.",
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
                            "obj_id": "Q2127993",
                            "obj_label": "Rainer Bernhardt",
                            "obj_aliases": {"Rainer Herbert Georg Bernhardt", "Bernhardt", "RB"},
                            "occurrences": 0,
                            "sentences": set(),
                        },
                    },
                },
            )
