import os
import unittest
from unittest.mock import patch, MagicMock

from sample_efficiency_evaluation import KnowledgeProber
from utility import utility


class KnowledgeProberTest(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = None
        self.test_resources_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_resources"))
        self.entity_relation_result_info_dict = {
            "P6": {
                "Q1519": {
                    "obj_aliases": [],
                    "obj_id": "Q1059948",
                    "obj_label": "Khalifa bin Zayed Al Nahyan",
                    "occurrences": 20,
                    "subj_aliases": ["Abudhabi", "Abū Dhabi"],
                    "subj_label": "Abu Dhabi",
                },
                "Q399": {
                    "obj_aliases": [],
                    "obj_id": "Q7035479",
                    "obj_label": "Nikol Pashinyan",
                    "occurrences": 200,
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
                    "occurrences": 10,
                    "subj_aliases": [],
                    "subj_label": "Gülcemal Sultan",
                },
                "Q837": {
                    "obj_aliases": [],
                    "obj_id": "Q3195923",
                    "obj_label": "Khadga Prasad Sharma Oli",
                    "occurrences": 1001,
                    "subj_aliases": ["Federal Democratic Republic of Nepal", "NEP", "NP", "NPL", "🇳🇵"],
                    "subj_label": "Nepal",
                },
            },
        }

    def test_get_accuracy_scores_over_all_relations_good_1(self):
        row_Q1519 = MagicMock()
        row_Q1519.correctly_predicted = True
        row_Q1519.sub_id = "Q1519"

        row_Q399 = MagicMock()
        row_Q399.correctly_predicted = False
        row_Q399.sub_id = "Q399"

        row_Q548114 = MagicMock()
        row_Q548114.correctly_predicted = True
        row_Q548114.sub_id = "Q548114"

        row_Q5626824 = MagicMock()
        row_Q5626824.correctly_predicted = False
        row_Q5626824.sub_id = "Q5626824"

        row_Q837 = MagicMock()
        row_Q837.correctly_predicted = True
        row_Q837.sub_id = "Q837"

        results_1 = MagicMock()
        results_1_instance_table = MagicMock()
        results_1_instance_table.apply.return_value = None
        results_1_instance_table.itertuples.return_value = [row_Q1519, row_Q399]
        results_1.instance_table = results_1_instance_table

        results_2 = MagicMock()
        results_2_instance_table = MagicMock()
        results_2_instance_table.apply.return_value = None
        results_2_instance_table.itertuples.return_value = [row_Q548114, row_Q5626824, row_Q837]
        results_2.instance_table = results_2_instance_table

        bear_results = {
            "P6": results_1,
            "P2": results_2,
        }

        with patch.object(utility, "load_json_dict", return_value=self.entity_relation_result_info_dict):
            prober = KnowledgeProber("test")
            prober.bear_results = bear_results

            self.assertEqual(
                prober.get_accuracy_scores_over_all_relations(),
                {
                    "0": {"correct": 1, "total": 1, "accuracy": 1.0},
                    "1-99": {"correct": 1, "total": 2, "accuracy": 0.5},
                    "100-299": {"correct": 0, "total": 1, "accuracy": 0.0},
                    "300-499": {"correct": 0, "total": 0, "accuracy": 0.0},
                    "500-699": {"correct": 0, "total": 0, "accuracy": 0.0},
                    "700-899": {"correct": 0, "total": 0, "accuracy": 0.0},
                    "900-999": {"correct": 0, "total": 0, "accuracy": 0.0},
                    "1000-inf": {"correct": 1, "total": 1, "accuracy": 1.0},
                },
            )

    def test_get_accuracy_scores_over_all_relations_good_2(self):
        row_Q1519 = MagicMock()
        row_Q1519.correctly_predicted = True
        row_Q1519.sub_id = "Q1519"

        row_Q399 = MagicMock()
        row_Q399.correctly_predicted = False
        row_Q399.sub_id = "Q399"

        row_Q548114 = MagicMock()
        row_Q548114.correctly_predicted = True
        row_Q548114.sub_id = "Q548114"

        row_Q5626824 = MagicMock()
        row_Q5626824.correctly_predicted = False
        row_Q5626824.sub_id = "Q5626824"

        row_Q837 = MagicMock()
        row_Q837.correctly_predicted = True
        row_Q837.sub_id = "Q837"

        results_1 = MagicMock()
        results_1_instance_table = MagicMock()
        results_1_instance_table.apply.return_value = None
        results_1_instance_table.itertuples.return_value = [row_Q1519, row_Q399]
        results_1.instance_table = results_1_instance_table

        results_2 = MagicMock()
        results_2_instance_table = MagicMock()
        results_2_instance_table.apply.return_value = None
        results_2_instance_table.itertuples.return_value = [row_Q548114, row_Q5626824, row_Q837]
        results_2.instance_table = results_2_instance_table

        bear_results = {
            "P6": results_1,
            "P2": results_2,
        }

        with patch.object(utility, "load_json_dict", return_value=self.entity_relation_result_info_dict):
            prober = KnowledgeProber("test")
            prober.bear_results = bear_results
            prober.relation_occurrence_buckets = [(1, 99), (100, 299), (500, float("inf"))]

            self.assertEqual(
                prober.get_accuracy_scores_over_all_relations(),
                {
                    "0": {"correct": 1, "total": 1, "accuracy": 1.0},
                    "1-99": {"correct": 1, "total": 2, "accuracy": 0.5},
                    "100-299": {"correct": 0, "total": 1, "accuracy": 0.0},
                    "500-inf": {"correct": 1, "total": 1, "accuracy": 1.0},
                },
            )

    def test_get_accuracy_scores_over_all_relations_good_3(self):
        row_Q399 = MagicMock()
        row_Q399.correctly_predicted = False
        row_Q399.sub_id = "Q399"

        row_Q548114 = MagicMock()
        row_Q548114.correctly_predicted = True
        row_Q548114.sub_id = "Q548114"

        row_Q5626824 = MagicMock()
        row_Q5626824.correctly_predicted = False
        row_Q5626824.sub_id = "Q5626824"

        row_Q837 = MagicMock()
        row_Q837.correctly_predicted = True
        row_Q837.sub_id = "Q837"

        results_1 = MagicMock()
        results_1_instance_table = MagicMock()
        results_1_instance_table.apply.return_value = None
        results_1_instance_table.itertuples.return_value = [row_Q399]
        results_1.instance_table = results_1_instance_table

        results_2 = MagicMock()
        results_2_instance_table = MagicMock()
        results_2_instance_table.apply.return_value = None
        results_2_instance_table.itertuples.return_value = [row_Q548114, row_Q5626824, row_Q837]
        results_2.instance_table = results_2_instance_table

        bear_results = {
            "P6": results_1,
            "P2": results_2,
        }

        with patch.object(utility, "load_json_dict", return_value=self.entity_relation_result_info_dict):
            prober = KnowledgeProber("test")
            prober.bear_results = bear_results

            self.assertEqual(
                prober.get_accuracy_scores_over_all_relations(),
                {
                    "0": {"correct": 1, "total": 1, "accuracy": 1.0},
                    "1-99": {"correct": 0, "total": 1, "accuracy": 0.0},
                    "100-299": {"correct": 0, "total": 1, "accuracy": 0.0},
                    "300-499": {"correct": 0, "total": 0, "accuracy": 0.0},
                    "500-699": {"correct": 0, "total": 0, "accuracy": 0.0},
                    "700-899": {"correct": 0, "total": 0, "accuracy": 0.0},
                    "900-999": {"correct": 0, "total": 0, "accuracy": 0.0},
                    "1000-inf": {"correct": 1, "total": 1, "accuracy": 1.0},
                },
            )

    def test_get_accuracy_scores_over_all_relations_good_4(self):
        row_Q548114 = MagicMock()
        row_Q548114.correctly_predicted = True
        row_Q548114.sub_id = "Q548114"

        row_Q5626824 = MagicMock()
        row_Q5626824.correctly_predicted = False
        row_Q5626824.sub_id = "Q5626824"

        row_Q837 = MagicMock()
        row_Q837.correctly_predicted = True
        row_Q837.sub_id = "Q837"

        results_2 = MagicMock()
        results_2_instance_table = MagicMock()
        results_2_instance_table.apply.return_value = None
        results_2_instance_table.itertuples.return_value = [row_Q548114, row_Q5626824, row_Q837]
        results_2.instance_table = results_2_instance_table

        bear_results = {
            "P2": results_2,
        }

        with patch.object(utility, "load_json_dict", return_value=self.entity_relation_result_info_dict):
            prober = KnowledgeProber("test")
            prober.bear_results = bear_results

            self.assertEqual(
                prober.get_accuracy_scores_over_all_relations(),
                {
                    "0": {"correct": 1, "total": 1, "accuracy": 1.0},
                    "1-99": {"correct": 0, "total": 1, "accuracy": 0.0},
                    "100-299": {"correct": 0, "total": 0, "accuracy": 0.0},
                    "300-499": {"correct": 0, "total": 0, "accuracy": 0.0},
                    "500-699": {"correct": 0, "total": 0, "accuracy": 0.0},
                    "700-899": {"correct": 0, "total": 0, "accuracy": 0.0},
                    "900-999": {"correct": 0, "total": 0, "accuracy": 0.0},
                    "1000-inf": {"correct": 1, "total": 1, "accuracy": 1.0},
                },
            )
