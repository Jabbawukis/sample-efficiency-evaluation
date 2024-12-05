import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from lm_pub_quiz import Dataset, Evaluator, DatasetResults

from utility import utility


class KnowledgeProber:
    """
    KnowledgeProber class to probe the model with BEAR facts and analyze the results.
    """

    def __init__(self, path_to_relation_occurrence_info_file: str):
        """
        Initialize the KnowledgeProber.

        :param path_to_relation_occurrence_info_file: Path to the relation occurrence information file.
        """
        self.entity_relation_occurrence_info_dict = utility.load_json_dict(path_to_relation_occurrence_info_file)
        self.relation_occurrence_buckets = [
            (1, 2),
            (2, 4),
            (4, 8),
            (8, 16),
            (16, 32),
            (32, 64),
            (64, 128),
            (128, 256),
            (256, 512),
            (512, 1024),
            (1024, float("inf")),
        ]
        self.bear_results = None

    @staticmethod
    def probe_model(
        model,
        path_to_bear_facts: str,
        result_save_path: str,
        model_type: str = "CLM",
        batch_size: int = 32,
        device: str = "cuda:0",
    ) -> None:
        """
        Probe the model.

        This method probes the model with the given BEAR facts and saves the results.
        :param model: Model to probe.
        :param path_to_bear_facts: Path to the BEAR facts directory.
        This is the directory where all the BEAR fact files (.jsonl) are stored.
        :param result_save_path: Path to save the probing results.
        :param model_type: Type of the model. Default is "CLM".
        :param batch_size: Batch size for probing. Default is 32.
        :param device: Device to run the model on.
        :return:
        """
        dataset = Dataset.from_path(path_to_bear_facts)
        evaluator = Evaluator.from_model(model, model_type=model_type, device=device)
        evaluator.evaluate_dataset(dataset, save_path=result_save_path, batch_size=batch_size)

    def load_bear_results(self, path_to_bear_results: str) -> None:
        """
        Load the BEAR results.

        :param path_to_bear_results: Path to the BEAR results.git stat
        :return:
        """
        self.bear_results = DatasetResults.from_path(path_to_bear_results)

    @staticmethod
    def create_fact_accuracy_histogram(
        relation_accuracy_scores_dict: dict, output_path: str, output_diagram_name: str = "accuracy_statistics"
    ) -> None:
        """
        Create fact accuracy statistics and plot a histogram.
        :param relation_accuracy_scores_dict:  Dictionary containing the entity relation information.
        :param output_path:  Path to save the diagram.
        :param output_diagram_name:  Name of the output diagram.
        :return:
        """

        x_labels = relation_accuracy_scores_dict.keys()
        accuracy_scores = [round(relation_accuracy_scores_dict[x_label]["accuracy"], 2) for x_label in x_labels]
        plt.bar(x_labels, accuracy_scores)
        for i, count in enumerate(accuracy_scores):
            plt.text(i, count, str(count), ha="center", va="bottom")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Occurrence Buckets")
        plt.ylabel("Accuracy")
        plt.title("Entity Pair Accuracy Histogram")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{output_diagram_name}.png"))

    def get_accuracy_scores_over_all_relations(self) -> dict:
        """
        Get the accuracy scores over all relations.
        :return: Dictionary containing the accuracy scores.
        """
        relation_accuracy_scores_dict = {}
        for occurrence in self.relation_occurrence_buckets:
            relation_accuracy_scores_dict[f"{occurrence[0]}-{occurrence[1]}"] = {
                "correct": 0,
                "total": 0,
            }
        relation_accuracy_scores_dict["0"] = {"correct": 0, "total": 0}

        for relation_id, relation_dict in self.entity_relation_occurrence_info_dict.items():
            try:
                relation_instance_table = self.bear_results[relation_id].instance_table
            except KeyError:
                logging.warning("Relation (%s) not found in the BEAR results.", relation_id)
                continue
            relation_instance_table["correctly_predicted"] = relation_instance_table.apply(
                lambda row: row.answer_idx == np.argmax(row.pll_scores), axis=1
            )
            for answer_row in relation_instance_table.itertuples():
                occurrences = relation_dict[answer_row.sub_id]["occurrences"]
                if occurrences == 0:
                    relation_accuracy_scores_dict["0"]["total"] += 1
                    if answer_row.correctly_predicted:
                        relation_accuracy_scores_dict["0"]["correct"] += 1
                    continue
                for bucket in self.relation_occurrence_buckets:
                    if bucket[0] <= occurrences <= bucket[1]:
                        relation_accuracy_scores_dict[f"{bucket[0]}-{bucket[1]}"]["total"] += 1
                        if answer_row.correctly_predicted:
                            relation_accuracy_scores_dict[f"{bucket[0]}-{bucket[1]}"]["correct"] += 1
        for _, bucket in relation_accuracy_scores_dict.items():
            if bucket["total"] == 0:
                bucket["accuracy"] = 0
            else:
                bucket["accuracy"] = bucket["correct"] / bucket["total"]
        return relation_accuracy_scores_dict
