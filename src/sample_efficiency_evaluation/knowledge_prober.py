import os
import logging
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from lm_pub_quiz import Dataset, Evaluator, DatasetResults

from utility import utility


class KnowledgeProber:
    """
    KnowledgeProber class to probe the model with BEAR facts and analyze the results.
    """

    def __init__(self, path_to_relation_occurrence_info_file: str, num_buckets: int = 14):
        """
        Initialize the KnowledgeProber.

        :param path_to_relation_occurrence_info_file: Path to the relation occurrence information file.
        :param num_buckets: Number of buckets to divide the relation occurrences into.
        The default is 14. Each bucket is a power of two starting from 1.
        e.g. (1, 2), (2, 4), (4, 8), ... ending with (8192, inf) for 14 buckets.
        """
        self.entity_relation_occurrence_info_dict = utility.load_json_dict(path_to_relation_occurrence_info_file)

        self.relation_occurrence_buckets = []
        for i in range(num_buckets):
            if i == num_buckets - 1:
                self.relation_occurrence_buckets.append((2**i, float("inf")))
                break
            self.relation_occurrence_buckets.append((2**i, 2 ** (i + 1)))

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

    def load_bear_results(self, path_to_bear_results: str, subset_indices: Optional[str] = None) -> None:
        """
        Load the BEAR results.

        :param path_to_bear_results: Path to the BEAR results jsonl files
        :param subset_indices: Indices of the subset to load.
        When probing a model on BEAR-big, the results can be used
        to evaluate the model answers on BEAR(-small) by providing the indices of the subset to load.
        The file in the dataset is called "bear_lite_indices.json".
        :return:
        """
        if subset_indices:
            subset_indices_dict: dict = utility.load_json_dict(subset_indices)
            self.bear_results = DatasetResults.from_path(path_to_bear_results).filter_subset(subset_indices_dict)
        else:
            self.bear_results = DatasetResults.from_path(path_to_bear_results)

    @staticmethod
    def create_fact_accuracy_histogram(
        relation_accuracy_scores_dict: dict, output_path: str, output_diagram_name: str = "accuracy_statistics"
    ) -> None:
        """
        Create fact accuracy statistics and plot a histogram with dual y-axes.
        :param relation_accuracy_scores_dict:  Dictionary containing the entity relation information.
        :param output_path:  Path to save the diagram.
        :param output_diagram_name:  Name of the output diagram.
        :return:
        """

        def get_num(x: str) -> int:
            number = x.split("-")[0]
            return int(number)

        x_labels = sorted(list(relation_accuracy_scores_dict.keys()), key=get_num)
        accuracy_scores = [round(relation_accuracy_scores_dict[x_label]["accuracy"], 2) for x_label in x_labels]
        total_values = [relation_accuracy_scores_dict[x_label]["total"] for x_label in x_labels]
        _, ax1 = plt.subplots(figsize=(5.5, 3.5))

        ax2 = ax1.twinx()
        bars = ax1.bar(x_labels, total_values, color="tab:blue", width=0.8)
        ax1.set_xlabel("Frequency Buckets")
        ax1.set_ylabel("Frequency", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="black")

        for _bar in bars:
            height = _bar.get_height()
            ax1.text(
                _bar.get_x() + _bar.get_width() / 2, 0, f"{height}", ha="center", va="bottom", color="black", fontsize=6
            )

        ax2.plot(x_labels, accuracy_scores, color="tab:red", marker="o", markersize=4)
        ax2.set_ylabel("Accuracy", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="black")
        ax2.set_ylim(0, 1.1)

        for x, y in zip(x_labels, accuracy_scores):
            ax2.text(x, y + 0.02, f"{y:.2f}", ha="center", va="bottom", fontsize=7, color="black")

        ax1.margins(x=0)
        ax1.set_xlim(-0.5, len(x_labels) - 0.5)
        plt.xticks(rotation=45, ha="right")
        ax1.set_xticklabels(x_labels, rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{output_diagram_name}.png"))
        plt.savefig(os.path.join(output_path, f"{output_diagram_name}.pdf"))
        plt.clf()
        plt.close()

    def get_accuracy_scores_over_all_relations(self) -> dict:
        """
        Get the accuracy scores over all relations.
        The bucket end is exclusive.

        :return: The Dictionary containing the accuracy scores.
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
                    bucket_start = bucket[0]
                    bucket_end = bucket[1]
                    if bucket_start <= occurrences < bucket_end:
                        relation_accuracy_scores_dict[f"{bucket_start}-{bucket_end}"]["total"] += 1
                        if answer_row.correctly_predicted:
                            relation_accuracy_scores_dict[f"{bucket_start}-{bucket_end}"]["correct"] += 1
                        break
        accuracy_scores_output = {}
        for key, bucket in relation_accuracy_scores_dict.items():
            if bucket["total"] == 0:
                continue
            accuracy_scores_output[key] = {
                "accuracy": bucket["correct"] / bucket["total"],
                "correct": bucket["correct"],
                "total": bucket["total"],
            }
        return accuracy_scores_output
