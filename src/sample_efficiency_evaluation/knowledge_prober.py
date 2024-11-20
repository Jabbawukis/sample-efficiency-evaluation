import numpy as np
from lm_pub_quiz import Dataset, Evaluator, DatasetResults
from utility import utility


class KnowledgeProber:
    def __init__(self, path_to_occurrence_file: str):
        self.entity_relation_occurrence_info_dict = utility.load_json_dict(path_to_occurrence_file)
        self.relation_occurrence_buckets = [
            (1, 99),
            (100, 299),
            (300, 499),
            (500, 699),
            (700, 899),
            (900, 999),
            (1000, float("inf")),
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

        :param path_to_bear_results: Path to the BEAR results.
        :return:
        """
        self.bear_results = DatasetResults.from_path(path_to_bear_results)

    def get_accuracy_scores_over_all_relations(self) -> dict:
        """
        Get the accuracy scores over all relations.
        :return: Dictionary containing the accuracy scores.
        """
        relation_occurrence_info_dict = {}
        for occurrence in self.relation_occurrence_buckets:
            relation_occurrence_info_dict[f"{occurrence[0]}-{occurrence[1]}"] = {
                "correct": 0,
                "total": 0,
            }
        relation_occurrence_info_dict["0"] = {"correct": 0, "total": 0}

        for relation_id, relation_dict in self.entity_relation_occurrence_info_dict.items():
            relation_instance_table = self.bear_results[relation_id].instance_table
            relation_instance_table["correctly_predicted"] = relation_instance_table.apply(
                lambda row: row.answer_idx == np.argmax(row.pll_scores), axis=1
            )
            for answer_row in relation_instance_table.itertuples():
                occurrences = relation_dict[answer_row.sub_id]["occurrences"]
                for bucket in self.relation_occurrence_buckets:
                    if occurrences == 0:
                        relation_occurrence_info_dict["0"]["total"] += 1
                        if answer_row.correctly_predicted:
                            relation_occurrence_info_dict["0"]["correct"] += 1
                        break
                    if bucket[0] <= occurrences <= bucket[1]:
                        relation_occurrence_info_dict[f"{bucket[0]}-{bucket[1]}"]["total"] += 1
                        if answer_row.correctly_predicted:
                            relation_occurrence_info_dict[f"{bucket[0]}-{bucket[1]}"]["correct"] += 1
                        break
        for _, bucket in relation_occurrence_info_dict.items():
            if bucket["total"] == 0:
                bucket["accuracy"] = 0
            else:
                bucket["accuracy"] = bucket["correct"] / bucket["total"]
        return relation_occurrence_info_dict
