from lm_pub_quiz import Dataset, Evaluator
from utility.utility import load_json_dict


class KnowledgeProber:
    def __init__(self, path_to_occurrence_file: str):
        self.entity_relation_occurrence_info_dict = load_json_dict(path_to_occurrence_file)
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
