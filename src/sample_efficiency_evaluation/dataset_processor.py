import os
from concurrent.futures import ProcessPoolExecutor
from typing import Union
import datasets
from sample_efficiency_evaluation.fact_matcher import FactMatcherHybrid, FactMatcherSimple, FactMatcherEntityLinking
from utility import utility


class DatasetProcessor:
    def __init__(
        self,
        num_slices: int,
        dataset_name: str,
        gpu_id: Union[int, list[int]] = 0,
        bear_data_path: str = "BEAR",
        matcher_type: str = "hybrid",
        read_existing_index: bool = True,
        require_gpu: bool = False,
        entity_linker_model: str = "en_core_web_trf",
        rel_info_output_dir: str = "output",
        dataset_text_key: str = "text",
        output_separately: bool = False,
        save_file_content: bool = False,
    ):
        """
        Initialize the DatasetProcessor with dataset parameters and settings for processing.

        For large datasets, it is recommended to divide the dataset into slices and process each slice in parallel.
        This class provides functionality to process a dataset in parallel using multiple instances of FactMatcher.

        :param num_slices: Number of slices to divide the dataset into, corresponding to the number of parallel workers.
        :param dataset_name: Name of the dataset to load and process.
        :param matcher_type: Type of FactMatcher to use; options are "simple", "hybrid", or "entity_linker".
        :param bear_data_path: Path to the BEAR data directory.
        :param read_existing_index: Flag indicating whether to use an existing index.
        :param require_gpu: Flag indicating whether GPU is required for processing.
        :param gpu_id: GPU ID to use for processing, or a list of GPU IDs, one ID for each Matcher instance
         (e.g. [1, 2, 3, 1]).
        :param entity_linker_model: Model name for entity linking, e.g., "en_core_web_trf".
        :param rel_info_output_dir: Directory path to save the relation information in JSON format.
        :param dataset_text_key: Key to use for text data in the dataset.
        :param output_separately: Flag indicating whether to save relation information separately for each slice.
        :param save_file_content: Flag indicating whether to save the content of the files in the output dictionary.
        """
        self.num_slices = num_slices
        self.dataset_name = dataset_name
        self.bear_data_path = bear_data_path
        self.read_existing_index = read_existing_index
        self.require_gpu = require_gpu
        if isinstance(gpu_id, int):
            self.gpu_id = [gpu_id] * num_slices
        else:
            self.gpu_id = gpu_id
        self.entity_linker_model = entity_linker_model
        self.matcher_type = matcher_type
        self.rel_info_output_dir = rel_info_output_dir
        self.dataset_text_key = dataset_text_key
        self.output_separately = output_separately
        self.save_file_content = save_file_content

        if not os.path.exists(self.rel_info_output_dir):
            os.mkdir(self.rel_info_output_dir)

    def _create_matcher(self, index_dir, gpu_id):
        """
        Create an instance of the appropriate FactMatcher class based on the specified matcher type.

        :param index_dir: Directory path to store the index files for this matcher instance.
        :param gpu_id: GPU ID to use for processing.
        :return: An instance of FactMatcherHybrid, FactMatcherSimple, or FactMatcherEntityLinking based on matcher_type.
        :raises ValueError: If an unknown matcher type is specified.
        """
        if self.matcher_type == "hybrid":
            return FactMatcherHybrid(
                bear_data_path=self.bear_data_path,
                read_existing_index=self.read_existing_index,
                require_gpu=self.require_gpu,
                file_index_dir=index_dir,
                entity_linker_model=self.entity_linker_model,
                save_file_content=self.save_file_content,
                gpu_id=gpu_id,
            )
        if self.matcher_type == "simple":
            return FactMatcherSimple(
                bear_data_path=self.bear_data_path,
                read_existing_index=self.read_existing_index,
                require_gpu=False,
                save_file_content=self.save_file_content,
                file_index_dir=index_dir,
            )
        if self.matcher_type == "entity_linker":
            return FactMatcherEntityLinking(
                bear_data_path=self.bear_data_path,
                read_existing_index=self.read_existing_index,
                require_gpu=self.require_gpu,
                file_index_dir=index_dir,
                entity_linker_model=self.entity_linker_model,
                save_file_content=self.save_file_content,
                gpu_id=gpu_id,
            )

        raise ValueError(f"Unknown matcher type: {self.matcher_type}")

    def _process_slice(self, start_idx, end_idx, index_dir, gpu_id):
        """
        Load and process a slice of the dataset using the specified FactMatcher instance.

        :param start_idx: Start index of the slice.
        :param end_idx: End index of the slice.
        :param index_dir: Directory path to store index files for this data slice.
        :param gpu_id: GPU ID to use for processing.
        :return:
        """
        ds = datasets.load_dataset(self.dataset_name, split=f"train[{start_idx}:{end_idx}]")
        fact_matcher = self._create_matcher(index_dir, gpu_id)

        if not self.read_existing_index:
            fact_matcher.index_dataset(ds, text_key=self.dataset_text_key)
            fact_matcher.close()

        fact_matcher.create_fact_statistics()
        return fact_matcher.entity_relation_info_dict

    def process_dataset(self):
        """
        Process the dataset by dividing it into slices and processing each slice in parallel.

        :return:
        """
        full_dataset = datasets.load_dataset(self.dataset_name, split="train")
        dataset_len = len(full_dataset)
        slice_size = dataset_len // self.num_slices
        slices_info = [
            (
                i * slice_size,  # Start index for the slice
                (
                    dataset_len + 1 if i == self.num_slices - 1 else (i + 1) * slice_size
                ),  # End index (last slice goes to end)
                f".index_dir_{i + 1}",  # Unique index directory
                self.gpu_id[i],  # GPU ID, if applicable
            )
            for i in range(self.num_slices)
        ]
        with ProcessPoolExecutor(max_workers=self.num_slices) as executor:
            futures = [
                executor.submit(self._process_slice, start_idx, end_idx, index_dir, gpu_id)
                for start_idx, end_idx, index_dir, gpu_id in slices_info
            ]
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"An error occurred: {e}")
        if self.output_separately:
            for idx, result in enumerate(results):
                utility.save_json_dict(result, f"{self.rel_info_output_dir}/{idx+1}_relation_info.json")
        else:
            if len(results) > 1:
                for rel_info_dict in results[1:]:
                    for relation_id, relations in rel_info_dict.items():
                        for sub_id, fact in relations.items():
                            results[0][relation_id][sub_id]["occurrences"] += fact["occurrences"]
                            results[0][relation_id][sub_id]["sentences"].update(fact["sentences"])
            utility.save_json_dict(results[0], f"{self.rel_info_output_dir}/relation_info.json")
