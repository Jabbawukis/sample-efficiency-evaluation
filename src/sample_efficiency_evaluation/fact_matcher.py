"""
FactMatcher
"""

import logging
import os.path
import hashlib

from abc import ABC, abstractmethod
from typing import Union

import spacy
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from tqdm import tqdm
from whoosh.index import create_in, exists_in, open_dir, FileIndex
from whoosh.fields import Schema, TEXT, ID
from whoosh.searching import Searcher
from whoosh.writing import SegmentWriter, BufferedWriter
from whoosh.qparser import QueryParser
from whoosh.query import And
from spacy.lang.en import English

from utility import utility


class FactMatcherBase(ABC):
    """
    FactMatcher
    """

    def __init__(self, **kwargs):
        """
        Initialize FactMatcher

        :param kwargs:
        - bear_data_path [str]: Path to bear data directory.
            This is the main directory where all the bear data is stored. It should contain the relation_info.json file
            and the BEAR facts directory.

        - bear_relation_info_path [Optional[str]]: Path to the BEAR relation info file.
            This file contains the relation information for the BEAR data. If not provided, it will be set to
            {bear_data_path}/relation_info.json.

        - bear_facts_path [Optional[str]]: Path to the BEAR facts directory.
            This is the directory where all the BEAR fact files (.jsonl) are stored. If not provided, it will be set to
            {bear_data_path}/BEAR. Note that the dataset provides a BEAR and a BEAR-big directory, with the latter
            containing more facts.

        - file_index_dir [Optional[str]]: Path to the index directory.
            This is the directory where the index will be stored.
            If not provided, it will be set to "indexdir".

        - read_existing_index [Optional[bool]]: If True, it will read the existing index. If False, it will create a new
            index. The index file provided in the file_index_dir argument will be used to read the existing index.
            If the file_index_dir argument is not set, the default "indexdir" will be used.

        - save_file_content [Optional[bool]]: If True, the content of the file where the entity is found will be saved
            in the relation dictionary. The default is False.

        - require_gpu [Optional[bool]]: If True, it will require a GPU for the spacy entity linker.
            The default is False.

        - gpu_id [Optional[int]]: GPU ID to use for the spacy entity linker.
            The default is 0.

        - entity_linker_model [str]: Entity linker model to use.
            The default is "en_core_web_md", which is a medium-sized model optimized for cpu.
            Refer to https://spacy.io/models/en for more information.
        """

        bear_data_path = kwargs.get("bear_data_path")

        self.index_path = kwargs.get("file_index_dir", "indexdir")

        self.save_file_content = kwargs.get("save_file_content", False)

        self.entity_relation_info_dict: dict = self._extract_entity_information(
            bear_data_path=kwargs.get("bear_facts_path", f"{bear_data_path}/BEAR"),
            bear_relation_info_path=kwargs.get("bear_relation_info_path", f"{bear_data_path}/relation_info.json"),
        )

        self.read_existing_index = kwargs.get("read_existing_index", False)
        if self.read_existing_index:
            self.writer, self.indexer = self._open_existing_index_dir(self.index_path)
        else:
            self.writer, self.indexer = self._initialize_index(self.index_path)

        self.query_parser = QueryParser("text", schema=self.indexer.schema)

        self.sentencizer = English()

        self.sentencizer.add_pipe("sentencizer")

    def _index_file(self, file_content: str) -> None:
        """
        Index file.

        Call the close() method after finishing indexing the files.
        :param file_content: File content to index.
        :return:
        """
        doc_hash = str(hashlib.sha256(file_content.encode()).hexdigest())
        self.writer.add_document(title=doc_hash, path=f"/{doc_hash}", text=file_content)

    def search_index(
        self, main_query: str, sub_query: str = "", searcher_passed: Searcher = None
    ) -> list[dict[str, str]]:
        """
        Search index for main-query and sub query.

        If the sub-query is not provided, it will only search for the main query.
        A simple heuristic is used to filter the search results where it is only considered a match if the query is
        found in the content field.
        :param main_query: The main query
        :param sub_query: The sub query
        :param searcher_passed: Searcher object
        :return: List of search results
        """
        collected_results = []
        searcher = searcher_passed
        if searcher is None:
            searcher = self.writer.searcher()
        main_q = self.query_parser.parse(main_query)
        if sub_query != "":
            sub_q = self.query_parser.parse(sub_query)
            results = searcher.search(And([main_q, sub_q]))
        else:
            results = searcher.search(main_q)
        for result in results:
            collected_results.append(dict(result))
        if searcher_passed is None:
            searcher.close()
        return collected_results

    def close(self) -> None:
        """
        Close the writer.

        This method should be called after finishing indexing the files, and/or searching the index.
        :return:
        """
        try:
            self.writer.close()
        except AttributeError:
            logging.error("Using SegmentWriter instead of BufferedWriter")

    def convert_relation_info_dict_to_json(self, file_path: str) -> None:
        """
        Convert relation info dictionary to json file.

        :param file_path: Path to save the json file.
        :return:
        """
        utility.save_json_dict(self.entity_relation_info_dict, file_path)

    @staticmethod
    def _initialize_index(index_path: str) -> tuple[BufferedWriter, FileIndex]:
        """
        Initialize index writer and indexer.
        :param index_path: Path to the index directory to create.
        :return:
        """
        indexing_schema = Schema(title=TEXT(stored=True), path=ID(stored=True), text=TEXT(stored=True))
        if not os.path.exists(index_path):
            os.mkdir(index_path)
        indexer = create_in(index_path, indexing_schema)
        writer = BufferedWriter(indexer, period=None)
        return writer, indexer

    @staticmethod
    def _open_existing_index_dir(index_path) -> tuple[SegmentWriter, FileIndex]:
        """
        Open an already existing index directory and return writer and indexer.

        If the index directory does not exist, it will raise an error.
        Within the directory, there should be one index file.
        :param index_path: Path to an existing index directory.
        :return: Writer and indexer.
        """
        if not exists_in(index_path):
            raise FileNotFoundError(f"Index directory not found: {index_path}")
        indexer = open_dir(index_path)
        writer = indexer.writer()
        return writer, indexer

    @staticmethod
    def _extract_entity_information(bear_data_path: str, bear_relation_info_path: str) -> dict:
        """
        Extract entity information from bear data.
        :param bear_data_path: Path to bear facts directory.
        :return: Relation dictionary
        """
        relation_dict: dict = {}
        bear_relation_info_dict: dict = utility.load_json_dict(bear_relation_info_path)
        for relation_key, _ in bear_relation_info_dict.items():
            try:
                fact_list: list[dict] = utility.load_json_line_dict(f"{bear_data_path}/{relation_key}.jsonl")
                relation_dict.update({relation_key: {}})
            except FileNotFoundError:
                logging.error("File not found: %s/%s.jsonl", bear_data_path, relation_key)
                continue
            for fact_dict in fact_list:
                logging.info("Extracting entity information for %s", relation_key)
                relation_dict[relation_key][fact_dict["sub_id"]] = {
                    "subj_label": fact_dict["sub_label"],
                    "subj_aliases": set(fact_dict["sub_aliases"]),
                    "obj_id": fact_dict["obj_id"],
                    "obj_label": fact_dict["obj_label"],
                    "obj_aliases": set(),
                    "occurrences": 0,
                    "sentences": set(),
                }
        for _, relations in relation_dict.items():
            for _, fact in relations.items():
                for _, relations_ in relation_dict.items():
                    try:
                        fact["obj_aliases"].update(relations_[fact["obj_id"]]["subj_aliases"])
                    except KeyError:
                        continue
        return relation_dict

    @abstractmethod
    def create_fact_statistics(self) -> None:
        """
        Create fact statistics and close the writer.

        :return:
        """

    @abstractmethod
    def index_dataset(
        self,
        file_contents: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
        text_key: str = "text",
        split_contents_into_sentences: bool = True,
    ) -> None:
        """
        Index dataset files, the dataset is a list of file contents.

        :return:
        """


class FactMatcherHybrid(FactMatcherBase):
    """
    FactMatcherHybrid

    FactMatcherHybrid is a class that uses a simple search by string heuristic to search for entities in the dataset.
    The dataset is searched and indexed on a document level.
    If a document contains a relation subject and object,
    the document is split into sentences
    and the entitiy linker model is used to search for entities in the sentence.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if kwargs.get("require_gpu", False):
            spacy.require_gpu(kwargs.get("gpu_id", 0))
        else:
            spacy.prefer_gpu(kwargs.get("gpu_id", 0))

        self.entity_linker = spacy.load(kwargs.get("entity_linker_model", "en_core_web_md"))

        self.entity_linker.add_pipe("entityLinker", last=True)

    def _get_entity_ids(self, content: str) -> set:
        """
        Get entity IDs from content.

        :param content: Content to extract entity IDs from.
        :return: Entity IDs
        """
        return self.entity_linker(content)._.linkedEntities

    def index_dataset(
        self,
        file_contents: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
        text_key: str = "text",
        split_contents_into_sentences: bool = False,
    ) -> None:
        """
        Index dataset files, the dataset is a list of file contents.

        Call the close() method after finishing indexing the files.
        :param text_key: Key to extract text from file content.
        Since the dataset is a list of dictionaries, we need to
        specify the key to extract the file content.
        That would be the case if we pass a huggingface dataset.
        :param file_contents: List of dictionaries containing the file contents
        :param split_contents_into_sentences: Apply sentence splitting to the file content before indexing.
        It Has no effect on this method.
        :return:
        """
        for file_content in tqdm(file_contents, desc="Indexing dataset"):
            content = utility.clean_string(file_content[text_key])
            self._index_file(content)

    def _search_for_entities_by_id_and_string(
        self, hits: list[dict], subj_id: str, obj_id: str, subj_label: str, obj_label: str
    ) -> set[str]:
        sent_with_occurrences = set()
        for hit in hits:
            content = hit["text"]
            split_doc = self.sentencizer(content)
            sentences = [sent.text for sent in split_doc.sents]
            for sentence in sentences:
                all_linked_entities = self._get_entity_ids(sentence)
                entity_ids = [f"Q{str(linked_entity.get_id())}" for linked_entity in all_linked_entities]
                if subj_id in entity_ids and obj_id in entity_ids:
                    sent_with_occurrences.update([sentence])
                elif subj_label in sentence and obj_label in sentence:
                    sent_with_occurrences.update([sentence])
        return sent_with_occurrences

    def create_fact_statistics(self) -> None:
        """
        Create fact statistics

        This method will iterate over all the relations and facts and search the index for the subject and object.
        If a document contains a relation subject and object, the document is split into sentences and the entity linker
        model is used to search for entities in the sentence (in combination with simple string search.).
        Additionally, it will cross-search for the aliases of the
        subject and object.
        The occurrences will be updated in the relation dictionary.
        :return:
        """
        if not self.read_existing_index:
            self.writer, self.indexer = self._open_existing_index_dir(self.index_path)
        with self.indexer.searcher() as searcher:
            for relation_key, relation in self.entity_relation_info_dict.items():
                for subj_id, fact in tqdm(relation.items(), desc=f"Creating fact statistics for {relation_key}"):
                    collected_results = set()
                    results = self.search_index(fact["subj_label"], fact["obj_label"], searcher)
                    occurrences = self._search_for_entities_by_id_and_string(
                        results, subj_id, fact["obj_id"], fact["subj_label"], fact["obj_label"]
                    )
                    collected_results.update(occurrences)

                    for alias in fact["obj_aliases"]:
                        results = self.search_index(fact["subj_label"], alias, searcher)
                        occurrences = self._search_for_entities_by_id_and_string(
                            results, subj_id, fact["obj_id"], fact["subj_label"], alias
                        )
                        collected_results.update(occurrences)

                    for alias in fact["subj_aliases"]:
                        results = self.search_index(alias, fact["obj_label"])
                        occurrences = self._search_for_entities_by_id_and_string(
                            results, subj_id, fact["obj_id"], alias, fact["obj_label"]
                        )
                        collected_results.update(occurrences)

                    for subj_aliases in fact["subj_aliases"]:
                        for obj_aliases in fact["obj_aliases"]:
                            results = self.search_index(subj_aliases, obj_aliases)
                            occurrences = self._search_for_entities_by_id_and_string(
                                results, subj_id, fact["obj_id"], subj_aliases, obj_aliases
                            )
                            collected_results.update(occurrences)
                    fact["occurrences"] += len(collected_results)
                    if self.save_file_content:
                        fact["sentences"].update(collected_results)


class FactMatcherEntityLinking(FactMatcherBase):
    """
    FactMatcherEntityLinking

    FactMatcherEntityLinking is a class that uses the entity linker model to index and search the dataset.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if kwargs.get("require_gpu", False):
            spacy.require_gpu(kwargs.get("gpu_id", 0))
        else:
            spacy.prefer_gpu(kwargs.get("gpu_id", 0))

        self.entity_linker = spacy.load(kwargs.get("entity_linker_model", "en_core_web_md"))

        self.entity_linker.add_pipe("entityLinker", last=True)

    def _get_entity_ids(self, content: str) -> set:
        """
        Get entity IDs from content.

        :param content: Content to extract entity IDs from.
        :return: Entity IDs
        """
        return self.entity_linker(content)._.linkedEntities

    def index_dataset(
        self,
        file_contents: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
        text_key: str = "text",
        split_contents_into_sentences: bool = True,
    ) -> None:
        """
        Index dataset files, the dataset is a list of file contents.

        Call the close() method after finishing indexing the files.
        :param text_key: Key to extract text from file content.
        Since the dataset is a list of dictionaries, we need to
        specify the key to extract the file content.
        That would be the case if we pass a huggingface dataset.
        :param file_contents: List of dictionaries containing the file contents
        :param split_contents_into_sentences: Apply sentence splitting to the file content before indexing.
        :return:
        """
        for file_content in tqdm(file_contents, desc="Indexing dataset"):
            content = utility.clean_string(file_content[text_key])
            if split_contents_into_sentences:
                split_doc = self.sentencizer(content)
                sentences = [sent.text for sent in split_doc.sents]
                for sentence in sentences:
                    all_linked_entities = self._get_entity_ids(sentence)
                    sentence = utility.decorate_sentence_with_ids(sentence, all_linked_entities)
                    self._index_file(sentence)
            else:
                all_linked_entities = self._get_entity_ids(content)
                content = utility.decorate_sentence_with_ids(content, all_linked_entities)
                self._index_file(content)

    def create_fact_statistics(self) -> None:
        if not self.read_existing_index:
            self.writer, self.indexer = self._open_existing_index_dir(self.index_path)
        with self.indexer.searcher() as searcher:
            for relation_key, relation in self.entity_relation_info_dict.items():
                for subj_id, fact in tqdm(relation.items(), desc=f"Creating fact statistics for {relation_key}"):
                    collected_results = set()
                    sentences = set()
                    results = self.search_index(subj_id, fact["obj_id"], searcher)
                    collected_results.update([result["title"] for result in results])
                    sentences.update([result["text"] for result in results])
                    fact["occurrences"] += len(collected_results)
                    if self.save_file_content:
                        fact["sentences"].update(sentences)


class FactMatcherSimple(FactMatcherBase):
    """
    FactMatcherSimple

    FactMatcherSimple is a class that uses a simple search by string heuristic to search for entities in the dataset.
    """

    def index_dataset(
        self,
        file_contents: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
        text_key: str = "text",
        split_contents_into_sentences: bool = True,
    ) -> None:
        """
        Index dataset files, the dataset is a list of file contents.

        Call the close() method after finishing indexing the files.
        :param text_key: Key to extract text from file content.
        Since the dataset is a list of dictionaries, we need to
        specify the key to extract the file content.
        That would be the case if we pass a huggingface dataset.
        :param file_contents: List of dictionaries containing the file contents
        :param split_contents_into_sentences: Apply sentence splitting to the file content before indexing.
        :return:
        """
        for file_content in tqdm(file_contents, desc="Indexing dataset"):
            content = utility.clean_string(file_content[text_key])
            if split_contents_into_sentences:
                split_doc = self.sentencizer(content)
                sentences = [sent.text for sent in split_doc.sents]
                for sentence in sentences:
                    self._index_file(sentence)
            else:
                self._index_file(content)

    def create_fact_statistics(self) -> None:
        """
        Create fact statistics

        This method will iterate over all the relations and facts and search the index for the subject and object.
        It will also cross-search for the aliases of the subject and object.
        The occurrences will be updated in the relation dictionary.
        :return:
        """
        if not self.read_existing_index:
            self.writer, self.indexer = self._open_existing_index_dir(self.index_path)
        with self.indexer.searcher() as searcher:
            for relation_key, relation in self.entity_relation_info_dict.items():
                for _, fact in tqdm(relation.items(), desc=f"Creating fact statistics for {relation_key}"):
                    collected_results = set()
                    sentences = set()
                    results = self.search_index(fact["subj_label"], fact["obj_label"], searcher)
                    collected_results.update([result["title"] for result in results])
                    sentences.update([result["text"] for result in results])

                    for alias in fact["obj_aliases"]:
                        results = self.search_index(fact["subj_label"], alias, searcher)
                        collected_results.update([result["title"] for result in results])
                        sentences.update([result["text"] for result in results])

                    for alias in fact["subj_aliases"]:
                        results = self.search_index(alias, fact["obj_label"])
                        collected_results.update([result["title"] for result in results])
                        sentences.update([result["text"] for result in results])

                    for subj_aliases in fact["subj_aliases"]:
                        for obj_aliases in fact["obj_aliases"]:
                            results = self.search_index(subj_aliases, obj_aliases)
                            collected_results.update([result["title"] for result in results])
                            sentences.update([result["text"] for result in results])
                    fact["occurrences"] += len(collected_results)
                    if self.save_file_content:
                        fact["sentences"].update(sentences)
