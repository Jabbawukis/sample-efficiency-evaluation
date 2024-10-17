"""
FactMatcher
"""

import logging
import os.path
import hashlib
import threading

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from whoosh.index import create_in, open_dir, FileIndex
from whoosh.fields import Schema, TEXT, ID
from whoosh.writing import SegmentWriter
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

        - save_file_content [Optional[bool]]: If True, it will save the file content in the index.
            The result of an index search will contain the file content in the text field
            e.g. {"path": "some_path", "title": "some_title", "text": "some_text"}.
            If False, it will not save the file content. The default is True.
        """
        bear_data_path = kwargs.get("bear_data_path")

        index_path = kwargs.get("file_index_dir", "indexdir")

        self.entity_relation_info_dict: dict = self._extract_entity_information(
            bear_data_path=kwargs.get("bear_facts_path", f"{bear_data_path}/BEAR"),
            bear_relation_info_path=kwargs.get("bear_relation_info_path", f"{bear_data_path}/relation_info.json"),
        )

        if kwargs.get("read_existing_index", False):
            self.writer, self.indexer = self._open_existing_index_dir(index_path)
        else:
            self.writer, self.indexer = self._initialize_index(index_path, kwargs.get("save_file_content", True))

        self.query_parser = QueryParser("text", schema=self.indexer.schema)

        self.nlp_pipeline = English()

        self.nlp_pipeline.add_pipe("sentencizer")

        self.lock = threading.Lock()

    def index_file(self, file_content: str) -> None:
        """
        Index file.
        :param file_content: File content to index.
        :return:
        """
        doc_hash = str(hashlib.sha256(file_content.encode()).hexdigest())
        self.writer.add_document(title=doc_hash, path=f"/{doc_hash}", text=file_content)

    def _index_data_set_file(self, file_content: dict, text_key: str, split_contents_into_sentences: bool) -> None:
        with self.lock:
            content = utility.clean_string(file_content[text_key])
            if split_contents_into_sentences:
                split_doc = self.nlp_pipeline(content)
                sentences = [sent.text for sent in split_doc.sents]
                for sentence in sentences:
                    self.index_file(sentence)
            else:
                self.index_file(content)
            self.commit_index()

    def index_dataset(
        self, file_contents: list[dict], text_key: str = "text", split_contents_into_sentences: bool = True
    ) -> None:
        """
        Index dataset files, the dataset is a list of file contents.
        :param text_key: Key to extract text from file content.
        Since the dataset is a list of dictionaries, we need to
        specify the key to extract the file content.
        That would be the case if we pass a huggingface dataset.
        :param file_contents: List of dictionaries containing the file contents
        :param split_contents_into_sentences: Apply sentence splitting to the file content before indexing.
        :return:
        """

        with ThreadPoolExecutor() as executor:
            for file_content in tqdm(file_contents, desc="Indexing dataset"):
                executor.submit(self._index_data_set_file, file_content, text_key, split_contents_into_sentences)

    def commit_index(self) -> None:
        self.writer.commit()

    def convert_relation_info_dict_to_json(self, file_path: str) -> None:
        """
        Convert relation info dictionary to json file.

        :param file_path: Path to save the json file.
        :return:
        """
        utility.save_json_dict(self.entity_relation_info_dict, file_path)

    @staticmethod
    def _initialize_index(index_path: str, save_file_content: bool) -> tuple[SegmentWriter, FileIndex]:
        """
        Initialize index writer and indexer.
        :param index_path: Path to the index directory to create.
        :param save_file_content: If True, it will save the file content in the index.
        :return:
        """
        indexing_schema = Schema(title=TEXT(stored=True), path=ID(stored=True), text=TEXT(stored=save_file_content))
        if not os.path.exists(index_path):
            os.mkdir(index_path)
        indexer = create_in(index_path, indexing_schema)
        writer = indexer.writer()
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
                relation_dict[relation_key][fact_dict["sub_label"]] = {
                    "aliases": fact_dict["sub_aliases"],
                    "obj_label": fact_dict["obj_label"],
                    "occurrences": 0,
                }
        return relation_dict

    @abstractmethod
    def search_index(self, main_query: str, sub_query: str = "") -> list[dict]:
        """
        Search index for main-query and sub query.

        :param main_query: The main query
        :param sub_query: The sub query
        :return: List of search results
        """

    @abstractmethod
    def create_fact_statistics(self) -> None:
        """
        Create fact statistics

        :return:
        """


class FactMatcherSimpleHeuristic(FactMatcherBase):
    """
    FactMatcherSimpleHeuristic
    """

    def create_fact_statistics(self) -> None:
        """
        Create fact statistics

        This method will iterate over all the relations and facts and search the index for the subject and object.
        It will also search for the aliases of the subject.
        The occurrences will be updated in the relation dictionary.
        :return:
        """
        for relation_key, relation in self.entity_relation_info_dict.items():
            for subj, fact in tqdm(relation.items(), desc=f"Creating fact statistics for {relation_key}"):
                results = self.search_index(subj, fact["obj_label"])
                relation[subj]["occurrences"] += len(results)
                for alias in fact["aliases"]:
                    results = self.search_index(alias, fact["obj_label"])
                    relation[subj]["occurrences"] += len(results)

    def search_index(self, main_query: str, sub_query: str = "") -> list[dict[str, str]]:
        """
        Search index for main-query and sub query.

        If the sub-query is not provided, it will only search for the main query.
        A simple heuristic is used to filter the search results where it is only considered a match if the query is
        found in the content field.
        :param main_query: The main query
        :param sub_query: The sub query
        :return: List of search results
        """
        collected_results = []
        with self.indexer.searcher() as searcher:
            main_q = self.query_parser.parse(main_query)
            if sub_query != "":
                sub_q = self.query_parser.parse(sub_query)
                results = searcher.search(And([main_q, sub_q]))
            else:
                results = searcher.search(main_q)
            for result in results:
                collected_results.append(dict(result))
        return collected_results
