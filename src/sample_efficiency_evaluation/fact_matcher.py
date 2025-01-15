import logging

from abc import ABC, abstractmethod
from typing import Union, Optional
from more_itertools import windowed

from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from tqdm import tqdm
from spacy.lang.en import English

from utility import utility
from utility.utility import word_in_sentence, load_json_dict, load_json_line_dict


class FactMatcherBase(ABC):
    """
    FactMatcherBase
    """

    def __init__(self, **kwargs):
        """
        Initialize FactMatcherBase class.
        """

        bear_data_path = kwargs.get("bear_data_path")

        self.entity_relation_occurrence_info_dict: dict = self.extract_entity_information(
            bear_facts_path=kwargs.get("bear_facts_path", f"{bear_data_path}/BEAR"),
            bear_relation_info_path=kwargs.get("bear_relation_info_path", f"{bear_data_path}/relation_info.json"),
            path_to_all_entities=kwargs.get("path_to_all_entities", None),
            exclude_aliases=kwargs.get("exclude_aliases", False),
        )

        self.nlp = English()

        self.tokenizer = self.nlp.tokenizer

        self.nlp.add_pipe("sentencizer")

    def convert_relation_occurrence_info_dict_to_json(self, json_output_file_path: str) -> None:
        """
        Convert relation info dictionary to json file.

        :param json_output_file_path: Path to save the json file.
        :return:
        """
        utility.save_dict_as_json(self.entity_relation_occurrence_info_dict, json_output_file_path)

    def get_tokens_from_sentence(
        self, sentence: str, only_lower: bool = True
    ) -> Union[list[str], tuple[list[str], list[str]]]:
        """
        Get tokens from sentence.
        :param sentence: Sentence
        :param only_lower: Return only lower case tokens
        :return: List of tokens
        """
        if only_lower:
            return [token.orth_ for token in self.tokenizer(sentence.lower())]
        return [token.orth_ for token in self.tokenizer(sentence)], [
            token.orth_ for token in self.tokenizer(sentence.lower())
        ]

    @staticmethod
    def extract_entity_information(
        bear_facts_path: str,
        bear_relation_info_path: str,
        path_to_all_entities: Optional[str] = None,
        exclude_aliases: bool = False,
    ) -> dict:
        """
        Extract entity information from bear data.

        First, the method will load the relation info dictionary.
        Then, it will iterate over the relation info dictionary and load the facts for each relation.
        The entity information will be extracted and stored in a dictionary.
        If the path to all entities is provided, the method will load the additional aliases for the entities
         (for both subject and object).
        Lastly, some more aliases are added to the object aliases by checking if the object id is a subject in another
        relation (in the BEAR relation files e.g. "P6.jsonl", subjects have aliases provided).
        This ensures that each entity has all the aliases from the BEAR data.
        :param bear_facts_path: Path to bear facts directory.
        :param bear_relation_info_path: Path to the BEAR relation info file.
        :param path_to_all_entities: Path to all entities file.
        This file contains additional aliases for the entities.
        :param exclude_aliases: If True, the aliases will not be included in the entity information.
        :return: Relation dictionary
        """
        relation_dict: dict = {}
        bear_relation_info_dict: dict = load_json_dict(bear_relation_info_path)
        all_entities_dict: dict = {}
        if path_to_all_entities:
            all_entities_dict = load_json_dict(path_to_all_entities)
        for relation_key, _ in bear_relation_info_dict.items():
            try:
                fact_list: list[dict] = load_json_line_dict(f"{bear_facts_path}/{relation_key}.jsonl")
                relation_dict.update({relation_key: {}})
            except FileNotFoundError:
                logging.error("File not found: %s/%s.jsonl", bear_facts_path, relation_key)
                continue
            for fact_dict in fact_list:
                relation_dict[relation_key][fact_dict["sub_id"]] = {
                    "subj_label": fact_dict["sub_label"],
                    "subj_aliases": set([] if exclude_aliases else fact_dict["sub_aliases"]),
                    "obj_id": fact_dict["obj_id"],
                    "obj_label": fact_dict["obj_label"],
                    "obj_aliases": set(),
                    "occurrences": 0,
                    "sentences": {},
                }
                if path_to_all_entities and not exclude_aliases:
                    if fact_dict["sub_id"] in all_entities_dict:
                        relation_dict[relation_key][fact_dict["sub_id"]]["subj_aliases"].update(
                            all_entities_dict[fact_dict["sub_id"]]["aliases"]
                        )
                    if fact_dict["obj_id"] in all_entities_dict:
                        relation_dict[relation_key][fact_dict["sub_id"]]["obj_aliases"].update(
                            all_entities_dict[fact_dict["obj_id"]]["aliases"]
                        )
        if not exclude_aliases:
            for _, relations in relation_dict.items():
                for _, fact in relations.items():
                    for _, relations_ in relation_dict.items():
                        try:
                            fact["obj_aliases"].update(relations_[fact["obj_id"]]["subj_aliases"])
                        except KeyError:
                            continue
        return relation_dict

    @abstractmethod
    def create_fact_statistics(
        self,
        file_contents: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
        text_key: str = "text",
        save_file_content: bool = False,
    ) -> None:
        """
        Create fact statistics
        """


class FactMatcherSimple(FactMatcherBase):
    """
    FactMatcherSimple is a class that uses a simple search by string heuristic to search for entities in the dataset.

    kwargs:
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

        - path_to_all_entities [Optional[str]]: Path to all entities extensions json file. This json contains additional
            aliases for the entities. If not provided, it will be set to None (should be "all_entities.json").

        - exclude_aliases [Optional[bool]]: If True, the aliases will not be included in the entity information. The
            default is False.

        - max_allowed_ngram_length [Optional[int]]: Maximum allowed ngram length to search for entities. The sentences
            will be split into ngrams of length 1 to max_allowed_ngram_length. The default is 10.

        - min_entity_name_length [Optional[int]]: Minimum length of the entity name to search for case-insensitive.
            The default is 4. In cases where an entity name is shorter than the min, the search will be case-sensitive.
            This is to avoid matching common words like "is" or "of" (e.g. "US" should not match "us" in "us together").
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.max_ngram = 0

        self.min_entity_name_length = kwargs.get("min_entity_name_length", 4)

        self.max_allowed_ngram_length = kwargs.get("max_allowed_ngram_length", 10)

        self.relation_mapping_dict = self._create_mapped_relations()

        self.match_tracker: set[tuple] = set()

    def _create_mapped_relations(self) -> dict:
        mapped_relations = {}
        for relation_id, relation_info in self.entity_relation_occurrence_info_dict.items():
            for entity_id, entity_info in relation_info.items():
                tokens = self.get_tokens_from_sentence(entity_info["subj_label"])
                tokenized_subj_label = " ".join(tokens)
                if len(tokenized_subj_label) < self.min_entity_name_length:
                    tokenized_subj_label = entity_info["subj_label"]
                if self.max_allowed_ngram_length >= len(tokens) > self.max_ngram:
                    self.max_ngram = len(tokens)
                try:
                    mapped_relations[tokenized_subj_label]["relations"].add((relation_id, entity_id))
                except KeyError:
                    mapped_relations[tokenized_subj_label] = {"relations": {(relation_id, entity_id)}}
                for alias in entity_info["subj_aliases"]:
                    tokens = self.get_tokens_from_sentence(alias)
                    tokenized_subj_label = " ".join(tokens)
                    if len(tokenized_subj_label) < self.min_entity_name_length:
                        tokenized_subj_label = alias
                    if self.max_allowed_ngram_length >= len(tokens) > self.max_ngram:
                        self.max_ngram = len(tokens)
                    try:
                        mapped_relations[tokenized_subj_label]["relations"].add((relation_id, entity_id))
                    except KeyError:
                        mapped_relations[tokenized_subj_label] = {"relations": {(relation_id, entity_id)}}
        return mapped_relations

    def _add_occurrences(self, ngram: str, sentence: str) -> None:
        """
        Add occurrences to the relation dictionary.

        This method will search for the ngram in the relation dictionary and update the occurrences if the object label
        is found in the sentence.
        :param ngram: The ngram to search for.
        :param sentence: The sentence where the ngram was found.
        :return:
        """
        for relation_subj_tuple in self.relation_mapping_dict[ngram]["relations"]:

            relation_id = relation_subj_tuple[0]
            subj_id = relation_subj_tuple[1]
            if (relation_id, subj_id) in self.match_tracker:
                continue

            obj_label: str = self.entity_relation_occurrence_info_dict[relation_id][subj_id]["obj_label"]
            obj_aliases: list[str] = self.entity_relation_occurrence_info_dict[relation_id][subj_id]["obj_aliases"]

            if word_in_sentence(obj_label, sentence, len(obj_label) > self.min_entity_name_length):
                if sentence in self.entity_relation_occurrence_info_dict[relation_id][subj_id]["sentences"]:
                    self.entity_relation_occurrence_info_dict[relation_id][subj_id]["sentences"][sentence] += 1
                else:
                    self.entity_relation_occurrence_info_dict[relation_id][subj_id]["sentences"][sentence] = 1

                count = sum(self.entity_relation_occurrence_info_dict[relation_id][subj_id]["sentences"].values())
                self.entity_relation_occurrence_info_dict[relation_id][subj_id]["occurrences"] = count
                self.match_tracker.add((relation_id, subj_id))
                continue

            for alias in obj_aliases:
                if word_in_sentence(alias, sentence, len(alias) > self.min_entity_name_length):
                    if sentence in self.entity_relation_occurrence_info_dict[relation_id][subj_id]["sentences"]:
                        self.entity_relation_occurrence_info_dict[relation_id][subj_id]["sentences"][sentence] += 1
                    else:
                        self.entity_relation_occurrence_info_dict[relation_id][subj_id]["sentences"][sentence] = 1

                    count = sum(self.entity_relation_occurrence_info_dict[relation_id][subj_id]["sentences"].values())
                    self.entity_relation_occurrence_info_dict[relation_id][subj_id]["occurrences"] = count
                    self.match_tracker.add((relation_id, subj_id))
                    break

    def _process_file_content(self, file_content: str) -> None:
        """
        Process file content.

        This method will split the document into sentences and search for entities in the sentences.
        The occurrences will be updated in the relation dictionary.clea
        clear
        :return:
        """
        content = utility.clean_string(file_content)
        split_doc = self.nlp(content)
        sentences = [sent.text for sent in split_doc.sents]
        for sentence in sentences:
            tokens, tokens_lower = self.get_tokens_from_sentence(sentence, only_lower=False)
            for ngram_size in range(1, self.max_ngram + 1):
                for ngram, ngram_lower in zip(windowed(tokens, ngram_size), windowed(tokens_lower, ngram_size)):
                    try:
                        joined_ngram = " ".join(ngram_lower)
                        if len(joined_ngram) < self.min_entity_name_length:
                            joined_ngram = " ".join(ngram)
                    except TypeError:
                        break
                    if joined_ngram not in self.relation_mapping_dict:
                        continue
                    self._add_occurrences(joined_ngram, sentence)
            self.match_tracker = set()

    def create_fact_statistics(
        self,
        file_contents: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
        text_key: str = "text",
        save_file_content: bool = False,
    ) -> None:
        """
        Create fact statistics.

        This method will iterate over documents and extract sentences.
        It will search for entities in the sentence.
        The occurrences will be updated in the relation dictionary.
        :param text_key: Key to extract text from file content.
        :param file_contents: List of dictionaries containing the file contents.
        :param save_file_content: If True, the content of the file where the entity is found will be saved
        in the relation dictionary.
        :return:
        """
        for file_content in tqdm(file_contents, desc="Processing dataset"):
            self._process_file_content(file_content[text_key])
        if not save_file_content:
            for _, entities in self.entity_relation_occurrence_info_dict.items():
                for _, fact in entities.items():
                    fact["sentences"] = {}
