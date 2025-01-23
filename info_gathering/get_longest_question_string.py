from transformers import AutoTokenizer
from sample_efficiency_evaluation import FactMatcherSimple
from utility.utility import load_json_dict

context_length = 128
tokenizer = AutoTokenizer.from_pretrained("gpt2")

bear_path = "../BEAR"


def tokenize(element):
    outputs = tokenizer(
        element,
        return_length=True,
    )
    return outputs["input_ids"]


fact_matcher = FactMatcherSimple(
    bear_data_path=bear_path,
    bear_facts_path=f"{bear_path}/BEAR-big",
    path_to_all_entities=f"{bear_path}/all_entities.json",
)

max_subj_token_length = {"subj": "", "length": 0}
max_obj_token_length = {"obj": "", "length": 0}
relation_id_of_longest_fact = None

for relation_id, relation_info in fact_matcher.entity_relation_occurrence_info_dict.items():
    for entity_id, entity_info in relation_info.items():

        tokens_subj = len(tokenize(entity_info["subj_label"]))
        subj_label = entity_info["subj_label"]

        tokens_obj = len(tokenize(entity_info["obj_label"]))
        obj_label = entity_info["obj_label"]

        for alias in entity_info["subj_aliases"]:
            tokens = len(tokenize(alias))
            if tokens > tokens_subj:
                tokens_subj = tokens
                subj_label = alias

        for alias in entity_info["obj_aliases"]:
            tokens = len(tokenize(alias))
            if tokens > tokens_obj:
                tokens_obj = tokens
                obj_label = alias

        if tokens_subj > max_subj_token_length["length"] and tokens_obj > max_obj_token_length["length"]:
            max_subj_token_length["length"] = tokens_subj
            max_subj_token_length["subj"] = subj_label
            max_obj_token_length["length"] = tokens_obj
            max_obj_token_length["obj"] = obj_label
            relation_id_of_longest_fact = relation_id

print(max_subj_token_length)
print(max_obj_token_length)
print(relation_id_of_longest_fact)

templates = load_json_dict(f"{bear_path}/BEAR-big/metadata_relations.json")[relation_id_of_longest_fact]["templates"]

max_tokens_sentence = {"sentence": "", "length": 0}
for template in templates:
    sentence = template.replace("[X]", max_subj_token_length["subj"]).replace("[Y]", max_obj_token_length["obj"])
    tokens = len(tokenize(sentence))
    if tokens > max_tokens_sentence["length"]:
        max_tokens_sentence["length"] = tokens
        max_tokens_sentence["sentence"] = sentence
print(max_tokens_sentence)
