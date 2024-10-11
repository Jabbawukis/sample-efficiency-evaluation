from datasets import load_dataset
from sample_efficiency_evaluation.fact_matcher import FactMatcherSimpleHeuristic

ds = load_dataset("PatrickHaller/pile-10M-words")

fact_matcher = FactMatcherSimpleHeuristic(bear_data_path="path_to_bear_dataset")

fact_matcher.index_dataset(ds["train"], text_key="text")

results = fact_matcher.search_index("Angela Merkel", sub_query="chancellor")
print(results)