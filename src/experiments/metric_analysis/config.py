import os
import experiments.paths as paths

ABS_PATH = os.path.abspath(os.path.dirname(__file__)).split("sample-efficiency-evaluation")[0]
BEAR_SIZES = ["small", "big"]
MODELS = [
    "gpt2_209m",
    "llama_208m",
    "xlstm_247m",
    "mamba2_172m",
    "gpt2_355m",
    "llama_360m",
    "xlstm_406m",
    "mamba2_432m",
]

COLUMNS = {
    "ACC": f"{ABS_PATH}/sample-efficiency-evaluation-results/sample_efficiency_measures/accuracy_over_slices/wikimedia_wikipedia_20231101_en/BEAR-{{bear_size}}/model_scores.json",
    "WAF": f"{ABS_PATH}/sample-efficiency-evaluation-results/sample_efficiency_measures/weighted_accuracy_over_slices/wikimedia_wikipedia_20231101_en/BEAR-{{bear_size}}/over_all_facts/model_scores.json",
    "WASB": f"{ABS_PATH}/sample-efficiency-evaluation-results/sample_efficiency_measures/weighted_accuracy_over_slices/wikimedia_wikipedia_20231101_en/BEAR-{{bear_size}}/on_buckets/model_scores.json",
    "α": f"{ABS_PATH}/sample-efficiency-evaluation-results/probing_results/BEAR-{{bear_size}}/{{model}}/{paths.model_optimized_params_wikipedia_20231101_en}/psf_optimized_alphas.json",
    "λ": f"{ABS_PATH}/sample-efficiency-evaluation-results/probing_results/BEAR-{{bear_size}}/{{model}}/{paths.model_optimized_params_wikipedia_20231101_en}/cdf_optimized_lambdas.json",
}

SUBSET_PERCENTAGE = {
    "big": {"dir": "0.8_0.2_2_0.2_0.8_seed_93_3000", "low": "0.8_2_0.2", "high": "0.2_2_0.8"},
    "small": {"dir": "0.8_0.2_8_0.2_0.8_seed_93_3000", "low": "0.8_8_0.2", "high": "0.2_8_0.8"},
}

NUM_BUCKETS = 14
RELATION_OCCURRENCE_BUCKETS = []
for i in range(NUM_BUCKETS):
    if i == NUM_BUCKETS - 1:
        RELATION_OCCURRENCE_BUCKETS.append((2**i, float("inf")))
        break
    RELATION_OCCURRENCE_BUCKETS.append((2**i, 2 ** (i + 1)))

SEED = 93
SLICES = 42
