import os
import random
import numpy as np
from tqdm import tqdm
import experiments.paths as paths
from lm_pub_quiz import DatasetResults
from utility.utility import save_dict_as_json, load_json_dict, split_relation_occurrences_info_json_on_occurrences
from experiments.model_accuracy.util import (
    get_checkpoint_accuracy_overall,
    get_checkpoint_occurrence_weighted_accuracy,
    get_checkpoint_occurrence_weighted_accuracy_overall,
)
from experiments.model_accuracy.eval_model_checkpoint_weighted_accuracy_on_slices import (
    weighting_function,
)
from experiments.correct_answer_probability_analysis.psf_nll_optimization import (
    optimize,
)
from experiments.correct_answer_probability_analysis.optimization_utils import (
    get_slice_data,
)
from src.experiments.metric_analysis.config import (
    ABS_PATH,
    MODELS,
    BEAR_SIZES,
    SUBSET_PERCENTAGE,
    RELATION_OCCURRENCE_BUCKETS,
    SEED,
)
from src.experiments.metric_analysis.utils import (
    vectorized_psf,
    vectorized_psf_big,
    plot_scores,
)


def get_model_answer_for_occurrences_in_data(
    path_to_probing_results: str, path_to_relation_info: str, split_info: dict
):
    occurrences = load_json_dict(path_to_relation_info)
    bear_results = DatasetResults.from_path(path_to_probing_results)
    fact_info_dict = {}

    for relation_id, entity_dict in occurrences.items():
        relation_instance_table = bear_results[relation_id].instance_table
        relation_instance_table["correctly_predicted"] = relation_instance_table.apply(
            lambda row: row.answer_idx == np.argmax(row.pll_scores), axis=1
        )

        fact_info_dict[relation_id] = {}

        for answer_row in relation_instance_table.itertuples():
            fact_info = {
                "occurrences_increase": [
                    {
                        "total": entity_dict[answer_row.sub_id]["occurrences"],
                        "correct": answer_row.correctly_predicted,
                    }
                ],
                "obj_id": entity_dict[answer_row.sub_id]["obj_id"],
            }
            fact_info_dict[relation_id][answer_row.sub_id] = fact_info

    final_output = {}
    for split, fact_list in split_info.items():
        relation_info = {}
        split_a_percent, split_b_percent = split

        for fact in fact_list["list"]:
            relation_id, entity_id = fact

            if relation_id not in relation_info:
                relation_info[relation_id] = {}
            relation_info[relation_id][entity_id] = fact_info_dict[relation_id][entity_id]

        final_output[f"{split_a_percent}_{fact_list['threshold']}_{split_b_percent}"] = relation_info
    return final_output


def main():
    psf_funcs = {
        "small": vectorized_psf,
        "big": vectorized_psf_big,
    }
    for bear_size in BEAR_SIZES:
        splits_file_appendix = (
            f"{SUBSET_PERCENTAGE[bear_size]['splits'][0][0]}_"
            f"{SUBSET_PERCENTAGE[bear_size]['splits'][0][1]}_"
            f"{SUBSET_PERCENTAGE[bear_size]['threshold']}_"
            f"{SUBSET_PERCENTAGE[bear_size]['splits'][1][0]}_"
            f"{SUBSET_PERCENTAGE[bear_size]['splits'][1][1]}_seed_{SEED}_"
            f"{SUBSET_PERCENTAGE[bear_size]['total_num_samples']}"
        )
        output_path = f"{ABS_PATH}/sample-efficiency-evaluation-results/sample_efficiency_measures/metric_robustness/wikimedia_wikipedia_20231101_en/BEAR-{bear_size}/{splits_file_appendix}/"
        os.makedirs(output_path, exist_ok=True)
        random.seed(SEED)
        splits = split_relation_occurrences_info_json_on_occurrences(
            f"{ABS_PATH}/sample-efficiency-evaluation-results/fact_matching_results/BEAR-{bear_size}/{paths.relation_occurrence_info_wikipedia_20231101_en}",
            **SUBSET_PERCENTAGE[bear_size],
        )

        model_scores = {"Accuracy": {}, "WASB": {}, "WAF": {}, "α": {}}
        model_alphas_dict = {}
        for split in SUBSET_PERCENTAGE[bear_size]["splits"]:
            model_scores["Accuracy"][f"{split[0]}_{SUBSET_PERCENTAGE[bear_size]['threshold']}_{split[1]}"] = {}
            model_scores["WASB"][f"{split[0]}_{SUBSET_PERCENTAGE[bear_size]['threshold']}_{split[1]}"] = {}
            model_scores["WAF"][f"{split[0]}_{SUBSET_PERCENTAGE[bear_size]['threshold']}_{split[1]}"] = {}
            model_scores["α"][f"{split[0]}_{SUBSET_PERCENTAGE[bear_size]['threshold']}_{split[1]}"] = {}
            model_alphas_dict[f"{split[0]}_{SUBSET_PERCENTAGE[bear_size]['threshold']}_{split[1]}"] = {}

        for model in tqdm(MODELS, desc=f"Evaluating Probe results in BEAR-{bear_size}"):
            probing_results_final_model = f"{ABS_PATH}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.final_model_probing_scores_wikipedia_20231101_en}"
            result = get_model_answer_for_occurrences_in_data(
                path_to_probing_results=probing_results_final_model,
                path_to_relation_info=f"{ABS_PATH}/sample-efficiency-evaluation-results/fact_matching_results/BEAR-{bear_size}/{paths.relation_occurrence_info_wikipedia_20231101_en}",
                split_info=splits,
            )
            samples_path = f"{output_path}/samples"
            os.makedirs(samples_path, exist_ok=True)
            for split, fact_dict in result.items():
                save_dict_as_json(fact_dict, f"{samples_path}/{model}_{split}_bear_{bear_size}.json")
                model_scores["Accuracy"][split][model] = {
                    "on_split": get_checkpoint_accuracy_overall(
                        1, f"{samples_path}/{model}_{split}_bear_{bear_size}.json"
                    )[0],
                    "total": get_checkpoint_accuracy_overall(
                        42,
                        f"{ABS_PATH}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.increasing_occurrences_in_slices_wikipedia_20231101_en}",
                    )[41],
                }
                model_scores["WASB"][split][model] = {
                    "on_split": get_checkpoint_occurrence_weighted_accuracy(
                        1,
                        f"{samples_path}/{model}_{split}_bear_{bear_size}.json",
                        weighting_function,
                        RELATION_OCCURRENCE_BUCKETS,
                    )[0],
                    "total": get_checkpoint_occurrence_weighted_accuracy(
                        42,
                        f"{ABS_PATH}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.increasing_occurrences_in_slices_wikipedia_20231101_en}",
                        weighting_function,
                        RELATION_OCCURRENCE_BUCKETS,
                    )[41],
                }
                model_scores["WAF"][split][model] = {
                    "on_split": get_checkpoint_occurrence_weighted_accuracy_overall(
                        1, f"{samples_path}/{model}_{split}_bear_{bear_size}.json", weighting_function
                    )[0],
                    "total": get_checkpoint_occurrence_weighted_accuracy_overall(
                        42,
                        f"{ABS_PATH}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.increasing_occurrences_in_slices_wikipedia_20231101_en}",
                        weighting_function,
                    )[41],
                }
                model_alphas_dict[split][model] = get_slice_data(
                    1,
                    f"{samples_path}/{model}_{split}_bear_{bear_size}.json",
                )
        for split, model_dict in model_alphas_dict.items():
            optimized_params = optimize(
                model_dict,
                psf_funcs[bear_size],
                1,
            )
            for model, optimized_param in optimized_params.items():
                path_to_total_op_alpha = f"{ABS_PATH}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.model_optimized_params_wikipedia_20231101_en}/psf_optimized_alphas.json"
                total_optimized_alpha = load_json_dict(path_to_total_op_alpha)["Alphas"][41]["alpha"]
                model_scores["α"][split][model] = {
                    "on_split": optimized_param["Alphas"][0]["alpha"],
                    "total": total_optimized_alpha,
                }
        save_dict_as_json(model_scores, f"{output_path}/samples/model_scores.json")
        final_diagram_output_path = f"{output_path}/dias/"
        os.makedirs(final_diagram_output_path, exist_ok=True)
        plot_scores(model_scores, final_diagram_output_path)


if __name__ == "__main__":
    main()
