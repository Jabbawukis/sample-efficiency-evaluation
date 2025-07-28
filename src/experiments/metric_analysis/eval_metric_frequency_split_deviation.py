import os
import numpy as np
from utility.utility import load_json_dict
from src.experiments.metric_analysis.config import BEAR_SIZES, SUBSET_PERCENTAGE, ABS_PATH
from src.experiments.metric_analysis.utils import plot_diverging_bars_rel


def main():
    splits = {"low": [], "high": []}
    for bear_size in BEAR_SIZES:
        columns_rel = {"Accuracy": [], "WASB": [], "WAF": [], "α": []}
        columns_abs = {"Accuracy": [], "WASB": [], "WAF": [], "α": []}
        path_to_split_results = f"{ABS_PATH}/sample-efficiency-evaluation-results/sample_efficiency_measures/metric_robustness/wikimedia_wikipedia_20231101_en/BEAR-{bear_size}/{SUBSET_PERCENTAGE[bear_size]['dir']}"
        plot_output = f"{path_to_split_results}/split_difference/"
        os.makedirs(plot_output, exist_ok=True)
        model_scores = load_json_dict(f"{path_to_split_results}/samples/model_scores.json")
        for split, _ in splits.items():
            for metric_name, _ in columns_rel.items():
                score_total = []
                split_list = []
                for model_name, model_dict in model_scores[metric_name][SUBSET_PERCENTAGE[bear_size][split]].items():
                    split_list.append(model_dict["on_split"])
                    score_total.append(model_dict["total"])

                rel_dev = np.subtract(split_list, score_total)
                rel_dev = np.divide(rel_dev, score_total)
                rel_dev = np.mean(rel_dev)
                columns_rel[metric_name].append(rel_dev)

                abs_dev = np.subtract(split_list, score_total)
                abs_dev = np.mean(abs_dev)
                columns_abs[metric_name].append(abs_dev)
        plot_diverging_bars_rel(
            columns_rel, f"{plot_output}/metric_frequency_split_rel_difference_{bear_size}", "Mean Relative"
        )
        plot_diverging_bars_rel(
            columns_abs, f"{plot_output}/metric_frequency_split_abs_difference_{bear_size}", "Mean Absolute"
        )


if __name__ == "__main__":
    main()
