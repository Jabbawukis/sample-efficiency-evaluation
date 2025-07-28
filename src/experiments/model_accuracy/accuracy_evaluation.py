import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
import logging
from tqdm import tqdm

import src.experiments.paths as paths
from utility.utility import save_dict_as_json, count_increasing_occurrences_in_slices, load_json_dict
from lm_pub_quiz import DatasetResults


def get_num(x: str) -> int:
    number = x.split("-")[-1]
    return int(number)


class ModelAccuracyEvaluator:
    def __init__(self, models, bear_sizes, abs_path, num_slices=42, num_buckets=14):
        self.models = models
        self.bear_sizes = bear_sizes
        self.abs_path = abs_path
        self.num_slices = num_slices
        self.num_buckets = num_buckets
        self.relation_occurrence_buckets = self._create_occurrence_buckets()

    def _create_occurrence_buckets(self):
        buckets = []
        for i in range(self.num_buckets):
            if i == self.num_buckets - 1:
                buckets.append((2**i, float("inf")))
                break
            buckets.append((2**i, 2 ** (i + 1)))
        return buckets

    def get_model_checkpoint_answers(self, bear_size, model, subset_indices_path=None):
        path_to_checkpoints_probing_results = f"{self.abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-big/{model}/{paths.checkpoints_extracted_wikipedia_20231101_en}"
        output_path_to_increasing_occurrences_in_slices = f"{self.abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.increasing_occurrences_in_slices_wikipedia_20231101_en}"
        path_to_relation_info_on_slices = f"{self.abs_path}/sample-efficiency-evaluation-results/fact_matching_results/BEAR-{bear_size}/{paths.relation_info_on_slices_wikipedia_20231101_en}"

        checkpoints: list = os.listdir(path_to_checkpoints_probing_results)
        sorted_checkpoints = sorted(checkpoints, key=get_num)
        increasing_occurrences = count_increasing_occurrences_in_slices(path_to_relation_info_on_slices)

        for idx, checkpoint in enumerate(tqdm(sorted_checkpoints, desc="Evaluating Probe results in slices")):
            if subset_indices_path:
                subset_indices_dict: dict = load_json_dict(os.path.realpath(subset_indices_path))
                bear_results = DatasetResults.from_path(
                    f"{path_to_checkpoints_probing_results}/{checkpoint}"
                ).filter_subset(subset_indices_dict)
            else:
                bear_results = DatasetResults.from_path(f"{path_to_checkpoints_probing_results}/{checkpoint}")

            for relation_id, entity_dict in increasing_occurrences.items():
                relation_instance_table = bear_results[relation_id].instance_table
                relation_instance_table["correctly_predicted"] = relation_instance_table.apply(
                    lambda row: row.answer_idx == np.argmax(row.pll_scores), axis=1
                )
                for answer_row in relation_instance_table.itertuples():
                    entity_dict[answer_row.sub_id]["occurrences_increase"][idx][
                        "correct"
                    ] = answer_row.correctly_predicted
                    entity_dict[answer_row.sub_id]["occurrences_increase"][idx]["checkpoint"] = checkpoint
                    assert entity_dict[answer_row.sub_id]["occurrences_increase"][idx]["Slice"] == idx

        save_dict_as_json(increasing_occurrences, output_path_to_increasing_occurrences_in_slices)

    def get_checkpoint_accuracy_overall(self, path_to_increasing_occurrences_in_slices: str, num_slices: int = None):
        if num_slices is None:
            num_slices = self.num_slices
        increasing_occurrences = load_json_dict(path_to_increasing_occurrences_in_slices)
        final_output = {}

        for idx in tqdm(range(num_slices), desc="Calculating overall accuracy"):
            correct = 0
            total = 0
            for relation_id, entity_dict in increasing_occurrences.items():
                for entity_id, fact in entity_dict.items():
                    try:
                        assert fact["occurrences_increase"][idx]["Slice"] == idx
                    except (AssertionError, KeyError):
                        logging.warning(f"Warning: slice in dict is not {idx}")
                    if fact["occurrences_increase"][idx]["correct"]:
                        correct += 1
                    total += 1
            final_output[idx] = correct / total
        return final_output

    def get_checkpoint_occurrence_bucket_accuracy(self, path_to_increasing_occurrences_in_slices: str):
        increasing_occurrences = load_json_dict(path_to_increasing_occurrences_in_slices)
        checkpoints = []
        for rel_id, entity_dict in increasing_occurrences.items():
            for ent_id, fact in entity_dict.items():
                checkpoints = [d["checkpoint"] for d in fact["occurrences_increase"]]
                break
            break

        sorted_checkpoints = sorted(checkpoints, key=get_num)
        final_output = {}

        for idx, _checkpoint in enumerate(tqdm(sorted_checkpoints, desc="Calculating bucket accuracy")):
            relation_accuracy_scores_dict = {"0": {"correct": 0, "total": 0}}
            for occurrence in self.relation_occurrence_buckets:
                relation_accuracy_scores_dict[f"{occurrence[0]}-{occurrence[1]}"] = {"correct": 0, "total": 0}

            for relation_id, entity_dict in increasing_occurrences.items():
                for entity_id, fact in entity_dict.items():
                    assert fact["occurrences_increase"][idx]["Slice"] == idx
                    assert fact["occurrences_increase"][idx]["checkpoint"] == _checkpoint
                    occurrences = fact["occurrences_increase"][idx]["total"]
                    if occurrences == 0:
                        relation_accuracy_scores_dict["0"]["total"] += 1
                        if fact["occurrences_increase"][idx]["correct"]:
                            relation_accuracy_scores_dict["0"]["correct"] += 1
                        continue
                    for bucket in self.relation_occurrence_buckets:
                        bucket_start, bucket_end = bucket
                        if bucket_start <= occurrences < bucket_end:
                            relation_accuracy_scores_dict[f"{bucket_start}-{bucket_end}"]["total"] += 1
                            if fact["occurrences_increase"][idx]["correct"]:
                                relation_accuracy_scores_dict[f"{bucket_start}-{bucket_end}"]["correct"] += 1
                            break

            accuracy_scores_output = {}
            for key, bucket in relation_accuracy_scores_dict.items():
                if bucket["total"] == 0:
                    continue
                accuracy_scores_output[key] = {
                    "accuracy": bucket["correct"] / bucket["total"],
                    "correct": bucket["correct"],
                    "total": bucket["total"],
                }
            final_output[_checkpoint] = accuracy_scores_output

        out_put_data = []
        for _checkpoint, buckets in final_output.items():
            for bucket, stats in buckets.items():
                out_put_data.append(
                    {
                        "Checkpoint": _checkpoint,
                        "Frequency Buckets": bucket,
                        "Accuracy": stats["accuracy"],
                        "Frequency": stats["total"],
                    }
                )
        return out_put_data

    def get_checkpoint_occurrence_weighted_accuracy(
        self,
        path_to_increasing_occurrences_in_slices: str,
        weighting_function: callable,
        on_buckets: bool,
        num_slices: int = None,
    ):
        if num_slices:
            self.num_slices = num_slices
        if on_buckets:
            return self._get_weighted_accuracy_on_buckets(path_to_increasing_occurrences_in_slices, weighting_function)
        else:
            return self._get_weighted_accuracy_overall(path_to_increasing_occurrences_in_slices, weighting_function)

    def _get_weighted_accuracy_on_buckets(
        self, path_to_increasing_occurrences_in_slices: str, weighting_function: callable
    ):
        increasing_occurrences = load_json_dict(path_to_increasing_occurrences_in_slices)
        final_output = {}

        for idx in tqdm(range(self.num_slices), desc="Calculating weighted accuracy on buckets"):
            sum_of_weights = []
            relation_accuracy_scores_dict = {"0": {"correct": 0, "total": 0, "lower_bound": 0}}
            for occurrence in self.relation_occurrence_buckets:
                relation_accuracy_scores_dict[f"{occurrence[0]}-{occurrence[1]}"] = {
                    "correct": 0,
                    "total": 0,
                    "lower_bound": occurrence[0],
                }

            for relation_id, entity_dict in increasing_occurrences.items():
                for entity_id, fact in entity_dict.items():
                    try:
                        assert fact["occurrences_increase"][idx]["Slice"] == idx
                    except (AssertionError, KeyError):
                        logging.warning(f"Warning: slice in dict is not {idx}")
                    occurrences = fact["occurrences_increase"][idx]["total"]
                    if occurrences == 0:
                        relation_accuracy_scores_dict["0"]["total"] += 1
                        if fact["occurrences_increase"][idx]["correct"]:
                            relation_accuracy_scores_dict["0"]["correct"] += 1
                        continue
                    for bucket in self.relation_occurrence_buckets:
                        bucket_start, bucket_end = bucket
                        if bucket_start <= occurrences < bucket_end:
                            relation_accuracy_scores_dict[f"{bucket_start}-{bucket_end}"]["total"] += 1
                            if fact["occurrences_increase"][idx]["correct"]:
                                relation_accuracy_scores_dict[f"{bucket_start}-{bucket_end}"]["correct"] += 1
                            break

            accuracy_scores_output = {}
            for occurrence, stats in relation_accuracy_scores_dict.items():
                if stats["total"] == 0:
                    continue
                weight = weighting_function(stats["lower_bound"])
                sum_of_weights.append(weight)
                accuracy_scores_output[occurrence] = (stats["correct"] / stats["total"]) * weight

            sum_of_weights = np.sum(np.array(sum_of_weights))
            sum_of_accuracy_scores = np.sum(np.array([stats for stats in accuracy_scores_output.values()]))
            final_output[idx] = (1 / sum_of_weights) * sum_of_accuracy_scores if sum_of_weights > 0 else 0
        return final_output

    def _get_weighted_accuracy_overall(
        self, path_to_increasing_occurrences_in_slices: str, weighting_function: callable
    ):
        increasing_occurrences = load_json_dict(path_to_increasing_occurrences_in_slices)
        final_output = {}

        for idx in tqdm(range(self.num_slices), desc="Calculating weighted accuracy overall"):
            sum_of_weights = []
            sum_of_weights_total = []
            for relation_id, entity_dict in increasing_occurrences.items():
                for entity_id, fact in entity_dict.items():
                    try:
                        assert fact["occurrences_increase"][idx]["Slice"] == idx
                    except (AssertionError, KeyError):
                        logging.warning(f"Warning: slice in dict is not {idx}")
                    occurrences = fact["occurrences_increase"][idx]["total"]
                    weight = weighting_function(occurrences)
                    sum_of_weights_total.append(weight)
                    if fact["occurrences_increase"][idx]["correct"]:
                        sum_of_weights.append(weight)

            sum_of_weights = np.sum(np.array(sum_of_weights))
            sum_of_weights_total = np.sum(np.array(sum_of_weights_total))
            final_output[idx] = sum_of_weights / sum_of_weights_total if sum_of_weights_total > 0 else 0
        return final_output

    def plot_accuracy_scores(
        self, scores_models: dict, output_path: str, output_diagram_name: str, title: str, ylabel: str
    ):
        plt.figure(figsize=(16, 10))
        plt.xticks(range(0, self.num_slices))

        for _model, model_scores in scores_models.items():
            slices = np.array(list(model_scores.keys()))
            scores = np.array(list(model_scores.values()))
            plt.plot(slices, scores, marker="o", linestyle="-", label=f"{_model}")
            last_x, last_y = slices[-1], scores[-1]
            plt.text(float(last_x), float(last_y), f"{last_y:.4f}", fontsize=12, color="black", ha="left", va="bottom")

        plt.title(title, fontsize=16)
        plt.xlabel("Slices", fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.5)
        plt.tight_layout()
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path, f"{output_diagram_name}.pdf"))
        plt.savefig(os.path.join(output_path, f"{output_diagram_name}.png"))
        plt.clf()
        plt.close()

    def plot_bucket_accuracy(self, _data, _final_diagram_output_path):
        df = pd.DataFrame(_data)
        max_accuracy = df["Accuracy"].max()
        max_occurrences = df["Frequency"].max()
        max_occurrences = math.ceil(max_occurrences / 1000) * 1000

        num_checkpoints = df["Checkpoint"].nunique()
        cols = 5
        rows = math.ceil(num_checkpoints / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(30, 4 * rows), sharey=False)
        axes = axes.flatten()
        checkpoints = sorted(df["Checkpoint"].unique(), key=lambda x: int(x.split("-")[-1]))

        for i, checkpoint in enumerate(checkpoints):
            ax = axes[i]
            checkpoint_data = df[df["Checkpoint"] == checkpoint]
            ax2 = ax.twinx()
            accuracy_plot = sns.barplot(
                data=checkpoint_data, x="Frequency Buckets", y="Accuracy", ax=ax, color="blue", label="Accuracy"
            )
            for p in accuracy_plot.patches:
                value = f"{p.get_height():.2f}"
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    p.get_height(),
                    value,
                    ha="center",
                    va="bottom",
                    color="blue",
                    fontsize=8,
                )

            occurrences_plot = sns.barplot(
                data=checkpoint_data,
                x="Frequency Buckets",
                y="Frequency",
                ax=ax2,
                color="red",
                alpha=0.5,
                label="Frequency",
            )
            for p in occurrences_plot.patches:
                value = f"{int(p.get_height())}"
                ax2.text(
                    p.get_x() + p.get_width() / 2,
                    p.get_height(),
                    value,
                    ha="center",
                    va="bottom",
                    color="red",
                    fontsize=8,
                )

            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_ylabel("Accuracy", color="blue")
            ax2.set_ylabel("Frequency", color="red")
            ax.set_xlabel("Frequency Buckets")
            ax.set_title(f"Checkpoint {checkpoint}")
            ax.set_ylim(0, max_accuracy)
            ax2.set_ylim(0, max_occurrences)

        for j in range(len(checkpoints), len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        output_dir = os.path.dirname(_final_diagram_output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(_final_diagram_output_path)
        plt.clf()
        plt.close()


def weighting_function(occurrences, lambda_=0.05):
    return np.exp(-lambda_ * occurrences) if occurrences > 0 else 0


if __name__ == "__main__":
    models = [
        "gpt2_209m",
        "gpt2_355m",
        "mamba2_172m",
        "mamba2_432m",
        "xlstm_247m",
        "xlstm_406m",
        "llama_208m",
        "llama_360m",
    ]
    bear_sizes = ["big", "small"]
    abs_path = os.path.abspath(os.path.dirname(__file__)).split("sample-efficiency-evaluation")[0]

    evaluator = ModelAccuracyEvaluator(models, bear_sizes, abs_path)

    # Task 1: Get model checkpoint answers
    for bear_size in bear_sizes:
        for model in models:
            subset_path = (
                f"{abs_path}/sample-efficiency-evaluation/BEAR/bear_lite_indices.json" if bear_size == "small" else None
            )
            evaluator.get_model_checkpoint_answers(bear_size, model, subset_path)

    # Task 2: Evaluate and plot overall accuracy
    for bear_size in bear_sizes:
        model_accuracy_on_slices = {}
        output_path = f"{abs_path}/sample-efficiency-evaluation-results/sample_efficiency_measures/accuracy_over_slices/wikimedia_wikipedia_20231101_en/BEAR-{bear_size}/"
        for model in models:
            path_to_data = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.increasing_occurrences_in_slices_wikipedia_20231101_en}"
            model_accuracy_on_slices[model] = evaluator.get_checkpoint_accuracy_overall(path_to_data)

        save_dict_as_json(model_accuracy_on_slices, f"{output_path}/model_scores.json")
        evaluator.plot_accuracy_scores(
            model_accuracy_on_slices,
            output_path,
            f"accuracy_on_slices_bear_{bear_size}",
            "Accuracy Scores Over All Facts",
            "Accuracy Score",
        )

    # Task 3: Evaluate and plot bucket accuracy
    for bear_size in bear_sizes:
        for model in models:
            path_to_data = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.increasing_occurrences_in_slices_wikipedia_20231101_en}"
            output_path = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png"
            bucket_accuracy_data = evaluator.get_checkpoint_occurrence_bucket_accuracy(path_to_data)
            evaluator.plot_bucket_accuracy(bucket_accuracy_data, output_path)

    # Task 4: Evaluate and plot weighted accuracy
    for bear_size in bear_sizes:
        for weight_on_buckets in [True, False]:
            model_weighted_accuracy = {}
            desc = "on_buckets" if weight_on_buckets else "over_all_facts"
            output_path = f"{abs_path}/sample-efficiency-evaluation-results/sample_efficiency_measures/weighted_accuracy_over_slices/wikimedia_wikipedia_20231101_en/BEAR-{bear_size}/{desc}"
            for model in models:
                path_to_data = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.increasing_occurrences_in_slices_wikipedia_20231101_en}"
                model_weighted_accuracy[model] = evaluator.get_checkpoint_occurrence_weighted_accuracy(
                    path_to_data, weighting_function, weight_on_buckets
                )

            save_dict_as_json(model_weighted_accuracy, f"{output_path}/model_scores.json")
            evaluator.plot_accuracy_scores(
                model_weighted_accuracy,
                output_path,
                f"weighted_accuracy_on_slices_bear_{bear_size}",
                "Weighted Accuracy Scores",
                "Weighted Accuracy Score",
            )
