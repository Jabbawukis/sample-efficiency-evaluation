import numpy as np
import os
import matplotlib.pyplot as plt
import info_gathering.paths as paths

from info_gathering.model_performance_analysis.util import (
    get_checkpoint_occurrence_weighted_accuracy,
    get_checkpoint_occurrence_weighted_accuracy_overall,
)


def weighting_function_on_buckets(occurrences, lambda_=0.01, num_buckets=14):
    occurrence_class = 0

    for i in range(num_buckets):
        bucket_start = 2**i
        if i == num_buckets - 1:
            if float("inf") > occurrences >= bucket_start:  # bucket end is exclusive
                occurrence_class = bucket_start
                break
        bucket_end = 2 ** (i + 1)
        if bucket_end > occurrences >= bucket_start:  # bucket end is exclusive
            occurrence_class = bucket_start
            break

    return np.exp(-lambda_ * occurrence_class) if occurrence_class > 0 else 0


def weighting_function(occurrences, lambda_=0.01):
    return np.exp(-lambda_ * occurrences) if occurrences > 0 else 0


def plot_params(weighed_scores_models: dict, output_path: str, output_diagram_name: str):
    plt.figure(figsize=(16, 10))

    bar_width = 0.2  # Adjust bar width for slimmer bars
    x_ticks = np.arange(42)  # Ensure all x-axis values are shown
    plt.xticks(x_ticks)

    num_models = len(weighed_scores_models)
    model_offsets = np.linspace(-bar_width * (num_models / 2), bar_width * (num_models / 2), num_models)

    for idx, (_model, model_scores) in enumerate(weighed_scores_models.items()):
        slices = np.array(list(model_scores.keys()))
        scores = np.array(list(model_scores.values()))

        # Adjust bar positions for multiple models
        bar_positions = slices + model_offsets[idx]

        plt.bar(bar_positions, scores, width=bar_width, label=f"{_model}")

        # Display the score on top of each bar
        for x, y in zip(bar_positions, scores):
            plt.text(x, y + 0.01, f"{y:.4f}", fontsize=12, color="black", ha="center", va="bottom")

    # Add titles, labels, and legend
    plt.title("Accuracy Scores Over All Facts", fontsize=16)
    plt.xlabel("Slices", fontsize=14)
    plt.ylabel("Accuracy Score", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()

    # Save figures
    plt.savefig(os.path.join(output_path, f"{output_diagram_name}.pdf"))
    plt.savefig(os.path.join(output_path, f"{output_diagram_name}.png"))

    plt.clf()
    plt.close()


if __name__ == "__main__":
    models = [
        "gpt2_124m",
        "gpt2_209m",
        "gpt2_355m",
        "mamba2_172m",
        "mamba2_432m",
        "xlstm_247m",
    ]  # results depend on other models
    bear_sizes = ["big", "small"]
    abs_path = os.path.abspath(os.path.dirname(__file__)).split("sample-efficiency-evaluation")[0]
    weight_on_buckets = True
    num_slices = 1
    num_buckets = 14

    relation_occurrence_buckets = []
    for i in range(num_buckets):
        if i == num_buckets - 1:
            relation_occurrence_buckets.append((2**i, float("inf")))
            break
        relation_occurrence_buckets.append((2**i, 2 ** (i + 1)))

    for bear_size in bear_sizes:
        model_weighted_accuracy_on_slices = {}
        final_diagram_output_path = ""
        for model in models:
            path_to_increasing_occurrences_in_slices = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/wikimedia_wikipedia_20231101_en/occurrence_info_0.2_1024_0.8.json"
            if weight_on_buckets:
                final_diagram_output_path = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/weighted_accuracy_over_slices/wikimedia_wikipedia_20231101_en/BEAR-{bear_size}/on_buckets"
                data = get_checkpoint_occurrence_weighted_accuracy(
                    num_slices,
                    path_to_increasing_occurrences_in_slices,
                    weighting_function_on_buckets,
                    relation_occurrence_buckets,
                )
            else:
                path_to_checkpoints_probing_results = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-big/{model}/{paths.checkpoints_extracted_wikipedia_20231101_en}"
                final_diagram_output_path = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/weighted_accuracy_over_slices/wikimedia_wikipedia_20231101_en/BEAR-{bear_size}/over_all_facts"
                data = get_checkpoint_occurrence_weighted_accuracy_overall(
                    path_to_checkpoints_probing_results, path_to_increasing_occurrences_in_slices, weighting_function
                )

            if not os.path.exists(final_diagram_output_path):
                os.makedirs(final_diagram_output_path)

            model_weighted_accuracy_on_slices[model] = data
        plot_params(
            model_weighted_accuracy_on_slices,
            final_diagram_output_path,
            f"weighted_accuracy_on_slices_bear_{bear_size}_0.2_1024_0.8",
        )
