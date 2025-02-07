import os
from tqdm import tqdm
from utility.utility import load_json_dict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

relation_occurrence_buckets = [
    (1, 2),
    (2, 4),
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
    (256, 512),
    (512, 1024),
    (1024, 2048),
    (2048, 4096),
    (4096, 8192),
    (8192, float("inf")),
]


def get_num(x: str) -> int:
    number = x.split("-")[-1]
    return int(number)


def get_checkpoint_accuracy(_path_to_checkpoints, _path_to_increasing_occurrences_in_slices):
    _checkpoints: list = os.listdir(_path_to_checkpoints)
    sorted_checkpoints = sorted(_checkpoints, key=get_num)
    increasing_occurrences = load_json_dict(_path_to_increasing_occurrences_in_slices)
    final_output = {}

    for idx, _checkpoint in enumerate(tqdm(sorted_checkpoints, desc="Evaluating Probe results in slices")):

        relation_accuracy_scores_dict = {"0": {"correct": 0, "total": 0}}
        for occurrence in relation_occurrence_buckets:
            relation_accuracy_scores_dict[f"{occurrence[0]}-{occurrence[1]}"] = {
                "correct": 0,
                "total": 0,
            }

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
                for bucket in relation_occurrence_buckets:
                    if bucket[0] <= occurrences <= bucket[1]:
                        relation_accuracy_scores_dict[f"{bucket[0]}-{bucket[1]}"]["total"] += 1
                        if fact["occurrences_increase"][idx]["correct"]:
                            relation_accuracy_scores_dict[f"{bucket[0]}-{bucket[1]}"]["correct"] += 1
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

    # Convert final_output to a DataFrame
    out_put_data = []
    for _checkpoint, buckets in final_output.items():
        for bucket, stats in buckets.items():
            out_put_data.append(
                {
                    "Checkpoint": _checkpoint,
                    "Occurrence Buckets": bucket,
                    "Accuracy": stats["accuracy"],
                    "Total Occurrences": stats["total"],
                }
            )
    return out_put_data


def plot_checkpoint_accuracy(_data, _final_diagram_output_path):
    df = pd.DataFrame(_data)

    max_accuracy = df["Accuracy"].max()
    max_occurrences = df["Total Occurrences"].max()

    # Round up the maximum occurrences for better scaling (e.g., to the nearest 1000)
    max_occurrences = math.ceil(max_occurrences / 1000) * 1000

    # Determine grid layout
    num_checkpoints = df["Checkpoint"].nunique()
    cols = 5
    rows = math.ceil(num_checkpoints / cols)

    # Create the grid of plots
    fig, axes = plt.subplots(rows, cols, figsize=(30, 4 * rows), sharey=False)
    axes = axes.flatten()

    # Get unique checkpoints for ordering
    checkpoints = sorted(df["Checkpoint"].unique(), key=lambda x: int(x.split("-")[-1]))

    # Iterate over each checkpoint to create individual plots
    for i, checkpoint in enumerate(checkpoints):
        ax = axes[i]
        checkpoint_data = df[df["Checkpoint"] == checkpoint]

        # Create a secondary axis for Total Occurrences
        ax2 = ax.twinx()

        # Plot accuracy on the primary y-axis
        accuracy_plot = sns.barplot(
            data=checkpoint_data, x="Occurrence Buckets", y="Accuracy", ax=ax, color="blue", label="Accuracy"
        )

        # Annotate accuracy bars
        for p in accuracy_plot.patches:
            value = f"{p.get_height():.2f}"
            ax.text(
                p.get_x() + p.get_width() / 2, p.get_height(), value, ha="center", va="bottom", color="blue", fontsize=8
            )

        # Plot total occurrences on the secondary y-axis
        occurrences_plot = sns.barplot(
            data=checkpoint_data,
            x="Occurrence Buckets",
            y="Total Occurrences",
            ax=ax2,
            color="red",
            alpha=0.5,
            label="Total Occurrences",
        )

        # Annotate total occurrences bars
        for p in occurrences_plot.patches:
            value = f"{int(p.get_height())}"  # Total occurrences is an integer
            ax2.text(
                p.get_x() + p.get_width() / 2, p.get_height(), value, ha="center", va="bottom", color="red", fontsize=8
            )

        # Rotate x-tick labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        # Set axis labels and titles
        ax.set_ylabel("Accuracy", color="blue")
        ax2.set_ylabel("Total Occurrences", color="red")
        ax.set_xlabel("Occurrence Buckets")
        ax.set_title(f"Checkpoint {checkpoint}")

        ax.set_ylim(0, max_accuracy)
        ax2.set_ylim(0, max_occurrences)
    # Remove unused subplots
    for j in range(len(checkpoints), len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout and save the plot
    fig.tight_layout()
    plt.savefig(_final_diagram_output_path)


models = ["gpt2_124m", "gpt2_209m", "mamba2_172m", "xlstm_247m"]
bear_sizes = ["big", "small"]
abs_path = os.path.abspath(os.path.dirname(__file__)).split("sample_efficiency_evaluation")[0]

for bear_size in bear_sizes:
    for model in models:
        path_to_checkpoints_probing_results = f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-big/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/probing_results_on_checkpoints/checkpoint_extracted"
        path_to_increasing_occurrences_in_slices = f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-{bear_size}/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/increasing_occurrences_in_slices.json"
        final_diagram_output_path = f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-{bear_size}/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png"

        data = get_checkpoint_accuracy(path_to_checkpoints_probing_results, path_to_increasing_occurrences_in_slices)
        plot_checkpoint_accuracy(data, final_diagram_output_path)
