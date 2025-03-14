import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import info_gathering.paths as paths
import math

from info_gathering.model_performance_analysis.util import get_checkpoint_occurrence_bucket_accuracy


def plot_checkpoint_accuracy(_data, _final_diagram_output_path):
    df = pd.DataFrame(_data)

    max_accuracy = df["Accuracy"].max()
    max_occurrences = df["Total Occurrences"].max()

    # Round up the maximum occurrences for better scaling (e.g., to the nearest 1000)
    max_occurrences = math.ceil(max_occurrences / 1000) * 1000

    # Determine grid layout
    num_checkpoints = df["Checkpoint"].nunique()
    cols = 1
    rows = math.ceil(num_checkpoints / cols)

    # Create the grid of plots
    fig, axes = plt.subplots(rows, cols, figsize=(4.5, 3))
    # axes = axes.flatten()

    # Get unique checkpoints for ordering
    checkpoints = sorted(df["Checkpoint"].unique(), key=lambda x: int(x.split("-")[-1]))
    # Iterate over each checkpoint to create individual plots
    for i, checkpoint in enumerate(checkpoints):
        ax = axes
        checkpoint_data = df[df["Checkpoint"] == checkpoint]

        # Create a secondary axis for Total Occurrences
        ax2 = ax.twinx()

        # Plot accuracy on the primary y-axis
        sns.lineplot(
            data=checkpoint_data, x="Occurrence Buckets", y="Accuracy", ax=ax2, color="tab:red", markers=True,
            style="Checkpoint", legend=False, markersize=3
        )

        for x, y in zip(checkpoint_data["Occurrence Buckets"], checkpoint_data["Accuracy"]):
            ax2.text(x, y + 0.02, f"{y:.2f}", ha="center", va="bottom", fontsize=5, color="k")
        # Annotate accuracy ba

        # Plot total occurrences on the secondary y-axis
        occurrences_plot = sns.barplot(
            data=checkpoint_data,
            x="Occurrence Buckets",
            y="Total Occurrences",
            ax=ax,
            color="tab:blue",
            #alpha=0.5,
            #label="Total Occurrences",
        )

        # Annotate total occurrences bars
        for p in occurrences_plot.patches:
            value = f"{int(p.get_height())}"  # Total occurrences is an integer
            ax.text(
                p.get_x() + p.get_width() / 2, 0, value, ha="center", va="bottom", color="k", fontsize=5
            )

        # Rotate x-tick labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right")

        # Set axis labels and titles
        ax2.set_ylabel("Accuracy", color="tab:red")
        ax.set_ylabel("Total Occurrences", color="tab:blue")
        ax2.set_xlabel("Occurrence Buckets")
        # ax.set_title(f"Checkpoint {checkpoint} (Slice {lol[i]})")

        ax2.set_ylim(0, 1.1)
        ax.set_ylim(0, max_occurrences)
    # # Remove unused subplots
    # for j in range(len(checkpoints), len(axes)):
    #     fig.delaxes(axes[j])

    # Adjust layout and save the plot
        fig.tight_layout()
        plt.savefig(_final_diagram_output_path)


if __name__ == "__main__":
    models = [
        # "gpt2_124m",
        # "gpt2_209m",
        # "gpt2_355m",
        # "mamba2_172m",
        "mamba2_432m",
        # "xlstm_247m",
    ]  # results dont depend on other models
    bear_sizes = ["small"]
    abs_path = os.path.abspath(os.path.dirname(__file__)).split("sample-efficiency-evaluation")[0]
    num_buckets = 14

    relation_occurrence_buckets = []
    for i in range(num_buckets):
        if i == num_buckets - 1:
            relation_occurrence_buckets.append((2**i, float("inf")))
            break
        relation_occurrence_buckets.append((2**i, 2 ** (i + 1)))

    for bear_size in bear_sizes:
        for model in models:
            path_to_checkpoints_probing_results = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-big/{model}/{paths.checkpoints_extracted_wikipedia_20231101_en}"
            path_to_increasing_occurrences_in_slices = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.increasing_occurrences_in_slices_wikipedia_20231101_en}"
            final_diagram_output_path = "./mamba2_432_buckets_accuracy_checkpoint-76650.pdf"
            data = get_checkpoint_occurrence_bucket_accuracy(
                path_to_checkpoints_probing_results,
                path_to_increasing_occurrences_in_slices,
                relation_occurrence_buckets,
            )

            plot_checkpoint_accuracy(data, final_diagram_output_path)
