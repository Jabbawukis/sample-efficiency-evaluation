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
    max_occurrences = df["Frequency"].max()

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
            data=checkpoint_data, x="Frequency Buckets", y="Accuracy", ax=ax, color="blue", label="Accuracy"
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
            x="Frequency Buckets",
            y="Frequency",
            ax=ax2,
            color="red",
            alpha=0.5,
            label="Frequency",
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
        ax2.set_ylabel("Frequency", color="red")
        ax.set_xlabel("Frequency Buckets")
        ax.set_title(f"Checkpoint {checkpoint}")

        ax.set_ylim(0, max_accuracy)
        ax2.set_ylim(0, max_occurrences)
    # Remove unused subplots
    for j in range(len(checkpoints), len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout and save the plot
    fig.tight_layout()
    plt.savefig(_final_diagram_output_path)


if __name__ == "__main__":
    models = [
        "gpt2_124m",
        "gpt2_209m",
        "gpt2_355m",
        "mamba2_172m",
        "mamba2_432m",
        "xlstm_247m",
    ]  # results dont depend on other models
    bear_sizes = ["big", "small"]
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
            final_diagram_output_path = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png"

            data = get_checkpoint_occurrence_bucket_accuracy(
                path_to_checkpoints_probing_results,
                path_to_increasing_occurrences_in_slices,
                relation_occurrence_buckets,
            )
            plot_checkpoint_accuracy(data, final_diagram_output_path)
