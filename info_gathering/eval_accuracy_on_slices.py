import os
from tqdm import tqdm
from utility.utility import load_json_dict

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
    (1024, float("inf")),
]

path_to_checkpoints = "../../sample_efficiency_evaluation_results/probing_results/BEAR-big/xlstm_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices/probing_results_on_checkpoints/checkpoint_extracted/"
path_to_increasing_occurrences_in_slices = "../../sample_efficiency_evaluation_results/probing_results/BEAR-big/xlstm_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices/increasing_occurrences_in_slices.json"
final_diagram_output_path = "../../sample_efficiency_evaluation_results/probing_results/BEAR-big/xlstm_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png"

def get_num(x: str) -> int:
    number = x.split("-")[-1]
    return int(number)

checkpoints: list = os.listdir(path_to_checkpoints)
sorted_checkpoints = sorted(checkpoints, key=get_num)
increasing_occurrences = load_json_dict(path_to_increasing_occurrences_in_slices)
final_output = {}

for idx, checkpoint in enumerate(tqdm(sorted_checkpoints, desc="Creating Diagrams")):

    relation_accuracy_scores_dict = {"0": {"correct": 0, "total": 0}}
    for occurrence in relation_occurrence_buckets:
        relation_accuracy_scores_dict[f"{occurrence[0]}-{occurrence[1]}"] = {
            "correct": 0,
            "total": 0,
        }

    for relation_id, entity_dict in increasing_occurrences.items():
        for entity_id, fact in entity_dict.items():
            assert fact["occurrences_increase"][idx]["Slice"] == idx
            assert fact["occurrences_increase"][idx]["checkpoint"] == checkpoint
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
        accuracy_scores_output[key] = {"accuracy": bucket["correct"] / bucket["total"],
                                       "correct": bucket["correct"],
                                       "total": bucket["total"]}
    final_output[checkpoint] = accuracy_scores_output

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Convert final_output to a DataFrame
data = []
for checkpoint, buckets in final_output.items():
    for bucket, stats in buckets.items():
        data.append({
            "Checkpoint": checkpoint,
            "Occurrence Buckets": bucket,
            "Accuracy": stats["accuracy"],
            "Total Occurrences": stats["total"]
        })

df = pd.DataFrame(data)

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
        data=checkpoint_data,
        x="Occurrence Buckets",
        y="Accuracy",
        ax=ax,
        color="blue",
        label="Accuracy"
    )

    # Annotate accuracy bars
    for p in accuracy_plot.patches:
        value = f"{p.get_height():.2f}"
        ax.text(
            p.get_x() + p.get_width() / 2,
            p.get_height(),
            value,
            ha="center",
            va="bottom",
            color="blue",
            fontsize=8
        )

    # Plot total occurrences on the secondary y-axis
    occurrences_plot = sns.barplot(
        data=checkpoint_data,
        x="Occurrence Buckets",
        y="Total Occurrences",
        ax=ax2,
        color="red",
        alpha=0.5,
        label="Total Occurrences"
    )

    # Annotate total occurrences bars
    for p in occurrences_plot.patches:
        value = f"{int(p.get_height())}"  # Total occurrences is an integer
        ax2.text(
            p.get_x() + p.get_width() / 2,
            p.get_height(),
            value,
            ha="center",
            va="bottom",
            color="red",
            fontsize=8
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
plt.savefig(final_diagram_output_path)
# plt.show()



