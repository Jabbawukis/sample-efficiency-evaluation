import os
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from utility.utility import load_json_dict

path_to_checkpoints_probing_results = "../../sample_efficiency_evaluation_results/probing_results/BEAR-big/gpt2_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices/probing_results_on_checkpoints/checkpoint_extracted"
path_to_increasing_occurrences_in_slices = "../../sample_efficiency_evaluation_results/fact_matching_results/BEAR-big/wikimedia_wikipedia_20231101_en/evaluation_on_slices/increasing_occurrences_in_slices.json"


def get_num(x: str) -> int:
    """Extract numerical suffix from a string."""
    number = x.split("-")[-1]
    return int(number)


def compute_pi(alpha, b_i, c_i):
    """Compute the probability of a correct response (p_i)."""
    return c_i + (1 - c_i) * (1 / (1 + np.exp(-alpha * b_i)))


def compute_log_likelihood(T):
    """
    Compute log-likelihood score for a given set of facts and occurrences.
    """
    return T * np.log(p_i) + (1 - T) * np.log(1 - p_i)

def plot_scores(scores, final_diagram_output_path):
    # Convert dictionary to sorted lists of keys and values
    iterations = list(map(int, scores.keys()))
    scores_values = list(scores.values())

    # Create a line plot using seaborn
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=iterations, y=scores_values, marker="o", label="Log-likelihood Score")

    # Customize the plot
    plt.title("Log-likelihood Scores over Checkpoints", fontsize=16)
    plt.xlabel("Checkpoints (Probing Iterations)", fontsize=14)

    plt.xticks(iterations)
    plt.ylabel("Log-likelihood Score", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig(final_diagram_output_path)


# Main processing
checkpoints = os.listdir(path_to_checkpoints_probing_results)
sorted_checkpoints = sorted(checkpoints, key=get_num)
increasing_occurrences = load_json_dict(path_to_increasing_occurrences_in_slices)
scores_on_slices = {}

for idx, checkpoint in enumerate(tqdm(sorted_checkpoints, desc="IRT Score results in slices")):
    # Load checkpoint metadata
    metadata = load_json_dict(f"{path_to_checkpoints_probing_results}/{checkpoint}/metadata_results.json")
    irt_sum = []

    for relation_id, entity_dict in increasing_occurrences.items():
        # Get number of possible answers for this relation
        num_possible_answers = len(metadata[relation_id]["answer_space_labels"])
        for entity_id, occurrences_increase in entity_dict.items():
            slice_info = occurrences_increase["occurrences_increase"][idx]

            # Ensure slice and checkpoint match expectations
            assert slice_info["Slice"] == idx
            assert slice_info["checkpoint"] == checkpoint

            # Extract occurrence and correctness
            occurrences = slice_info["total"]
            T = 1 if slice_info["correct"] else 0

            c = 1 / num_possible_answers
            b = np.log(occurrences + 1)

            alph = 0.5

            if occurrences > 0:
                p_i = compute_pi(alph, b, c)
            else:
                p_i = c
            score = compute_log_likelihood(T)
            irt_sum.append(score)

    # Sum scores for the current slice
    scores_on_slices[f"{idx}"] = sum(irt_sum)

# Plot the scores
final_output = "../../sample_efficiency_evaluation_results/probing_results/BEAR-big/gpt2_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices/irt_scores_on_slices.png"
plot_scores(scores_on_slices, final_output)
