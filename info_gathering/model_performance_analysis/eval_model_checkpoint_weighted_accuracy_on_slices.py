import numpy as np
import os
import matplotlib.pyplot as plt

from info_gathering.model_performance_analysis.util import get_checkpoint_weighted_accuracy


def weighting_function(occurrences, lambda_=0.01):
    return np.exp(-lambda_ * occurrences) if occurrences > 0 else 0


def plot_params(weighed_scores_models: dict, output_path: str, output_diagram_name: str):

    plt.figure(figsize=(24, 18))

    # Ensure all x-axis values are shown
    plt.xticks(range(0, 42))

    for _model, model_scores in weighed_scores_models.items():

        # Get the x and y values
        slices = np.array(list(model_scores.keys()))
        scores = np.array(list(model_scores.values()))

        plt.plot(slices, scores, marker="o", linestyle="-", label=f"{_model}")

        last_x = slices[-1]
        last_y = scores[-1]
        plt.text(
            float(last_x),
            float(last_y),
            f"{last_y:.4f}",
            fontsize=12,
            color="black",
            ha="left",  # Align text to the left of the point
            va="bottom",
        )

    # Add titles, labels, and legend
    plt.title("Weighed Accuracy Scores Over All Facts", fontsize=16)
    plt.xlabel("Slices", fontsize=14)
    plt.ylabel("Weighed Accuracy Score", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(output_path, f"{output_diagram_name}.png"))
    plt.clf()
    plt.close()


if __name__ == "__main__":
    models = ["gpt2_124m", "gpt2_209m", "mamba2_172m", "xlstm_247m", "gpt2_355m"] # results depend on other models
    bear_sizes = ["big", "small"]
    abs_path = os.path.abspath(os.path.dirname(__file__)).split("sample_efficiency_evaluation")[0]

    for bear_size in bear_sizes:
        model_weighted_accuracy_on_slices = {}
        final_diagram_output_path = f"{abs_path}/sample_efficiency_evaluation_results/probing_results/weighted_accuracy_over_slices/wikimedia_wikipedia_20231101_en/BEAR-{bear_size}"

        for model in models:
            path_to_checkpoints_probing_results = f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-big/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/probing_results_on_checkpoints/checkpoint_extracted"
            path_to_increasing_occurrences_in_slices = f"{abs_path}/sample_efficiency_evaluation_results/probing_results/BEAR-{bear_size}/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/increasing_occurrences_in_slices.json"
            data = get_checkpoint_weighted_accuracy(
                path_to_checkpoints_probing_results, path_to_increasing_occurrences_in_slices, weighting_function
            )
            model_weighted_accuracy_on_slices[model] = data

        plot_params(
            model_weighted_accuracy_on_slices,
            final_diagram_output_path,
            f"weighted_accuracy_on_slices_bear_{bear_size}",
        )
