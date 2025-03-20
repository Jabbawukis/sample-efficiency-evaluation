import numpy as np
import os
import matplotlib.pyplot as plt
import info_gathering.paths as paths

from info_gathering.model_performance_analysis.util import get_checkpoint_accuracy_overall


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
    num_slices = 1
    abs_path = os.path.abspath(os.path.dirname(__file__)).split("sample-efficiency-evaluation")[0]
    for bear_size in bear_sizes:
        model_accuracy_on_slices = {}
        final_diagram_output_path = ""
        for model in models:
            path_to_increasing_occurrences_in_slices = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/wikimedia_wikipedia_20231101_en/occurrence_info_0.2_1024_0.8.json"
            final_diagram_output_path = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/accuracy_over_slices/wikimedia_wikipedia_20231101_en/BEAR-{bear_size}/"
            data = get_checkpoint_accuracy_overall(
                num_slices,
                path_to_increasing_occurrences_in_slices
            )

            if not os.path.exists(final_diagram_output_path):
                os.makedirs(final_diagram_output_path)

            model_accuracy_on_slices[model] = data
        plot_params(
            model_accuracy_on_slices,
            final_diagram_output_path,
            f"accuracy_on_slices_bear_{bear_size}_0.2_1024_0.8",
        )
