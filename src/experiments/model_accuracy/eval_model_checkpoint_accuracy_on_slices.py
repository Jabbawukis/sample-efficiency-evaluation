import numpy as np
import os
import matplotlib.pyplot as plt
import experiments.paths as paths

from experiments.model_accuracy.util import get_checkpoint_accuracy_overall
from utility.utility import save_dict_as_json


def plot_scores(scores_models: dict, output_path: str, output_diagram_name: str):

    plt.figure(figsize=(16, 10))

    # Ensure all x-axis values are shown
    plt.xticks(range(0, 42))

    for _model, model_scores in scores_models.items():

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
    plt.title("Accuracy Scores Over All Facts", fontsize=16)
    plt.xlabel("Slices", fontsize=14)
    plt.ylabel("Accuracy Score", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{output_diagram_name}.pdf"))
    plt.savefig(os.path.join(output_path, f"{output_diagram_name}.png"))
    plt.clf()
    plt.close()


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
    ]  # results depend on other models
    bear_sizes = ["big", "small"]
    num_slices = 42
    abs_path = os.path.abspath(os.path.dirname(__file__)).split("sample-efficiency-evaluation")[0]
    for bear_size in bear_sizes:
        model_accuracy_on_slices = {}
        final_diagram_output_path = ""
        for model in models:
            path_to_increasing_occurrences_in_slices = f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{bear_size}/{model}/{paths.increasing_occurrences_in_slices_wikipedia_20231101_en}"
            final_diagram_output_path = f"{abs_path}/sample-efficiency-evaluation-results/sample_efficiency_measures/accuracy_over_slices/wikimedia_wikipedia_20231101_en/BEAR-{bear_size}/"
            data = get_checkpoint_accuracy_overall(num_slices, path_to_increasing_occurrences_in_slices)

            if not os.path.exists(final_diagram_output_path):
                os.makedirs(final_diagram_output_path)

            model_accuracy_on_slices[model] = data
        save_dict_as_json(model_accuracy_on_slices, f"{final_diagram_output_path}/model_scores.json")
        plot_scores(
            model_accuracy_on_slices,
            final_diagram_output_path,
            f"accuracy_on_slices_bear_{bear_size}",
        )
