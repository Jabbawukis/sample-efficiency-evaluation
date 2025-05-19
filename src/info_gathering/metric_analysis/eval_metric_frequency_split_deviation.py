import numpy as np
import os

import matplotlib.pyplot as plt
from utility.utility import load_json_dict


def plot_diverging_bars_rel(metrics_dict: dict, output_path: str, title: str):
    """
    Plots a diverging horizontal bar chart for each metric’s low- and high-frequency
    relative deviations from the full dataset. Saves the figure as PNG and PDF.

    metrics_dict: dict of {metric_name: [low_dev, high_dev], ...}
    output_path:   path (without extension) to save the figure.
    """
    # Prepare data
    metrics = list(metrics_dict.keys())
    low_vals = np.array([v[0] for v in metrics_dict.values()])
    high_vals = np.array([v[1] for v in metrics_dict.values()])
    y = np.arange(len(metrics))
    height = 0.4  # bar thickness

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot low-frequency bars (centered left/right of y)
    ax.barh(
        y - height / 2,
        low_vals,
        height=height,
        label="Low freq-split",
        color="tab:blue",
        alpha=0.8,
    )
    # Plot high-frequency bars
    ax.barh(
        y + height / 2,
        high_vals,
        height=height,
        label="High freq-split",
        color="tab:orange",
        alpha=0.8,
    )

    # Zero line
    ax.axvline(0, color="black", linewidth=1)

    # Formatting
    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Difference from Full Dataset")
    # ax.set_title(f"Metric {title} Deviations")
    ax.legend(loc="lower right")
    plt.tight_layout()

    # Save outputs
    fig.savefig(f"{output_path}.png", dpi=300)
    fig.savefig(f"{output_path}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    bear_sizes = ["big", "small"]
    abs_path = os.path.abspath(os.path.dirname(__file__)).split("sample-efficiency-evaluation")[0]
    models = [
        "gpt2_209m",
        "llama_208m",
        "xlstm_247m",
        "mamba2_172m",
        "gpt2_355m",
        "llama_360m",
        "xlstm_406m",
        "mamba2_432m",
    ]
    splits = {"low": [], "high": []}
    subset_percentage = {
        "big": {"dir": "0.8_0.2_2_0.2_0.8_seed_93", "low": "0.8_2_0.2", "high": "0.2_2_0.8"},
        "small": {"dir": "0.8_0.2_8_0.2_0.8_seed_93", "low": "0.8_8_0.2", "high": "0.2_8_0.8"},
    }
    for bear_size in bear_sizes:
        columns_rel = {"Accuracy": [], "WASB": [], "WAF": [], "α": []}
        columns_abs = {"Accuracy": [], "WASB": [], "WAF": [], "α": []}
        path_to_split_results = f"{abs_path}/sample-efficiency-evaluation-results/sample_efficiency_measures/metric_robustness/wikimedia_wikipedia_20231101_en/BEAR-{bear_size}/{subset_percentage[bear_size]['dir']}"
        plot_output = f"{path_to_split_results}/split_difference/"
        os.makedirs(plot_output, exist_ok=True)
        model_scores = load_json_dict(f"{path_to_split_results}/samples/model_scores.json")
        for split, _ in splits.items():
            for metric_name, _ in columns_rel.items():
                score_total = []
                split_list = []
                for model_name, model_dict in model_scores[metric_name][subset_percentage[bear_size][split]].items():
                    split_list.append(model_dict["on_split"])
                    score_total.append(model_dict["total"])

                rel_dev = np.subtract(split_list, score_total)
                rel_dev = np.divide(rel_dev, score_total)
                rel_dev = np.mean(rel_dev)
                columns_rel[metric_name].append(rel_dev)

                abs_dev = np.subtract(split_list, score_total)
                abs_dev = np.mean(abs_dev)
                columns_abs[metric_name].append(abs_dev)
        plot_diverging_bars_rel(
            columns_rel, f"{plot_output}/metric_frequency_split_rel_difference_{bear_size}", "Mean Relative"
        )
        plot_diverging_bars_rel(
            columns_abs, f"{plot_output}/metric_frequency_split_abs_difference_{bear_size}", "Mean Absolute"
        )
