import numpy as np
import os

import matplotlib.pyplot as plt
from utility.utility import load_json_dict


def plot_scores(metrics_dict: dict, output_path: str):
    """
    Plots grouped bars for each metric across low-split, high-split, and average,
    annotates each bar with its value, and saves to PNG and PDF.
    """
    colors = {"Accuracy": "tab:blue", "α": "tab:green", "WASB": "tab:red", "WAF": "tab:orange"}
    categories = ["Low frequency-split", "High frequency-split"]
    x = np.arange(len(categories))
    width = 0.2  # width of each bar

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metric, values) in enumerate(metrics_dict.items()):
        # plot bars and capture the BarContainer
        bars = ax.bar(
            x + i * width - width * (len(metrics_dict) - 1) / 2,
            values,
            width,
            label=metric,
            color=colors[metric],
            alpha=0.9,
        )
        # annotate each bar with its value, rounded to 4 decimals
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=3)

    # Labels and legend
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Deviation To Full Dataset")
    # ax.set_xlabel('Dataset Split')
    # ax.set_title('Average Deviation of Splits to Full Dataset')
    ax.legend(title="Metrics")
    plt.tight_layout()

    # Save outputs
    plt.savefig(f"{output_path}.png")
    plt.savefig(f"{output_path}.pdf")
    plt.clf()
    plt.close()


if __name__ == "__main__":
    bear_sizes = ["small", "big"]
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
        columns = {"Accuracy": [], "WASB": [], "WAF": [], "α": []}
        model_scores = load_json_dict(
            f"{abs_path}/sample-efficiency-evaluation-results/sample_efficiency_measures/metric_robustness/wikimedia_wikipedia_20231101_en/BEAR-{bear_size}/{subset_percentage[bear_size]['dir']}/samples/model_scores.json"
        )
        for split, _ in splits.items():
            for metric_name, _ in columns.items():
                score_total = []
                split_list = []
                for model_name, model_dict in model_scores[metric_name][subset_percentage[bear_size][split]].items():
                    split_list.append(model_dict["on_split"])
                    score_total.append(model_dict["total"])
                subtraction = np.subtract(split_list, score_total)
                subtraction = np.absolute(subtraction)
                subtraction = np.mean(subtraction)
                columns[metric_name].append(subtraction)
        print(columns)
        plot_scores(columns, f"./metric_frequency_split_deviation_{bear_size}")
