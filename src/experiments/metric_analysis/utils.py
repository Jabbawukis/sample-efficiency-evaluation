import os
import numpy as np
import matplotlib.pyplot as plt


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


def plot_scores(data: dict, output_path: str):
    plt.figure(figsize=(10, 4))
    metrics = ["Accuracy", "WASB", "WAF", "α"]
    colors = {"Accuracy": "tab:blue", "α": "tab:green", "WASB": "tab:red", "WAF": "tab:orange"}

    for split in data["Accuracy"].keys():
        models = list(data["Accuracy"][split].keys())
        x = np.arange(len(models))  # Model positions on x-axis
        width = 0.11  # Reduce width to fit bars properly without overlap

        fig, ax = plt.subplots(figsize=(7, 5))

        bar_containers = []
        labels = []

        for i, metric in enumerate(metrics):
            on_split_values = [data[metric][split][model]["on_split"] for model in models]
            total_values = [data[metric][split][model]["total"] for model in models]

            bar1 = ax.bar(
                x + (i - 1) * 2 * width,
                on_split_values,
                width,
                label=f"{metric} (on split)",
                color=colors[metric],
                alpha=0.6,
            )
            bar2 = ax.bar(
                x + (i - 1) * 2 * width + width,
                total_values,
                width,
                label=f"{metric} (on all data)",
                color=colors[metric],
                alpha=1.0,
                hatch="//",
            )

            bar_containers.append(bar1)
            bar_containers.append(bar2)
            labels.append(f"{metric} (on split)")
            labels.append(f"{metric} (on all data)")

            # Add value labels above bars
            for b in bar1:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height(),
                    f"{b.get_height():.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=50,
                )
            for b in bar2:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height(),
                    f"{b.get_height():.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=50,
                )

        ax.set_ylabel("Scores")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.set_ylim(0, 1.1)
        # Creating legend with metric colors for both on_split and total
        legend_labels = [plt.Rectangle((0, 0), 0.5, 0.5, color=colors[m], alpha=0.6) for m in metrics] + [
            plt.Rectangle((0, 0), 1, 1, fc=colors[m], alpha=1.0, hatch="//") for m in metrics
        ]
        legend_texts = [f"{m} (on split)" for m in metrics] + [f"{m} (on all data)" for m in metrics]
        ax.legend(legend_labels, legend_texts, title="Metrics", loc="upper left")
        # ax.set_xlim(-0.45, len(models) - 0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{split}.png"))
        plt.savefig(os.path.join(output_path, f"{split}.pdf"))
        plt.clf()
        plt.close()


def power_scaling_function_small(alpha, x, L_0, x_0):
    return 1 - (0 + 0.88 / (np.power((1 + x), alpha)))


def power_scaling_function_big(alpha, x, L_0, x_0):
    return 1 - (0 + 0.92 / (np.power((1 + x), alpha)))


vectorized_psf = np.vectorize(power_scaling_function_small, excluded=["alpha", "L_0", "x_0"])
vectorized_psf_big = np.vectorize(power_scaling_function_big, excluded=["alpha", "L_0", "x_0"])
