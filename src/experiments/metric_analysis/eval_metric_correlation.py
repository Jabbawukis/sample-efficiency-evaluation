import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utility.utility import load_json_dict
from src.experiments.metric_analysis.config import BEAR_SIZES, MODELS, COLUMNS, SLICES, ABS_PATH


def plot_correlation_heatmap(df, output_path, bear_size, slice_info):
    df_corr = df.corr(method="pearson")
    sns.heatmap(df_corr, annot=True)
    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(f"{output_path}/corr_{slice_info}_bear_{bear_size}.png")
    plt.savefig(f"{output_path}/corr_{slice_info}_bear_{bear_size}.pdf")
    plt.clf()
    plt.close()


def main():
    for bear_size in BEAR_SIZES:
        scores_matrix = []
        for column in COLUMNS.keys():
            score = []
            if column == "α":
                for model in MODELS:
                    path = COLUMNS[column].format(bear_size=bear_size, model=model)
                    scores = load_json_dict(path)
                    score.append(scores["Alphas"][-1]["alpha"])
            elif column == "λ":
                for model in MODELS:
                    path = COLUMNS[column].format(bear_size=bear_size, model=model)
                    scores = load_json_dict(path)
                    score.append(scores["Lambdas"][-1]["lambda"])
            else:
                path = COLUMNS[column].format(bear_size=bear_size)
                scores = load_json_dict(path)
                for model in MODELS:
                    score.append(scores[model]["41"])
            scores_matrix.append(score)

        scores_matrix = np.array(scores_matrix).T
        df = pd.DataFrame(scores_matrix, columns=list(COLUMNS.keys()), index=MODELS)
        output_path = f"{ABS_PATH}/sample-efficiency-evaluation-results/sample_efficiency_measures/metric_correlation/wikimedia_wikipedia_20231101_en/last_slice/BEAR-{bear_size}/"
        plot_correlation_heatmap(df, output_path, bear_size, "last_slice")

    for bear_size in BEAR_SIZES:
        scores_matrix = []
        for column in COLUMNS.keys():
            score = []
            if column == "α":
                for model in MODELS:
                    path = COLUMNS[column].format(bear_size=bear_size, model=model)
                    scores = load_json_dict(path)
                    for slice_index in range(SLICES):
                        score.append(scores["Alphas"][slice_index]["alpha"])
            elif column == "λ":
                for model in MODELS:
                    path = COLUMNS[column].format(bear_size=bear_size, model=model)
                    scores = load_json_dict(path)
                    for slice_index in range(SLICES):
                        try:
                            score.append(scores["Lambdas"][slice_index]["lambda"])
                        except IndexError:
                            score.append(np.nan)
            else:
                path = COLUMNS[column].format(bear_size=bear_size)
                scores = load_json_dict(path)
                for model in MODELS:
                    for slice_index in range(SLICES):
                        score.append(scores[model][str(slice_index)])
            scores_matrix.append(score)

        scores_matrix = np.array(scores_matrix).T
        df = pd.DataFrame(scores_matrix, columns=list(COLUMNS.keys()), index=MODELS * SLICES)
        output_path = f"{ABS_PATH}/sample-efficiency-evaluation-results/sample_efficiency_measures/metric_correlation/wikimedia_wikipedia_20231101_en/all_slices/BEAR-{bear_size}/"
        plot_correlation_heatmap(df, output_path, bear_size, "all_slices")


if __name__ == "__main__":
    main()
