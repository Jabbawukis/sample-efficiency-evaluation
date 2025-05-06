import numpy as np
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from utility.utility import load_json_dict
import info_gathering.paths as paths

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
    columns = {
        "ACC": f"{abs_path}/sample-efficiency-evaluation-results/sample_efficiency_measures/accuracy_over_slices/wikimedia_wikipedia_20231101_en/BEAR-{{bear_size}}/model_scores.json",
        "WAF": f"{abs_path}/sample-efficiency-evaluation-results/sample_efficiency_measures/weighted_accuracy_over_slices/wikimedia_wikipedia_20231101_en/BEAR-{{bear_size}}/over_all_facts/model_scores.json",
        "WASB": f"{abs_path}/sample-efficiency-evaluation-results/sample_efficiency_measures/weighted_accuracy_over_slices/wikimedia_wikipedia_20231101_en/BEAR-{{bear_size}}/on_buckets/model_scores.json",
        "α": f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{{bear_size}}/{{model}}/{paths.model_optimized_params_wikipedia_20231101_en}/psf_optimized_alphas.json",
        "λ": f"{abs_path}/sample-efficiency-evaluation-results/probing_results/BEAR-{{bear_size}}/{{model}}/{paths.model_optimized_params_wikipedia_20231101_en}/cdf_optimized_lambdas.json",
    }
    for bear_size in bear_sizes:
        scores_matrix = []
        for column in columns.keys():
            score = []
            if column == "α":
                for model in models:
                    path = columns[column].format(bear_size=bear_size, model=model)
                    scores = load_json_dict(path)
                    score.append(scores["Alphas"][-1]["alpha"])
            elif column == "λ":
                for model in models:
                    path = columns[column].format(bear_size=bear_size, model=model)
                    scores = load_json_dict(path)
                    score.append(scores["Lambdas"][-1]["lambda"])
            else:
                path = columns[column].format(bear_size=bear_size)
                scores = load_json_dict(path)
                for model in models:
                    score.append(scores[model]["41"])
            scores_matrix.append(score)
        scores_matrix = np.array(scores_matrix)
        scores_matrix = scores_matrix.T
        df = pd.DataFrame(scores_matrix, columns=list(columns.keys()), index=models)
        df_corr = df.corr(method="pearson")
        sns.heatmap(df_corr, annot=True)
        plt.tight_layout()
        output_path = f"{abs_path}/sample-efficiency-evaluation-results/sample_efficiency_measures/metric_correlation/wikimedia_wikipedia_20231101_en/last_slice/BEAR-{bear_size}/"
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(f"{output_path}/corr_last_slice_bear_{bear_size}.png")
        plt.savefig(f"{output_path}/corr_last_slice_bear_{bear_size}.pdf")
        plt.clf()
        plt.close()

    slices = 42
    for bear_size in bear_sizes:
        scores_matrix = []
        for column in columns.keys():
            score = []
            if column == "α":
                for model in models:
                    path = columns[column].format(bear_size=bear_size, model=model)
                    scores = load_json_dict(path)
                    for slice_index in range(slices):
                        score.append(scores["Alphas"][slice_index]["alpha"])
            elif column == "λ":
                for model in models:
                    path = columns[column].format(bear_size=bear_size, model=model)
                    scores = load_json_dict(path)
                    for slice_index in range(slices):
                        try:
                            score.append(scores["Lambdas"][slice_index]["lambda"])
                        except IndexError:
                            score.append(np.nan)
            else:
                path = columns[column].format(bear_size=bear_size)
                scores = load_json_dict(path)
                for model in models:
                    for slice_index in range(slices):
                        score.append(scores[model][str(slice_index)])
            scores_matrix.append(score)
        scores_matrix = np.array(scores_matrix)
        scores_matrix = scores_matrix.T
        df = pd.DataFrame(scores_matrix, columns=list(columns.keys()), index=models * slices)
        df_corr = df.corr(method="pearson")
        sns.heatmap(df_corr, annot=True)
        plt.tight_layout()
        output_path = f"{abs_path}/sample-efficiency-evaluation-results/sample_efficiency_measures/metric_correlation/wikimedia_wikipedia_20231101_en/all_slices/BEAR-{bear_size}/"
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(f"{output_path}/corr_all_slices_bear_{bear_size}.png")
        plt.savefig(f"{output_path}/corr_all_slices_bear_{bear_size}.pdf")
        plt.clf()
        plt.close()
