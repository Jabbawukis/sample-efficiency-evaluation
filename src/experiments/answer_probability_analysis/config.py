import os
from src.experiments.answer_probability_analysis.cdf_nll_optimization import (
    vectorized_cdf,
)
from src.experiments.answer_probability_analysis.psf_nll_optimization import (
    vectorized_psf,
    vectorized_psf_default,
)
from src.experiments.answer_probability_analysis.optimization_utils import (
    get_slice_data,
)

ABS_PATH = os.path.abspath(os.path.dirname(__file__)).split("sample-efficiency-evaluation")[0]
MODELS = [
    "gpt2_209m",
    "gpt2_355m",
    "mamba2_172m",
    "mamba2_432m",
    "xlstm_247m",
    "xlstm_406m",
    "llama_208m",
    "llama_360m",
]
BEAR_SIZES = ["big", "small"]
NUM_SLICES = 42
FUNCTIONS = [
    {
        "function_method": vectorized_cdf,
        "function_name": "EDF",
        "file_name": "cdf_optimized_lambdas.json",
        "get_slice_data": get_slice_data,
        "Params": "Lambdas",
        "Param": "lambda",
    },
    {
        "function_method": vectorized_psf,
        "function_name": "PSF",
        "file_name": "psf_optimized_alphas.json",
        "get_slice_data": get_slice_data,
        "Params": "Alphas",
        "Param": "alpha",
    },
    {
        "function_method": vectorized_psf_default,
        "function_name": r"$PSF_{0,1}$",
        "file_name": "psf_optimized_alphas_default.json",
        "get_slice_data": get_slice_data,
        "Params": "Alphas",
        "Param": "alpha",
    },
]
