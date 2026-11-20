"""Step A of the hybrid predictor: analytical roofline time estimate for a (kernel, size, GPU).

predicted_time = max(FLOPs / peak_FLOPS, bytes_moved / peak_bandwidth)

This alone should land in the right order of magnitude for compute-bound kernels (matmul,
conv2d, attention) and memory-bound ones (reduction); softmax sits in between. The gap between
this prediction and measured time is what analysis/residual_model.py learns to correct.
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KERNELS_DIR = ROOT / "kernels"
SPECS_PATH = ROOT / "gpu_specs" / "specs.csv"


def load_baseline(kernel_type: str):
    path = KERNELS_DIR / kernel_type / "pytorch_baseline.py"
    spec = importlib.util.spec_from_file_location(f"{kernel_type}_baseline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_specs() -> pd.DataFrame:
    return pd.read_csv(SPECS_PATH, comment="#")


def arithmetic_intensity(kernel_type: str, size) -> float:
    baseline = load_baseline(kernel_type)
    return baseline.flops(size) / baseline.bytes_moved(size)


def predict_time_ms(kernel_type: str, size, gpu_specs_row: dict, dtype: str = "fp32") -> float:
    """gpu_specs_row: a dict-like with fp32_tflops/fp16_tflops and mem_bandwidth_gbps
    (as found in gpu_specs/specs.csv)."""
    baseline = load_baseline(kernel_type)
    total_flops = baseline.flops(size)
    total_bytes = baseline.bytes_moved(size)

    peak_tflops = gpu_specs_row["fp16_tflops"] if dtype in ("fp16", "bf16") else gpu_specs_row["fp32_tflops"]
    peak_flops_per_s = peak_tflops * 1e12
    peak_bytes_per_s = gpu_specs_row["mem_bandwidth_gbps"] * 1e9

    compute_bound_s = total_flops / peak_flops_per_s
    memory_bound_s = total_bytes / peak_bytes_per_s
    predicted_s = max(compute_bound_s, memory_bound_s)
    return predicted_s * 1000


if __name__ == "__main__":
    specs = load_specs()
    if specs.empty:
        print(f"No GPU rows in {SPECS_PATH} yet -- fill it in before running predictions.")
        sys.exit(0)
    baseline = load_baseline("matmul")
    size = baseline.DEFAULT_SIZES[0]
    for _, gpu_row in specs.iterrows():
        pred = predict_time_ms("matmul", size, gpu_row, dtype="fp32")
        print(f"{gpu_row['gpu_name']}: predicted matmul{size} time = {pred:.4f} ms "
              f"(arithmetic intensity = {arithmetic_intensity('matmul', size):.2f} FLOPs/byte)")
