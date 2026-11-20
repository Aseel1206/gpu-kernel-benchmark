"""Load one kernel implementation, time it with CUDA events, and check correctness
against the PyTorch baseline for the same kernel_type before trusting the timing.

Usage as a library (see runner.py for the CLI sweep):
    from bench import bench_one
    row = bench_one(kernel_type="matmul", impl_module=some_module, size=(1024, 1024, 1024),
                     dtype=torch.float32, device="cuda", source="pytorch")
"""
import importlib.util
import os
import time
from pathlib import Path

import torch

# Force true FP32 everywhere. On Ampere+/Ada GPUs (A100, L4, ...) cuDNN/cuBLAS default to TF32
# for conv/matmul, so the PyTorch reference result silently loses precision relative to the
# hand-written FP32 CUDA kernels it's being compared against -- on Turing (T4) this is a non-issue
# since TF32 doesn't exist there, which is exactly why this only showed up on L4 runs.
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

ROOT = Path(__file__).resolve().parent.parent
KERNELS_DIR = ROOT / "kernels"
EXT_CACHE_DIR = ROOT / ".torch_ext_cache"

# Relative tolerance per kernel type -- attention/conv2d accumulate more rounding error
# than a single reduction or matmul at fp32, so give them a looser bound.
TOLERANCES = {
    "reduction": 1e-2,
    "softmax": 1e-4,
    "matmul": 1e-2,
    "conv2d": 1e-2,
    "attention": 1e-2,
}


def load_baseline(kernel_type: str):
    path = KERNELS_DIR / kernel_type / "pytorch_baseline.py"
    spec = importlib.util.spec_from_file_location(f"{kernel_type}_baseline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_impl(kernel_type: str, llm_name: str, version: str = "v1"):
    """Load an LLM-generated kernel file; it must define run(...) matching the baseline.

    Each (kernel_type, llm_name) gets its own TORCH_EXTENSIONS_DIR so that two different
    LLM-generated CUDA sources that happen to pick the same `load_inline(name=...)` string don't
    collide in PyTorch's extension build cache when the harness loads multiple implementations
    back-to-back in the same process.
    """
    os.environ["TORCH_EXTENSIONS_DIR"] = str(EXT_CACHE_DIR / kernel_type / llm_name)
    path = KERNELS_DIR / kernel_type / f"llm_{llm_name}" / f"{version}.py"
    spec = importlib.util.spec_from_file_location(f"{kernel_type}_{llm_name}_{version}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def bench_one(
    kernel_type: str,
    impl_module,
    size,
    dtype: torch.dtype,
    device: str,
    source: str,
    warmup: int = 10,
    iters: int = 50,
    power_logger=None,
):
    """Run correctness check + timed benchmark for one (kernel_type, implementation, size).

    Returns a dict matching schema.RESULT_COLUMNS (minus identifiers the caller fills in:
    run_id, source, prompt_id, prompt_quality_score, gen_temperature, gpu_name, platform, notes).
    """
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    baseline = load_baseline(kernel_type)
    inputs = baseline.get_inputs(size, dtype, device)

    def sync():
        if is_cuda:
            torch.cuda.synchronize()

    with torch.no_grad():
        ref_out = baseline.run(**inputs)
        sync()

        try:
            out = impl_module.run(**inputs)
            sync()
        except Exception as e:  # noqa: BLE001 - a failed generated kernel is a valid result
            return {
                "kernel_type": kernel_type,
                "input_shape": str(size),
                "dtype": str(dtype),
                "correctness_pass": False,
                "max_abs_error": None,
                "mean_time_ms": None,
                "std_time_ms": None,
                "throughput": None,
                "notes": f"runtime error: {e}",
            }

        err = _max_abs_error(ref_out, out)
        passed = err <= TOLERANCES.get(kernel_type, 1e-2)

        if not passed:
            return {
                "kernel_type": kernel_type,
                "input_shape": str(size),
                "dtype": str(dtype),
                "correctness_pass": False,
                "max_abs_error": err,
                "mean_time_ms": None,
                "std_time_ms": None,
                "throughput": None,
                "notes": "failed correctness check -- timing not recorded",
            }

        # Warmup
        for _ in range(warmup):
            impl_module.run(**inputs)
        sync()

        if power_logger is not None:
            power_logger.start()

        if is_cuda:
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            for i in range(iters):
                start_events[i].record()
                impl_module.run(**inputs)
                end_events[i].record()
            sync()
            times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        else:
            # CPU fallback (dev-machine scaffold testing only -- not a real benchmark platform).
            times_ms = []
            for _ in range(iters):
                t0 = time.perf_counter()
                impl_module.run(**inputs)
                times_ms.append((time.perf_counter() - t0) * 1000)

        if power_logger is not None:
            power_logger.stop()

        mean_ms = sum(times_ms) / len(times_ms)
        std_ms = (sum((t - mean_ms) ** 2 for t in times_ms) / len(times_ms)) ** 0.5

        total_flops = baseline.flops(size)
        throughput_gflops = total_flops / (mean_ms / 1000) / 1e9

        row = {
            "kernel_type": kernel_type,
            "input_shape": str(size),
            "dtype": str(dtype),
            "mean_time_ms": mean_ms,
            "std_time_ms": std_ms,
            "throughput": throughput_gflops,
            "correctness_pass": True,
            "max_abs_error": err,
            "notes": "",
        }
        if power_logger is not None:
            row["avg_power_w"] = power_logger.avg_power_w
            row["peak_power_w"] = power_logger.peak_power_w
            row["avg_temp_c"] = power_logger.avg_temp_c
        return row


if __name__ == "__main__":
    # Smoke test: PyTorch baseline benched against itself (should always pass, err == 0).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("No CUDA device available -- this is a scaffold-only smoke test on CPU, "
              "timings will not be meaningful. Run on an actual GPU platform for real results.")
    baseline = load_baseline("matmul")
    size = baseline.DEFAULT_SIZES[0]
    result = bench_one("matmul", baseline, size, torch.float32, device, source="pytorch",
                        warmup=3, iters=5)
    print(result)
