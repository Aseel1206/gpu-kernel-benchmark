"""Sustained-load thermal sweep: run one kernel implementation back-to-back for an extended
period, sampling GPU temperature/clock/power alongside performance at regular intervals, so you
can see how rising GPU temperature (thermal throttling) affects speed over time.

This is a different question from the short correctness+timing benchmark in bench.py (which runs
~50-100 iterations in a couple seconds -- nowhere near long enough to heat the GPU up). It matters
most on a laptop GPU with weaker cooling (the RTX 3050), where sustained load can hit the thermal
limit and clock down; a datacenter GPU (Colab/Kaggle/L4) is far less likely to show this, but
running the same sweep there gives a useful contrast point.

Usage:
    python thermal_sweep.py --kernel matmul --source pytorch --size "(2048,2048,2048)" \
        --platform rtx3050_laptop --duration-min 10 --window-s 5

Output: one row per sampling window to results/thermal/<platform>/<kernel>_<source>_<ts>.csv
with columns: elapsed_s, window_mean_time_ms, window_iters, gpu_temp_c, gpu_clock_mhz, power_w.
"""
import argparse
import ast
import csv
import time
from pathlib import Path

import torch

from bench import load_baseline, load_impl

ROOT = Path(__file__).resolve().parent.parent


class ThermalSampler:
    """Point-in-time GPU temp/clock/power reads (not an averaging background thread like
    PowerLogger -- here we want an instantaneous snapshot at each window boundary)."""

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception:
            self.pynvml = None
            self.handle = None

    def read(self):
        if self.handle is None:
            return {"gpu_temp_c": None, "gpu_clock_mhz": None, "power_w": None}
        p = self.pynvml
        return {
            "gpu_temp_c": p.nvmlDeviceGetTemperature(self.handle, p.NVML_TEMPERATURE_GPU),
            "gpu_clock_mhz": p.nvmlDeviceGetClockInfo(self.handle, p.NVML_CLOCK_SM),
            "power_w": p.nvmlDeviceGetPowerUsage(self.handle) / 1000.0,
        }


def run_sweep(kernel_type: str, source: str, size, dtype: torch.dtype, device: str,
              duration_s: float, window_s: float):
    baseline = load_baseline(kernel_type)
    impl = baseline if source == "pytorch" else load_impl(kernel_type, source)
    inputs = baseline.get_inputs(size, dtype, device)
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    sampler = ThermalSampler()

    # Warmup so the first window isn't skewed by one-time compile/cache-fill cost.
    for _ in range(10):
        impl.run(**inputs)
    if is_cuda:
        torch.cuda.synchronize()

    rows = []
    start = time.perf_counter()
    window_start = start
    window_iters = 0
    window_time_total_ms = 0.0

    while True:
        t0 = time.perf_counter()
        impl.run(**inputs)
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        window_iters += 1
        window_time_total_ms += (t1 - t0) * 1000

        now = time.perf_counter()
        if now - window_start >= window_s:
            reading = sampler.read()
            rows.append({
                "elapsed_s": now - start,
                "window_mean_time_ms": window_time_total_ms / window_iters,
                "window_iters": window_iters,
                **reading,
            })
            print(f"  t={now - start:6.1f}s  mean={window_time_total_ms / window_iters:8.3f}ms "
                  f"temp={reading['gpu_temp_c']}C clock={reading['gpu_clock_mhz']}MHz "
                  f"power={reading['power_w']}W")
            window_start = now
            window_iters = 0
            window_time_total_ms = 0.0

        if now - start >= duration_s:
            break

    return rows


def write_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", required=True,
                   choices=["reduction", "softmax", "matmul", "conv2d", "attention"])
    p.add_argument("--source", default="pytorch", help='"pytorch" or an LLM name from llm_lineup.yaml')
    p.add_argument("--size", default=None, help="python literal, e.g. \"(2048,2048,2048)\"; "
                                                 "defaults to the kernel's largest DEFAULT_SIZES entry")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--platform", required=True, help="e.g. rtx3050_laptop, colab, kaggle, l4")
    p.add_argument("--duration-min", type=float, default=10.0)
    p.add_argument("--window-s", type=float, default=5.0)
    args = p.parse_args()

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device detected -- thermal readings will be None and this sweep "
              "is meaningless as a thermal-throttling study; it only exercises the code path.")

    baseline = load_baseline(args.kernel)
    size = ast.literal_eval(args.size) if args.size else baseline.DEFAULT_SIZES[-1]

    print(f"Running {args.kernel}/{args.source} size={size} dtype={args.dtype} "
          f"for {args.duration_min} min, sampling every {args.window_s}s ...")
    rows = run_sweep(args.kernel, args.source, size, dtype, device,
                      duration_s=args.duration_min * 60, window_s=args.window_s)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = (ROOT / "results" / "thermal" / args.platform /
                f"{args.kernel}_{args.source}_{ts}.csv")
    write_csv(rows, out_path)
    print(f"\n{len(rows)} windows written to {out_path}")


if __name__ == "__main__":
    main()
