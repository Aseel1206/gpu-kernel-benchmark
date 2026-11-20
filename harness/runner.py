"""CLI sweep: kernel type x implementation (pytorch baseline + every llm_<name>/v1.py found)
x input size x dtype, appending one result row per run to results/raw/<platform>/<ts>.csv.

Example:
    python runner.py --kernel matmul --platform colab
    python runner.py --kernel all --platform kaggle --iters 100
"""
import argparse
import csv
import json
import time
import uuid
from pathlib import Path

import torch

from bench import bench_one, load_baseline, load_impl, KERNELS_DIR
from power_logger import PowerLogger
from schema import RESULT_COLUMNS, new_row

ROOT = Path(__file__).resolve().parent.parent
ALL_KERNELS = ["reduction", "softmax", "matmul", "conv2d", "attention"]
DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def discover_llm_implementations(kernel_type: str):
    """Return [(llm_name, version), ...] for every llm_*/v*.py found under this kernel."""
    kernel_dir = KERNELS_DIR / kernel_type
    found = []
    for llm_dir in sorted(kernel_dir.glob("llm_*")):
        llm_name = llm_dir.name[len("llm_"):]
        for version_file in sorted(llm_dir.glob("v*.py")):
            found.append((llm_name, version_file.stem))
    return found


def read_meta(kernel_type: str, llm_name: str, version: str) -> dict:
    meta_path = KERNELS_DIR / kernel_type / f"llm_{llm_name}" / f"{version}.meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def run_sweep(kernel_type: str, platform: str, gpu_name: str, sizes, dtypes, iters, warmup,
              out_dir: Path):
    baseline = load_baseline(kernel_type)
    sizes = sizes or baseline.DEFAULT_SIZES
    device = "cuda" if torch.cuda.is_available() else "cpu"

    implementations = [("pytorch", None, baseline)]
    load_errors = {}  # llm_name -> error string, for kernels that failed to import/compile
    for llm_name, version in discover_llm_implementations(kernel_type):
        try:
            impl = load_impl(kernel_type, llm_name, version)
            implementations.append((llm_name, version, impl))
        except Exception as e:  # noqa: BLE001 -- a failed compile/import is a valid result, not a crash
            load_errors[llm_name] = str(e)
            print(f"  LOAD/COMPILE FAILED: {llm_name} {kernel_type} -- {e}")

    rows = []
    for llm_name, version, impl_module in implementations:
        meta = {} if llm_name == "pytorch" else read_meta(kernel_type, llm_name, version)
        for size in sizes:
            for dtype_name in dtypes:
                dtype = DTYPE_MAP[dtype_name]
                logger = PowerLogger()
                result = bench_one(
                    kernel_type=kernel_type,
                    impl_module=impl_module,
                    size=size,
                    dtype=dtype,
                    device=device,
                    source=llm_name,
                    warmup=warmup,
                    iters=iters,
                    power_logger=logger,
                )
                row = new_row(
                    run_id=str(uuid.uuid4()),
                    source=llm_name,
                    prompt_id=meta.get("prompt_id"),
                    prompt_quality_score=meta.get("prompt_quality_score"),
                    gen_temperature=meta.get("gen_temperature"),
                    gpu_name=gpu_name,
                    platform=platform,
                    **result,
                )
                rows.append(row)
                print(f"[{kernel_type}] {llm_name:12s} size={size} dtype={dtype_name} "
                      f"-> pass={row['correctness_pass']} "
                      f"time_ms={row['mean_time_ms']}")

    for llm_name, err in load_errors.items():
        meta = read_meta(kernel_type, llm_name, "v1")
        for size in sizes:
            for dtype_name in dtypes:
                row = new_row(
                    run_id=str(uuid.uuid4()),
                    kernel_type=kernel_type,
                    source=llm_name,
                    prompt_id=meta.get("prompt_id"),
                    prompt_quality_score=meta.get("prompt_quality_score"),
                    gen_temperature=meta.get("gen_temperature"),
                    gpu_name=gpu_name,
                    platform=platform,
                    input_shape=str(size),
                    dtype=str(DTYPE_MAP[dtype_name]),
                    correctness_pass=False,
                    notes=f"load/compile failed: {err[:500]}",
                )
                rows.append(row)
                print(f"[{kernel_type}] {llm_name:12s} size={size} dtype={dtype_name} "
                      f"-> pass=False (load/compile failed)")
    return rows


def write_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", required=True, choices=ALL_KERNELS + ["all"])
    p.add_argument("--platform", required=True,
                   help='e.g. colab, kaggle, l4, rtx3050_laptop')
    p.add_argument("--gpu-name", default=None,
                   help="defaults to torch.cuda.get_device_name(0) if a CUDA device is present")
    p.add_argument("--sizes", default=None,
                   help="comma-separated override, format depends on kernel (see pytorch_baseline.py)")
    p.add_argument("--dtype", default="fp32", help="comma-separated: fp32,fp16,bf16")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()

    gpu_name = args.gpu_name
    if gpu_name is None and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    elif gpu_name is None:
        gpu_name = "cpu (no CUDA device detected)"

    kernels = ALL_KERNELS if args.kernel == "all" else [args.kernel]
    dtypes = args.dtype.split(",")
    out_dir = ROOT / "results" / "raw" / args.platform
    ts = time.strftime("%Y%m%d_%H%M%S")

    for kernel_type in kernels:
        sizes = None  # per-kernel size parsing from --sizes is left as a TODO; use DEFAULT_SIZES
        rows = run_sweep(kernel_type, args.platform, gpu_name, sizes, dtypes,
                          args.iters, args.warmup, out_dir)
        write_csv(rows, out_dir / f"{kernel_type}_{ts}.csv")

    print(f"\nResults written under {out_dir}")


if __name__ == "__main__":
    main()
