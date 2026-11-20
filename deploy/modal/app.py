"""Run the GPU kernel benchmark harness on a Modal GPU.

Cost control: `timeout=1800` hard-caps any single run at 30 minutes of billed GPU time so a
hang can't run away with the account's monthly credit. Per-GPU-hour rates (as of late 2026):
T4 ~$0.59, L4 ~$0.80, A10G ~$1.10, L40S ~$1.95, A100-40GB ~$2.10. A full 5-kernel sweep takes
~30-45 min, so budget roughly rate x 0.75 per full run.

Usage (run from this directory -- `add_local_dir` below is relative to it):
    cd deploy/modal
    modal run app.py                          # quick sanity check on L4 (one kernel, few iters)
    modal run app.py --full                    # full 5-kernel x 5-LLM sweep on L4
    modal run app.py --gpu A10G                # sanity check on a different GPU
    modal run app.py --full --gpu A10G          # full sweep on a different GPU
"""
import modal

app = modal.App("gpu-kernel-benchmark")

# nvidia/cuda *-devel image includes nvcc, required because the harness JIT-compiles each
# LLM-generated CUDA kernel via torch.utils.cpp_extension.load_inline at runtime.
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("ninja-build")
    .pip_install("torch", "ninja", "pynvml", "pyyaml", "pandas")
    .add_local_dir(
        "../..",
        remote_path="/root/gpu-kernel-benchmark",
        ignore=["results/raw/**", "results/combined/**", "__pycache__", ".torch_ext_cache"],
    )
)


@app.function(image=image, gpu="L4", timeout=1800)
def run_benchmark(kernel: str, iters: int, warmup: int, platform: str):
    import os
    import subprocess
    import sys

    os.chdir("/root/gpu-kernel-benchmark/harness")
    subprocess.run(
        [sys.executable, "runner.py", "--kernel", kernel, "--platform", platform,
         "--iters", str(iters), "--warmup", str(warmup)],
        check=True,
    )

    results_dir = f"/root/gpu-kernel-benchmark/results/raw/{platform}"
    out = {}
    if os.path.isdir(results_dir):
        for fname in os.listdir(results_dir):
            with open(os.path.join(results_dir, fname), "rb") as f:
                out[fname] = f.read()
    return out


@app.local_entrypoint()
def main(full: bool = False, gpu: str = "L4"):
    import pathlib

    kernel = "all" if full else "reduction"
    iters = 50 if full else 20
    warmup = 10 if full else 5
    platform = f"modal_{gpu.lower().replace('-', '_')}"

    print(f"Running kernel={kernel} iters={iters} warmup={warmup} on Modal {gpu} "
          f"(platform tag: {platform}) ...")
    results = run_benchmark.with_options(gpu=gpu).remote(
        kernel=kernel, iters=iters, warmup=warmup, platform=platform)

    out_dir = pathlib.Path(f"results_{platform}")
    out_dir.mkdir(exist_ok=True)
    for fname, data in results.items():
        (out_dir / fname).write_bytes(data)
    print(f"Wrote {len(results)} file(s) to {out_dir.resolve()}")
