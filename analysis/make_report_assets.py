"""Generate every figure + numeric summary needed for the IEEE report, from the data already
collected. Run once; outputs land in report/figures/*.pdf and report/numbers.json (so the LaTeX
source can cite exact figures without anyone re-typing them from memory).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "analysis"))

from roofline import load_specs, arithmetic_intensity, predict_time_ms  # noqa: E402
from residual_model import build_feature_table, leave_one_gpu_out_cv  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

GPU_ORDER = ["Tesla T4", "NVIDIA L4", "NVIDIA A10G", "NVIDIA GeForce RTX 3050 6GB Laptop GPU"]
GPU_SHORT = {
    "Tesla T4": "T4", "NVIDIA L4": "L4", "NVIDIA A10G": "A10G",
    "NVIDIA GeForce RTX 3050 6GB Laptop GPU": "RTX 3050",
}
LLM_ORDER = ["pytorch", "claude", "deepseek", "gpt", "grok", "gemini"]
KERNEL_ORDER = ["reduction", "softmax", "matmul", "conv2d", "attention"]

numbers = {}

df = pd.read_parquet(ROOT / "results" / "combined" / "all_results.parquet")
passing = df[df["correctness_pass"] == True].copy()  # noqa: E712

# ---------------------------------------------------------------------------
# Fig 1: correctness pass-rate matrix (LLM x platform)
# ---------------------------------------------------------------------------
pr = df.groupby(["source", "platform"])["correctness_pass"].mean().unstack()
platform_order = [p for p in ["colab", "kaggle", "l4", "modal_a10g", "rtx3050_laptop"] if p in pr.columns]
pr = pr.reindex(index=[s for s in LLM_ORDER if s in pr.index], columns=platform_order)
platform_labels = {"colab": "Colab\n(T4)", "kaggle": "Kaggle\n(T4)", "l4": "Modal\n(L4)",
                    "modal_a10g": "Modal\n(A10G)", "rtx3050_laptop": "Laptop\n(RTX 3050)"}

fig, ax = plt.subplots(figsize=(5.2, 3.0))
im = ax.imshow(pr.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(pr.columns)))
ax.set_xticklabels([platform_labels.get(c, c) for c in pr.columns])
ax.set_yticks(range(len(pr.index)))
ax.set_yticklabels(pr.index)
for i in range(pr.shape[0]):
    for j in range(pr.shape[1]):
        v = pr.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center",
                     color="black" if 0.3 < v < 0.9 else "white", fontsize=8)
fig.colorbar(im, ax=ax, label="Correctness pass rate", shrink=0.8)
ax.set_title("Correctness pass rate by LLM and platform")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_correctness_matrix.pdf")
plt.close(fig)

numbers["correctness_matrix"] = pr.round(3).to_dict()

# ---------------------------------------------------------------------------
# Fig 2: latency comparison, compute-bound (matmul) vs memory-bound (reduction)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
for ax, kernel, title in zip(axes, ["matmul", "reduction"],
                              ["Matmul (compute-bound), largest size", "Reduction (memory-bound), largest size"]):
    sub = passing[passing["kernel_type"] == kernel]
    sizes = sub["input_shape"].unique()
    # pick the largest size by string length heuristic then numeric parse fallback
    largest = sorted(sizes, key=lambda s: eval(s) if isinstance(eval(s), (int, float)) else max(eval(s)))[-1]
    sub = sub[sub["input_shape"] == largest]
    piv = sub.pivot_table(index="gpu_name", columns="source", values="mean_time_ms", aggfunc="mean")
    piv = piv.reindex(index=[g for g in GPU_ORDER if g in piv.index],
                       columns=[s for s in LLM_ORDER if s in piv.columns])
    piv.rename(index=GPU_SHORT).plot(kind="bar", ax=ax, legend=False, width=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("Mean time (ms, log scale)")
    ax.set_xlabel("")
    ax.set_title(title, fontsize=8)
    ax.tick_params(axis="x", rotation=0)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.12), fontsize=7)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_latency_comparison.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 3: roofline plot
# ---------------------------------------------------------------------------
specs = load_specs()
fig, ax = plt.subplots(figsize=(5.2, 3.6))
intensities = np.logspace(-1, 3, 200)
colors = plt.cm.tab10(np.linspace(0, 1, len(specs)))
for (_, gpu), color in zip(specs.iterrows(), colors):
    peak_flops = gpu["fp32_tflops"] * 1e12
    peak_bw = gpu["mem_bandwidth_gbps"] * 1e9
    ceiling = np.minimum(peak_flops, intensities * peak_bw) / 1e9
    ax.plot(intensities, ceiling, "--", color=color, linewidth=1, alpha=0.6,
            label=f"{GPU_SHORT.get(gpu['gpu_name'], gpu['gpu_name'])} roofline")

for (gpu_name, source), group in passing.groupby(["gpu_name", "source"]):
    if source == "pytorch":
        continue
    ai_vals, gflops_vals = [], []
    for _, row in group.iterrows():
        size = eval(row["input_shape"])
        ai = arithmetic_intensity(row["kernel_type"], size)
        ai_vals.append(ai)
        gflops_vals.append(row["throughput"])
    color = colors[GPU_ORDER.index(gpu_name)] if gpu_name in GPU_ORDER else "gray"
    marker = {"claude": "o", "deepseek": "s", "gpt": "^", "grok": "D"}.get(source, "x")
    ax.scatter(ai_vals, gflops_vals, color=color, marker=marker, s=18, alpha=0.8)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Arithmetic intensity (FLOPs/byte)")
ax.set_ylabel("Achieved throughput (GFLOPs/s)")
ax.set_title("Roofline: achieved throughput vs. arithmetic intensity")
ax.legend(fontsize=6, loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_roofline.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 4: power efficiency (GFLOPs/Watt), matmul largest size, compute-bound kernels only
# ---------------------------------------------------------------------------
eff = passing[passing["kernel_type"].isin(["matmul", "conv2d", "attention"])].copy()
eff = eff[eff["avg_power_w"].notna()]
eff["gflops_per_watt"] = eff["throughput"] / eff["avg_power_w"]
piv_eff = eff.groupby(["kernel_type", "source"])["gflops_per_watt"].mean().unstack()
piv_eff = piv_eff.reindex(index=[k for k in KERNEL_ORDER if k in piv_eff.index],
                           columns=[s for s in LLM_ORDER if s in piv_eff.columns])

fig, ax = plt.subplots(figsize=(5.2, 3.0))
piv_eff.plot(kind="bar", ax=ax, width=0.8)
ax.set_ylabel("GFLOPs / Watt")
ax.set_xlabel("")
ax.set_title("Power efficiency by kernel and source (all GPUs pooled)")
ax.legend(fontsize=7, ncol=2)
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_power_efficiency.pdf")
plt.close(fig)

numbers["power_efficiency_gflops_per_watt"] = piv_eff.round(3).to_dict()

# ---------------------------------------------------------------------------
# Fig 5: leave-one-GPU-out MAE comparison
# ---------------------------------------------------------------------------
feat = build_feature_table(passing, specs)
lin = leave_one_gpu_out_cv(feat, LinearRegression)
rf = leave_one_gpu_out_cv(feat, lambda: RandomForestRegressor(n_estimators=100, max_depth=4, random_state=0))

fig, ax = plt.subplots(figsize=(5.2, 3.0))
x = np.arange(len(lin))
width = 0.25
ax.bar(x - width, lin["roofline_only_mae_ms"], width, label="Roofline only")
ax.bar(x, lin["hybrid_mae_ms"], width, label="Hybrid (linear)")
ax.bar(x + width, rf["hybrid_mae_ms"], width, label="Hybrid (random forest)")
ax.set_xticks(x)
ax.set_xticklabels([GPU_SHORT.get(g, g) for g in lin["held_out_gpu"]])
ax.set_ylabel("MAE (ms), held-out GPU")
ax.set_title("Leave-one-GPU-out prediction error")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_logo_cv.pdf")
plt.close(fig)

numbers["logo_cv_linear"] = lin.round(3).to_dict(orient="records")
numbers["logo_cv_rf"] = rf.round(3).to_dict(orient="records")

# ---------------------------------------------------------------------------
# Fig 6: RTX 3050 thermal sweep
# ---------------------------------------------------------------------------
thermal_files = list((ROOT / "results" / "thermal" / "rtx3050_laptop").glob("*.csv"))
if thermal_files:
    tdf = pd.read_csv(thermal_files[0])
    fig, ax1 = plt.subplots(figsize=(5.2, 3.0))
    ax1.plot(tdf["elapsed_s"], tdf["window_mean_time_ms"], color="tab:red", label="Latency (ms)")
    ax1.set_xlabel("Elapsed time (s)")
    ax1.set_ylabel("Mean latency (ms)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(tdf["elapsed_s"], tdf["gpu_temp_c"], color="tab:blue", label="Temp (C)")
    ax2.set_ylabel("GPU temperature (C)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    ax1.set_title("RTX 3050: sustained-load latency vs. GPU temperature\n(Claude attention kernel, seq_len=2048)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_thermal_sweep.pdf")
    plt.close(fig)

    numbers["thermal_sweep"] = {
        "duration_s": float(tdf["elapsed_s"].max()),
        "temp_start_c": float(tdf["gpu_temp_c"].iloc[0]),
        "temp_end_c": float(tdf["gpu_temp_c"].iloc[-1]),
        "clock_start_mhz": float(tdf["gpu_clock_mhz"].iloc[0]),
        "clock_end_mhz": float(tdf["gpu_clock_mhz"].iloc[-1]),
        "power_mean_w": float(tdf["power_w"].mean()),
        "power_std_w": float(tdf["power_w"].std()),
        "latency_early_ms": float(tdf["window_mean_time_ms"].iloc[2:8].mean()),
        "latency_late_ms": float(tdf["window_mean_time_ms"].iloc[-8:].mean()),
    }

# ---------------------------------------------------------------------------
# Numeric summaries (no plot, but needed for the text/tables)
# ---------------------------------------------------------------------------
numbers["total_rows"] = int(len(df))
numbers["passing_rows"] = int(len(passing))
numbers["gpus"] = list(df["gpu_name"].dropna().unique())
numbers["platforms"] = list(df["platform"].dropna().unique())
numbers["overall_pass_rate_by_source"] = df.groupby("source")["correctness_pass"].mean().round(3).to_dict()

mean_time_by_kernel_source_gpu = passing.pivot_table(
    index=["kernel_type", "gpu_name"], columns="source", values="mean_time_ms", aggfunc="mean"
).round(4)
numbers["mean_time_ms_table"] = {str(k): v for k, v in mean_time_by_kernel_source_gpu.to_dict(orient="index").items()}

# run-to-run variance (two independent kaggle runs -- the second run is kept out of the main
# dataset deliberately, per the user's request, so it's read from its separate directory)
kaggle_run2_dir = ROOT / "results" / "raw_kaggle_run2_separate"
kaggle_files_1 = sorted((ROOT / "results" / "raw" / "kaggle").glob("*.csv"))
kaggle_files_2 = sorted(kaggle_run2_dir.glob("*.csv")) if kaggle_run2_dir.is_dir() else []
if kaggle_files_1 and kaggle_files_2:
    r1 = pd.concat([pd.read_csv(f, dtype={"input_shape": str}) for f in kaggle_files_1], ignore_index=True)
    r2 = pd.concat([pd.read_csv(f, dtype={"input_shape": str}) for f in kaggle_files_2], ignore_index=True)
    r1p = r1[r1["correctness_pass"] == True]  # noqa: E712
    r2p = r2[r2["correctness_pass"] == True]  # noqa: E712
    key = ["kernel_type", "source", "input_shape", "dtype"]
    mm = r1p.merge(r2p, on=key, suffixes=("_1", "_2"))
    mm["pct_diff"] = (mm["mean_time_ms_2"] / mm["mean_time_ms_1"] - 1) * 100
    numbers["kaggle_variance"] = {
        "n_matched": int(len(mm)),
        "pct_diff_mean": float(mm["pct_diff"].mean()),
        "pct_diff_std": float(mm["pct_diff"].std()),
        "pct_diff_min": float(mm["pct_diff"].min()),
        "pct_diff_max": float(mm["pct_diff"].max()),
    }
    key2 = ["kernel_type", "source", "input_shape"]
    cm = r1.merge(r2, on=key2, suffixes=("_1", "_2"))
    mismatches = cm[cm["correctness_pass_1"] != cm["correctness_pass_2"]]
    numbers["kaggle_correctness_mismatches"] = int(len(mismatches))

with open(ROOT / "report" / "numbers.json", "w") as f:
    json.dump(numbers, f, indent=2, default=str)

print("Wrote figures to", FIG_DIR)
print("Wrote numbers to", ROOT / "report" / "numbers.json")
