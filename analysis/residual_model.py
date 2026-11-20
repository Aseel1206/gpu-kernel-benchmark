"""Step B of the hybrid predictor: fit a regression on the residual
(actual_time - roofline_predicted_time), validated with leave-one-GPU-out cross-validation.

With only a handful of measured GPUs this is a proof-of-concept correction term, not a claim of
strong generalization -- report the leave-one-GPU-out numbers honestly rather than overclaiming.
"""
import ast
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from roofline import load_specs, predict_time_ms, arithmetic_intensity, load_baseline

ROOT = Path(__file__).resolve().parent.parent
COMBINED_PATH = ROOT / "results" / "combined" / "all_results.parquet"

FEATURE_COLUMNS = [
    "sm_count", "boost_clock_mhz", "mem_bandwidth_gbps",
    "fp32_tflops", "fp16_tflops", "l2_cache_mb", "tdp_w",
    "arithmetic_intensity",
]


def build_feature_table(results: pd.DataFrame, specs: pd.DataFrame) -> pd.DataFrame:
    """Join each result row with its GPU's static specs + kernel arithmetic intensity,
    and compute the roofline baseline prediction + residual target."""
    df = results.merge(specs, on="gpu_name", how="inner", suffixes=("", "_spec"))

    def row_ai_and_pred(row):
        size = ast.literal_eval(row["input_shape"])
        ai = arithmetic_intensity(row["kernel_type"], size)
        pred_ms = predict_time_ms(row["kernel_type"], size, row, dtype=row["dtype"].replace("torch.", ""))
        return pd.Series({"arithmetic_intensity": ai, "roofline_pred_ms": pred_ms})

    df = df.join(df.apply(row_ai_and_pred, axis=1))
    df["residual_ms"] = df["mean_time_ms"] - df["roofline_pred_ms"]
    return df


def leave_one_gpu_out_cv(df: pd.DataFrame, model_cls=LinearRegression):
    """For each GPU in turn: train the residual model on all other GPUs, predict on the
    held-out one, report MAE of (roofline + residual_model) vs the roofline-only baseline."""
    results = []
    for held_out_gpu in df["gpu_name"].unique():
        train = df[df["gpu_name"] != held_out_gpu]
        test = df[df["gpu_name"] == held_out_gpu]
        if len(train) < 5 or len(test) == 0:
            continue

        model = model_cls()
        model.fit(train[FEATURE_COLUMNS], train["residual_ms"])
        predicted_residual = model.predict(test[FEATURE_COLUMNS])
        hybrid_pred = test["roofline_pred_ms"].values + predicted_residual

        hybrid_mae = mean_absolute_error(test["mean_time_ms"], hybrid_pred)
        roofline_only_mae = mean_absolute_error(test["mean_time_ms"], test["roofline_pred_ms"])
        results.append({
            "held_out_gpu": held_out_gpu,
            "n_test_rows": len(test),
            "roofline_only_mae_ms": roofline_only_mae,
            "hybrid_mae_ms": hybrid_mae,
            "improvement_pct": 100 * (roofline_only_mae - hybrid_mae) / roofline_only_mae
            if roofline_only_mae else None,
        })
    return pd.DataFrame(results)


def main():
    if not COMBINED_PATH.exists():
        print(f"{COMBINED_PATH} not found -- run analysis/merge_results.py first.")
        return

    results = pd.read_parquet(COMBINED_PATH)
    results = results[results["correctness_pass"] == True]  # noqa: E712
    specs = load_specs()

    df = build_feature_table(results, specs)

    print("=== Linear regression residual correction ===")
    print(leave_one_gpu_out_cv(df, LinearRegression))

    print("\n=== Random forest residual correction ===")
    print(leave_one_gpu_out_cv(df, lambda: RandomForestRegressor(n_estimators=100, max_depth=4)))


if __name__ == "__main__":
    main()
