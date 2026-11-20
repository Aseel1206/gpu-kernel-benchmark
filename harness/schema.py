"""Single source of truth for the results row schema.

Every harness/analysis script imports RESULT_COLUMNS instead of hardcoding
column names, so the schema only has to change in one place.
"""

RESULT_COLUMNS = [
    "run_id",
    "kernel_type",       # reduction | softmax | conv2d | matmul | attention
    "source",            # llm name, or "pytorch" for the baseline
    "prompt_id",
    "prompt_quality_score",
    "gen_temperature",
    "gpu_name",
    "platform",          # colab | kaggle | l4 | rtx3050_laptop | ...
    "input_shape",
    "dtype",
    "mean_time_ms",
    "std_time_ms",
    "throughput",        # GFLOPs/s for compute-bound kernels, elems/s for reduction
    "avg_power_w",
    "peak_power_w",
    "avg_temp_c",
    "correctness_pass",
    "max_abs_error",
    "notes",
]


def new_row(**kwargs) -> dict:
    """Build a result row dict with every schema column present (None if unset)."""
    row = {col: None for col in RESULT_COLUMNS}
    unknown = set(kwargs) - set(RESULT_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown result columns: {sorted(unknown)}")
    row.update(kwargs)
    return row
