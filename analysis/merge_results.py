"""Merge every results/raw/<platform>/*.csv into one combined table for analysis."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "results" / "raw"
OUT_PATH = ROOT / "results" / "combined" / "all_results.parquet"


def main():
    frames = []
    for csv_path in RAW_DIR.glob("*/*.csv"):
        # input_shape is always str(size), but for 1D kernels (reduction) that's a bare
        # int-looking string ("65536"), so pandas infers int64 there vs. object elsewhere --
        # force str so concatenating frames across kernel types doesn't produce a mixed-type
        # column that pyarrow refuses to write.
        df = pd.read_csv(csv_path, dtype={"input_shape": str})
        df["platform_dir"] = csv_path.parent.name  # cross-check against the 'platform' column
        frames.append(df)

    if not frames:
        print(f"No CSVs found under {RAW_DIR} yet -- run harness/runner.py on a GPU platform first.")
        return

    combined = pd.concat(frames, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False)
    print(f"Merged {len(frames)} file(s), {len(combined)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
