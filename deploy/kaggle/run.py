import os
import shutil
import subprocess
import sys

INPUT_ROOT = "/kaggle/input"
EXTRACT_TO = "/kaggle/working/gpu-kernel-benchmark"

print("Listing /kaggle/input:")
for root, dirs, files in os.walk(INPUT_ROOT):
    depth = root[len(INPUT_ROOT):].count(os.sep)
    if depth > 2:
        dirs[:] = []
        continue
    print(root, "->", dirs, files[:5], f"(+{max(0, len(files) - 5)} more)" if len(files) > 5 else "")

# Find the dataset folder that actually contains our project (robust to Kaggle mounting it
# under a different path than the declared dataset slug).
source = None
for root, dirs, files in os.walk(INPUT_ROOT):
    if "harness" in dirs and "kernels" in dirs:
        source = root
        break

if source is None:
    raise SystemExit("Could not locate the gpu-kernel-benchmark project under /kaggle/input -- see listing above.")

print(f"Found project source at: {source}")
shutil.copytree(source, EXTRACT_TO)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ninja", "pynvml", "pyyaml"], check=True)

import torch
print(torch.__version__, torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU detected")

os.chdir(os.path.join(EXTRACT_TO, "harness"))
subprocess.run([sys.executable, "runner.py", "--kernel", "all", "--platform", "kaggle"], check=True)
