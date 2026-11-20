# Shared rules for every kernel prompt (prepend to each kernel-specific prompt)

Give this exact preamble to every LLM, for every kernel, unchanged — only the kernel-specific
body below it should vary. Keeping it identical across LLMs is what makes the comparison fair.

```
Write a raw CUDA kernel for PyTorch (torch >= 2.1), compiled at import time with
torch.utils.cpp_extension.load_inline.

Requirements:
- Output a single, self-contained Python code block. It must, when executed, define a Python
  wrapper function named exactly `run`, with the exact keyword arguments listed in "Function
  signature" below, that is backed by a hand-written __global__ CUDA kernel (not a call to the
  torch.* equivalent).
- Structure it as: a `cuda_source` string containing #include <torch/extension.h> plus your
  __global__ kernel(s) and a C++ launcher function that calls them; a matching `cpp_source`
  string declaring that launcher's signature; one call to
  `torch.utils.cpp_extension.load_inline(name=..., cpp_sources=cpp_source, cuda_sources=cuda_source,
  functions=[...], verbose=False)`; then the `run` wrapper that calls into the compiled extension
  and returns a torch.Tensor on the same device/dtype as the inputs (unless told otherwise).
- Do not call the equivalent torch.* function to compute the result (e.g. no torch.matmul,
  torch.softmax, torch.sum, torch.nn.functional.conv2d, torch.nn.functional.scaled_dot_product_attention)
  inside `run` — the whole point is a hand/LLM-written kernel, not a passthrough wrapper. Using
  torch only for tensor allocation/metadata (e.g. torch.empty, .shape, .stride) is fine.
- Only output the one Python code block. No prose, no markdown outside the code block.
- Assume inputs are already on a CUDA device and contiguous unless stated otherwise.
```

Record per generation, in the matching `*.meta.json` next to the saved output:
`llm_name`, `llm_version` (if known), `gen_temperature` (the LLM's sampling temperature if you
used an API/playground that exposes it; "unspecified" if you used a plain chat UI that doesn't —
this is *not* the GPU's thermal temperature, which is tracked separately during benchmarking, see
`harness/thermal_sweep.py`), `prompt_id`, `prompt_quality_score` (see `rubric.md`), `timestamp`,
and `attempt` (if you had to retry after a compile/runtime failure — keep the failed attempts too,
as their own numbered files, rather than overwriting them).
