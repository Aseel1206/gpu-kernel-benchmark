"""Reference implementation + input generator for the row-wise softmax kernel.

Every LLM-generated kernel under kernels/softmax/llm_<name>/ must expose
`run(x: torch.Tensor) -> torch.Tensor` with this exact signature (see
prompts/softmax_prompt_v1.md).
"""
import torch

# (batch, N) sizes to sweep.
DEFAULT_SIZES = [(4096, 128), (2048, 1024), (512, 8192)]


def get_inputs(size: tuple, dtype: torch.dtype, device: str) -> dict:
    batch, n = size
    return {"x": torch.randn(batch, n, dtype=dtype, device=device)}


def run(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def flops(size: tuple) -> int:
    batch, n = size
    return batch * n * 5  # max, sub, exp, sum, div -- rough op count


def bytes_moved(size: tuple, dtype_size: int = 4) -> int:
    batch, n = size
    return 2 * batch * n * dtype_size  # read + write the full matrix
