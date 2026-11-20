"""Reference implementation + input generator for the parallel-reduction (sum) kernel.

Every LLM-generated kernel under kernels/reduction/llm_<name>/ must expose
`run(x: torch.Tensor) -> torch.Tensor` with this exact signature (see
prompts/reduction_prompt_v1.md) so the harness can swap implementations transparently.
"""
import torch

# (N,) sizes to sweep. Keep in sync with what the harness passes to get_inputs.
DEFAULT_SIZES = [2**16, 2**20, 2**24]


def get_inputs(size: int, dtype: torch.dtype, device: str) -> dict:
    return {"x": torch.randn(size, dtype=dtype, device=device)}


def run(x: torch.Tensor) -> torch.Tensor:
    """PyTorch-native baseline — this IS the "pytorch" row in the results table."""
    return torch.sum(x)


def flops(size: int) -> int:
    return size - 1  # N-1 additions


def bytes_moved(size: int, dtype_size: int = 4) -> int:
    return size * dtype_size  # read N elements, write 1 scalar (negligible)
