"""Reference implementation + input generator for dense matrix multiplication.

Every LLM-generated kernel under kernels/matmul/llm_<name>/ must expose
`run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor` with this exact signature (see
prompts/matmul_prompt_v1.md).
"""
import torch

# (M, K, N) sizes to sweep.
DEFAULT_SIZES = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]


def get_inputs(size: tuple, dtype: torch.dtype, device: str) -> dict:
    m, k, n = size
    return {
        "a": torch.randn(m, k, dtype=dtype, device=device),
        "b": torch.randn(k, n, dtype=dtype, device=device),
    }


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)


def flops(size: tuple) -> int:
    m, k, n = size
    return 2 * m * k * n  # multiply-add pairs


def bytes_moved(size: tuple, dtype_size: int = 4) -> int:
    m, k, n = size
    return (m * k + k * n + m * n) * dtype_size
