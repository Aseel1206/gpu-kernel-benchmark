"""Reference implementation + input generator for non-causal scaled dot-product attention.

Every LLM-generated kernel under kernels/attention/llm_<name>/ must expose
`run(q, k, v) -> torch.Tensor` with this exact signature (see prompts/attention_prompt_v1.md).
"""
import math
import torch
import torch.nn.functional as F

# (batch, heads, seq_len, head_dim) sizes to sweep.
DEFAULT_SIZES = [(8, 8, 128, 64), (4, 8, 512, 64), (2, 8, 2048, 64)]


def get_inputs(size: tuple, dtype: torch.dtype, device: str) -> dict:
    batch, heads, seq, head_dim = size
    shape = (batch, heads, seq, head_dim)
    return {
        "q": torch.randn(shape, dtype=dtype, device=device),
        "k": torch.randn(shape, dtype=dtype, device=device),
        "v": torch.randn(shape, dtype=dtype, device=device),
    }


def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)


def flops(size: tuple) -> int:
    batch, heads, seq, head_dim = size
    qk_flops = 2 * batch * heads * seq * seq * head_dim
    av_flops = 2 * batch * heads * seq * seq * head_dim
    return qk_flops + av_flops


def bytes_moved(size: tuple, dtype_size: int = 4) -> int:
    batch, heads, seq, head_dim = size
    qkv_bytes = 3 * batch * heads * seq * head_dim * dtype_size
    out_bytes = batch * heads * seq * head_dim * dtype_size
    return qkv_bytes + out_bytes
