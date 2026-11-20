"""Reference implementation + input generator for valid, stride-1 2D convolution.

Every LLM-generated kernel under kernels/conv2d/llm_<name>/ must expose
`run(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor` with this exact signature (see
prompts/conv2d_prompt_v1.md).
"""
import torch
import torch.nn.functional as F

# (batch, in_ch, H, W, out_ch, kh, kw) sizes to sweep.
DEFAULT_SIZES = [
    (8, 3, 64, 64, 16, 3, 3),
    (8, 16, 128, 128, 32, 3, 3),
    (4, 32, 256, 256, 64, 5, 5),
]


def get_inputs(size: tuple, dtype: torch.dtype, device: str) -> dict:
    batch, in_ch, h, w, out_ch, kh, kw = size
    return {
        "x": torch.randn(batch, in_ch, h, w, dtype=dtype, device=device),
        "weight": torch.randn(out_ch, in_ch, kh, kw, dtype=dtype, device=device),
    }


def run(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.conv2d(x, weight, bias=None, stride=1, padding=0)


def flops(size: tuple) -> int:
    batch, in_ch, h, w, out_ch, kh, kw = size
    out_h, out_w = h - kh + 1, w - kw + 1
    return 2 * batch * out_ch * out_h * out_w * in_ch * kh * kw


def bytes_moved(size: tuple, dtype_size: int = 4) -> int:
    batch, in_ch, h, w, out_ch, kh, kw = size
    out_h, out_w = h - kh + 1, w - kw + 1
    x_bytes = batch * in_ch * h * w * dtype_size
    w_bytes = out_ch * in_ch * kh * kw * dtype_size
    out_bytes = batch * out_ch * out_h * out_w * dtype_size
    return x_bytes + w_bytes + out_bytes
