prompt_id: conv2d_v1

(Prepend `common_header.md`'s preamble verbatim, then this body.)

Operation: 2D convolution, no padding, stride 1 (keep it simple so every LLM is solving the same
problem; do not add padding/stride/dilation support).

Function signature:
```
def run(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    x:      4D float32 tensor, shape (batch, in_channels, H, W).
    weight: 4D float32 tensor, shape (out_channels, in_channels, kh, kw).
    returns: 4D float32 tensor, shape (batch, out_channels, H-kh+1, W-kw+1) — valid convolution,
             stride 1, no padding, no bias.
    """
```

Use a direct-convolution CUDA kernel (threads/blocks over output spatial positions and output
channels, accumulate over in_channels/kh/kw in a register), not an im2col-plus-matmul-via-torch
shortcut and not a call to torch.nn.functional.conv2d.
