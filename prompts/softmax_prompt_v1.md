prompt_id: softmax_v1

(Prepend `common_header.md`'s preamble verbatim, then this body.)

Operation: row-wise, numerically-stable softmax of a 2D tensor.

Function signature:
```
def run(x: torch.Tensor) -> torch.Tensor:
    """
    x: 2D float32 tensor, shape (batch, N). N varies across test cases (e.g. 128 to 8192),
       not guaranteed to be a power of 2.
    returns: tensor of the same shape, softmax applied along dim=-1 (each row sums to 1).
    """
```

Must be numerically stable (subtract the row max before exponentiating). One CUDA block per row
(or per row-tile), using shared memory / warp shuffles to compute the row max, then the
shifted-exp sum, then the normalized output, all without calling torch.softmax or
torch.nn.functional.softmax.
