prompt_id: matmul_v1

(Prepend `common_header.md`'s preamble verbatim, then this body.)

Operation: dense matrix multiplication.

Function signature:
```
def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: 2D float32 tensor, shape (M, K).
    b: 2D float32 tensor, shape (K, N).
    returns: 2D float32 tensor, shape (M, N), equal to a @ b.
    """
```

Use a tiled/blocked CUDA matmul kernel: load tiles of A and B into `__shared__` memory and
accumulate in a register, the standard shared-memory-tiling pattern — not a naive kernel that
reads directly from global memory per output element. M, K, N are not guaranteed to be multiples
of any block/tile size — handle boundary checks. Do not call torch.matmul, torch.mm, or the `@`
operator to compute the result.
