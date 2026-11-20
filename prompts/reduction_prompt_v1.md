prompt_id: reduction_v1

(Prepend `common_header.md`'s preamble verbatim, then this body.)

Operation: parallel sum reduction of a 1D tensor.

Function signature:
```
def run(x: torch.Tensor) -> torch.Tensor:
    """
    x: 1D float32 tensor, shape (N,), N not guaranteed to be a power of 2 or a
       multiple of any particular block size.
    returns: 0-dimensional (scalar) tensor containing sum(x).
    """
```

Use a genuine parallel reduction strategy (e.g. shared-memory tree reduction within a block,
optionally with warp-level `__shfl_down_sync` for the final warp, plus a second kernel launch or
`atomicAdd` to combine partial sums across blocks) — not a single-threaded sequential loop. Handle
N not divisible by the block size (bounds checking) correctly.
