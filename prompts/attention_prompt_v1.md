prompt_id: attention_v1

(Prepend `common_header.md`'s preamble verbatim, then this body.)

Operation: scaled dot-product attention, non-causal, no mask, single pass
("flash-attention-lite": tiled/online-softmax style is encouraged but a simpler fused kernel
is acceptable as long as it is not a plain torch implementation).

Function signature:
```
def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    q, k, v: 4D float32 tensors, shape (batch, heads, seq_len, head_dim). seq_len and head_dim
             vary across test cases (e.g. seq_len in {128, 512, 2048}, head_dim in {32, 64}).
    returns: 4D float32 tensor, shape (batch, heads, seq_len, head_dim), equal to
             softmax(q @ k.transpose(-1, -2) / sqrt(head_dim)) @ v, no mask, no dropout.
    """
```

Prefer an online-softmax (flash-attention style) tiled CUDA kernel that never materializes the
full (seq_len x seq_len) attention matrix in global memory, to keep this feasible at seq_len=2048+;
a simpler kernel that does materialize it (e.g. one kernel for Q@K^T, one for the softmax, one for
@V) is acceptable as a fallback if the LLM cannot produce the fused tiled version, but note which
style was produced in `notes`. Do not call torch.nn.functional.scaled_dot_product_attention or
implement this by calling torch.softmax/torch.matmul directly on the full attention matrix as the
"kernel" (hand-written CUDA is required for at least the core matmul+softmax math).
