# Model: DeepSeek (latest), Kernel: 5. Attention (online-softmax / flash-style, fused)
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// One block per (batch, head, query). Each thread owns one head-dim of the
// output accumulator; the q.k dot product is block-reduced via warp shuffles.
// Online-softmax (running m, l, o) keeps memory O(head_dim) per block -- the
// full (seq_len x seq_len) score matrix is never materialized.
__global__ void attention_kernel(const float* __restrict__ q,
                                  const float* __restrict__ k,
                                  const float* __restrict__ v,
                                  float* __restrict__ out,
                                  int seq_len, int head_dim, float scale) {
    int qi = blockIdx.x;
    int h  = blockIdx.y;
    int b  = blockIdx.z;
    int tid = threadIdx.x;
    int bd  = blockDim.x;

    long bh_off = ((long)b * gridDim.y + h) * (long)seq_len * head_dim;
    const float* qv = q + bh_off + (long)qi * head_dim;
    const float* kb = k + bh_off;
    const float* vb = v + bh_off;
    float* ov = out + bh_off + (long)qi * head_dim;

    __shared__ float s_reduce[32];

    float q_val = (tid < head_dim) ? qv[tid] : 0.0f;
    float o_acc = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;
    unsigned mask = 0xffffffffu;

    for (int j = 0; j < seq_len; j++) {
        // --- dot(q, k_j) ---
        float partial = (tid < head_dim) ? q_val * kb[(long)j * head_dim + tid] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_down_sync(mask, partial, off);

        if ((tid & 31) == 0) s_reduce[tid >> 5] = partial;
        __syncthreads();

        if (tid < 32) {
            int nw = bd >> 5;
            float val = (tid < nw) ? s_reduce[tid] : 0.0f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                val += __shfl_down_sync(mask, val, off);
            if (tid == 0) s_reduce[0] = val;
        }
        __syncthreads();

        float s = s_reduce[0] * scale;

        // --- online softmax update ---
        float m_new = fmaxf(m, s);
        float alpha = __expf(m - m_new);
        float beta  = __expf(s - m_new);
        l = l * alpha + beta;

        float vval = (tid < head_dim) ? vb[(long)j * head_dim + tid] : 0.0f;
        o_acc = o_acc * alpha + beta * vval;
        m = m_new;
        __syncthreads();
    }

    if (tid < head_dim) ov[tid] = o_acc / l;
}

torch::Tensor attention_launcher(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    int B = (int)q.size(0);
    int H = (int)q.size(1);
    int S = (int)q.size(2);
    int D = (int)q.size(3);
    auto out = torch::empty_like(q);

    float scale = 1.0f / sqrtf((float)D);
    int threads = ((D + 31) / 32) * 32;
    if (threads < 32) threads = 32;
    if (threads > 1024) threads = 1024;

    dim3 grid(S, H, B);
    attention_kernel<<<grid, threads>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        out.data_ptr<float>(), S, D, scale);
    return out;
}
"""

cpp_source = "torch::Tensor attention_launcher(torch::Tensor q, torch::Tensor k, torch::Tensor v);"

ext = load_inline(
    name="attention_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["attention_launcher"],
    verbose=False,
)

def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return ext.attention_launcher(q, k, v)
