# Model: Grok (xAI)
# Kernel: Attention

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 32

// Simpler fused but not full flash: per-head, materialize scores in shared for small tiles
// For generality and correctness we use a multi-kernel approach that stays within GPU memory
// for the tested sizes, with hand-written kernels for QK^T, softmax, and PV.

__global__ void qk_kernel(const float* __restrict__ q, const float* __restrict__ k, float* __restrict__ scores,
                           int B, int H, int S, int D, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S * S;
    if (idx >= total) return;

    int s2 = idx % S;
    int tmp = idx / S;
    int s1 = tmp % S;
    tmp = tmp / S;
    int h = tmp % H;
    int b = tmp / H;

    const float* q_ptr = q + ((b * H + h) * S + s1) * D;
    const float* k_ptr = k + ((b * H + h) * S + s2) * D;

    float sum = 0.0f;
    for (int d = 0; d < D; ++d) {
        sum += q_ptr[d] * k_ptr[d];
    }
    scores[((b * H + h) * S + s1) * S + s2] = sum * scale;
}

__global__ void softmax_scores_kernel(float* __restrict__ scores, int B, int H, int S) {
    int row = blockIdx.x;
    int total_rows = B * H * S;
    if (row >= total_rows) return;

    float* row_ptr = scores + row * S;
    int tid = threadIdx.x;

    // max
    float max_val = -FLT_MAX;
    for (int i = tid; i < S; i += blockDim.x) {
        float v = row_ptr[i];
        if (v > max_val) max_val = v;
    }
    __shared__ float smax[256];
    smax[tid] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smax[tid + s] > smax[tid]) smax[tid] = smax[tid + s];
        __syncthreads();
    }
    max_val = smax[0];
    __syncthreads();

    // sum exp
    float sum_exp = 0.0f;
    for (int i = tid; i < S; i += blockDim.x) {
        float e = expf(row_ptr[i] - max_val);
        row_ptr[i] = e;
        sum_exp += e;
    }
    __shared__ float ssum[256];
    ssum[tid] = sum_exp;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) ssum[tid] += ssum[tid + s];
        __syncthreads();
    }
    sum_exp = ssum[0];
    float inv = 1.0f / sum_exp;
    __syncthreads();

    for (int i = tid; i < S; i += blockDim.x) {
        row_ptr[i] *= inv;
    }
}

__global__ void pv_kernel(const float* __restrict__ scores, const float* __restrict__ v, float* __restrict__ out,
                           int B, int H, int S, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S * D;
    if (idx >= total) return;

    int d = idx % D;
    int tmp = idx / D;
    int s1 = tmp % S;
    tmp = tmp / S;
    int h = tmp % H;
    int b = tmp / H;

    const float* score_row = scores + ((b * H + h) * S + s1) * S;
    const float* v_base = v + (b * H + h) * S * D;

    float sum = 0.0f;
    for (int s2 = 0; s2 < S; ++s2) {
        sum += score_row[s2] * v_base[s2 * D + d];
    }
    out[((b * H + h) * S + s1) * D + d] = sum;
}

torch::Tensor attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.dim() == 4 && q.scalar_type() == torch::kFloat32, "q invalid");
    TORCH_CHECK(k.is_cuda() && k.is_contiguous() && k.dim() == 4 && k.scalar_type() == torch::kFloat32, "k invalid");
    TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.dim() == 4 && v.scalar_type() == torch::kFloat32, "v invalid");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q,k,v shape mismatch");

    int B = q.size(0);
    int H = q.size(1);
    int S = q.size(2);
    int D = q.size(3);
    float scale = 1.0f / sqrtf(static_cast<float>(D));

    auto scores = torch::empty({B, H, S, S}, q.options());
    auto out = torch::empty_like(q);

    int total_qk = B * H * S * S;
    int threads = 256;
    int blocks_qk = (total_qk + threads - 1) / threads;
    qk_kernel<<<blocks_qk, threads>>>(q.data_ptr<float>(), k.data_ptr<float>(), scores.data_ptr<float>(),
                                       B, H, S, D, scale);

    int total_rows = B * H * S;
    softmax_scores_kernel<<<total_rows, threads>>>(scores.data_ptr<float>(), B, H, S);

    int total_pv = B * H * S * D;
    int blocks_pv = (total_pv + threads - 1) / threads;
    pv_kernel<<<blocks_pv, threads>>>(scores.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<float>(),
                                       B, H, S, D);

    return out;
}
"""

cpp_source = r"""
torch::Tensor attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v);
"""

module = load_inline(
    name="attention_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["attention_cuda"],
    verbose=False,
)

def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return module.attention_cuda(q, k, v)
