# Model used for generation: GPT-6
# Kernel: Scaled Dot-Product Attention
# Strategy: Fused CUDA attention with shared-memory softmax
# Notes: Fused implementation; does not materialize the global attention matrix

import torch
import math
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define BLOCK 256

__global__ void attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B,
    int H,
    int S,
    int D
) {
    int tid = threadIdx.x;

    int query_idx = blockIdx.x % S;
    int head_idx = (blockIdx.x / S) % H;
    int batch_idx = blockIdx.x / (S * H);

    // Each block handles one query position.
    // Dynamic shared memory stores attention scores
    // and the reduction buffer.

    extern __shared__ float shared[];
    float* scores = shared;
    float* reduction = shared + S;

    int q_offset =
        ((batch_idx * H + head_idx) * S + query_idx) * D;

    int head_offset =
        (batch_idx * H + head_idx) * S * D;

    float scale = rsqrtf((float)D);

    // Step 1: Calculate Q.K^T
    float local_max = -FLT_MAX;

    for (int key_idx = tid; key_idx < S; key_idx += BLOCK) {

        float dot = 0.0f;

        for (int d = 0; d < D; d++) {
            float q = Q[q_offset + d];
            float k = K[head_offset + key_idx * D + d];
            dot += q * k;
        }

        float score = dot * scale;
        scores[key_idx] = score;

        local_max = fmaxf(local_max, score);
    }

    // Step 2: Parallel maximum reduction

    reduction[tid] = local_max;
    __syncthreads();

    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            reduction[tid] = fmaxf(
                reduction[tid],
                reduction[tid + s]
            );
        }
        __syncthreads();
    }

    float max_score = reduction[0];

    // Step 3: Calculate stable exponentials

    float local_sum = 0.0f;

    for (int key_idx = tid; key_idx < S; key_idx += BLOCK) {
        float e = expf(scores[key_idx] - max_score);
        scores[key_idx] = e;
        local_sum += e;
    }

    // Step 4: Parallel sum reduction

    reduction[tid] = local_sum;
    __syncthreads();

    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            reduction[tid] += reduction[tid + s];
        }
        __syncthreads();
    }

    float denominator = reduction[0];

    // Normalize attention weights
    for (int key_idx = tid; key_idx < S; key_idx += BLOCK) {
        scores[key_idx] /= denominator;
    }

    __syncthreads();

    // Step 5: Weighted sum of V
    for (int d = tid; d < D; d += BLOCK) {
        float result = 0.0f;

        for (int key_idx = 0; key_idx < S; key_idx++) {
            float weight = scores[key_idx];
            float value =
                V[head_offset + key_idx * D + d];
            result += weight * value;
        }

        O[q_offset + d] = result;
    }
}

torch::Tensor attention_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {

    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);

    auto O = torch::empty_like(Q);

    if (B == 0 || H == 0 || S == 0 || D == 0) {
        return O;
    }

    int total_blocks = B * H * S;

    size_t shared_memory =
        (S + BLOCK) * sizeof(float);

    attention_kernel<<<
        total_blocks,
        BLOCK,
        shared_memory
    >>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        B, H, S, D
    );

    return O;
}
"""

cpp_source = """
torch::Tensor attention_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
);
"""

module = load_inline(
    name="attention_module",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["attention_cuda"],
    verbose=False
)

def run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor
) -> torch.Tensor:
    return module.attention_cuda(q, k, v)
