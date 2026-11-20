# Model: Grok (xAI)
# Kernel: Softmax

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

#define BLOCK_SIZE 256

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int batch, int N) {
    int row = blockIdx.x;
    if (row >= batch) return;

    const float* row_x = x + row * N;
    float* row_y = y + row * N;

    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;

    // 1. Find max
    float max_val = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = row_x[i];
        if (v > max_val) max_val = v;
    }
    sdata[tid] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // 2. Compute sum of exp(x - max)
    float sum_exp = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum_exp += expf(row_x[i] - max_val);
    }
    sdata[tid] = sum_exp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    sum_exp = sdata[0];
    __syncthreads();

    // 3. Normalize
    float inv_sum = 1.0f / sum_exp;
    for (int i = tid; i < N; i += blockDim.x) {
        row_y[i] = expf(row_x[i] - max_val) * inv_sum;
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 && x.scalar_type() == torch::kFloat32, "x must be 2D float32 CUDA contiguous");
    int batch = x.size(0);
    int N = x.size(1);
    auto y = torch::empty_like(x);

    if (batch == 0 || N == 0) return y;

    const int threads = BLOCK_SIZE;
    softmax_kernel<<<batch, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), batch, N);
    return y;
}
"""

cpp_source = r"""
torch::Tensor softmax_cuda(torch::Tensor x);
"""

module = load_inline(
    name="softmax_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["softmax_cuda"],
    verbose=False,
)

def run(x: torch.Tensor) -> torch.Tensor:
    return module.softmax_cuda(x)
