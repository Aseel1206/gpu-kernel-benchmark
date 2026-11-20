# Model: Grok (xAI)
# Kernel: Reduction

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void reduce_sum_kernel(const float* __restrict__ x, float* __restrict__ partial, int N) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    while (i < N) {
        sum += x[i];
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_sum_final_kernel(const float* __restrict__ partial, float* __restrict__ out, int n_partials) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < n_partials; i += blockDim.x) {
        sum += partial[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[0] = sdata[0];
    }
}

torch::Tensor reduce_sum_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 1 && x.scalar_type() == torch::kFloat32, "x must be 1D float32 CUDA contiguous");
    int N = x.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    if (N == 0) {
        return torch::zeros({}, options);
    }

    const int threads = BLOCK_SIZE;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;

    auto partial = torch::empty({blocks}, options);
    reduce_sum_kernel<<<blocks, threads>>>(x.data_ptr<float>(), partial.data_ptr<float>(), N);

    auto out = torch::empty({}, options);
    reduce_sum_final_kernel<<<1, threads>>>(partial.data_ptr<float>(), out.data_ptr<float>(), blocks);
    return out;
}
"""

cpp_source = r"""
torch::Tensor reduce_sum_cuda(torch::Tensor x);
"""

module = load_inline(
    name="reduce_sum_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["reduce_sum_cuda"],
    verbose=False,
)

def run(x: torch.Tensor) -> torch.Tensor:
    return module.reduce_sum_cuda(x)
