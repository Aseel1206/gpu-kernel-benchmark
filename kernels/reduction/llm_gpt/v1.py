# Model used for generation: GPT-6
# Kernel: Parallel Sum Reduction
# Strategy: Two-stage shared-memory reduction

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK 256

__global__ void reduce_kernel(
    const float* x,
    float* partial,
    int N
) {
    __shared__ float sdata[BLOCK];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float sum = 0.0f;
    for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
        sum += x[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

__global__ void final_reduce_kernel(
    const float* partial,
    float* output,
    int N
) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < N; i += BLOCK) {
        sum += partial[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = sdata[0];
    }
}

torch::Tensor reduction_cuda(torch::Tensor x) {

    int N = x.numel();

    auto output = torch::empty(
        {},
        x.options()
    );

    if (N == 0) {
        output.zero_();
        return output;
    }

    int blocks = min(
        (N + BLOCK - 1) / BLOCK,
        1024
    );

    auto partial = torch::empty(
        {blocks},
        x.options()
    );

    reduce_kernel<<<blocks, BLOCK>>>(
        x.data_ptr<float>(),
        partial.data_ptr<float>(),
        N
    );

    final_reduce_kernel<<<1, BLOCK>>>(
        partial.data_ptr<float>(),
        output.data_ptr<float>(),
        blocks
    );

    return output;
}
"""

cpp_source = """
torch::Tensor reduction_cuda(torch::Tensor x);
"""

module = load_inline(
    name="reduction_module",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["reduction_cuda"],
    verbose=False
)

def run(x: torch.Tensor) -> torch.Tensor:
    return module.reduction_cuda(x)
