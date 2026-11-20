# Model used for generation: GPT-6
# Kernel: Numerically Stable Row-wise Softmax
# Strategy: One CUDA block per row with shared-memory reductions

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

#define BLOCK 256

__global__ void softmax_kernel(
    const float* x,
    float* output,
    int rows,
    int N
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    __shared__ float sdata[BLOCK];

    float local_max = -FLT_MAX;

    for (int i = tid; i < N; i += BLOCK) {
        local_max = fmaxf(
            local_max,
            x[row * N + i]
        );
    }

    sdata[tid] = local_max;
    __syncthreads();

    // Parallel maximum reduction
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(
                sdata[tid],
                sdata[tid + s]
            );
        }
        __syncthreads();
    }

    float max_val = sdata[0];

    // Calculate exponentials
    float local_sum = 0.0f;

    for (int i = tid; i < N; i += BLOCK) {
        float val = expf(
            x[row * N + i] - max_val
        );

        output[row * N + i] = val;
        local_sum += val;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    // Parallel sum reduction
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float sum = sdata[0];

    // Normalize
    for (int i = tid; i < N; i += BLOCK) {
        output[row * N + i] /= sum;
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {

    int rows = x.size(0);
    int N = x.size(1);

    auto output = torch::empty_like(x);

    if (rows == 0 || N == 0) {
        return output;
    }

    softmax_kernel<<<rows, BLOCK>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        N
    );

    return output;
}
"""

cpp_source = """
torch::Tensor softmax_cuda(torch::Tensor x);
"""

module = load_inline(
    name="softmax_module",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["softmax_cuda"],
    verbose=False
)

def run(x: torch.Tensor) -> torch.Tensor:
    return module.softmax_cuda(x)
