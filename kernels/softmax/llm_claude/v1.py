# Model used for generation: Claude Sonnet 5.5
# Kernel: Numerically Stable Row-wise Softmax
# Strategy: One CUDA block per row with shared-memory reductions

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <float.h>

#define BLOCK 256

__global__ void softmax_kernel(const float* x, float* out, int rows, int N) {
    __shared__ float sdata[BLOCK];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    const float* xr = x + (size_t)row * N;
    float* outr = out + (size_t)row * N;

    // 1) row max
    float local_max = -FLT_MAX;
    for (int i = tid; i < N; i += BLOCK) local_max = fmaxf(local_max, xr[i]);
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // 2) exponentials and sum
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += BLOCK) {
        float e = expf(xr[i] - max_val);
        outr[i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float total = sdata[0];

    // 3) normalize
    for (int i = tid; i < N; i += BLOCK) outr[i] /= total;
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (rows, N)");
    x = x.contiguous();
    int rows = x.size(0);
    int N = x.size(1);
    auto out = torch::empty_like(x);
    if (rows == 0 || N == 0) return out;

    softmax_kernel<<<rows, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), rows, N);
    return out;
}
"""

cpp_source = "torch::Tensor softmax_cuda(torch::Tensor x);"

module = load_inline(
    name="softmax_module",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["softmax_cuda"],
    verbose=False,
)

def run(x: torch.Tensor) -> torch.Tensor:
    return module.softmax_cuda(x)
