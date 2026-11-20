# Model used for generation: Claude Sonnet 5.5
# Kernel: Parallel Sum Reduction
# Strategy: Two-stage shared-memory reduction (grid-stride loop, then single-block final pass)

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define BLOCK 256

__global__ void reduce_kernel(const float* x, float* out, int N)
{
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    size_t idx = (size_t)blockIdx.x * blockDim.x + tid;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    float sum = 0.0f;
    for (size_t i = idx; i < (size_t)N; i += stride) {
        sum += x[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

torch::Tensor reduce_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    x = x.contiguous();
    int N = (int)x.numel();

    auto out = torch::zeros({1}, x.options());
    if (N == 0) return out.reshape({});

    int blocks = (N + BLOCK - 1) / BLOCK;
    if (blocks > 1024) blocks = 1024;

    auto partial = torch::empty({blocks}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Stage 1: per-block partial sums
    reduce_kernel<<<blocks, BLOCK, 0, stream>>>(
        x.data_ptr<float>(), partial.data_ptr<float>(), N);
    // Stage 2: combine partials with a single block
    reduce_kernel<<<1, BLOCK, 0, stream>>>(
        partial.data_ptr<float>(), out.data_ptr<float>(), blocks);

    return out.reshape({});
}
"""

cpp_source = "torch::Tensor reduce_cuda(torch::Tensor x);"

module = load_inline(
    name="reduce_module",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["reduce_cuda"],
    verbose=False,
)

def run(x: torch.Tensor) -> torch.Tensor:
    return module.reduce_cuda(x)
