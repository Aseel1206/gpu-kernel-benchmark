# Model: DeepSeek (latest), Kernel: 1. Reduction (parallel sum over 1D tensor)
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void reduce_kernel(const float* __restrict__ x, float* __restrict__ out, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float val = (idx < N) ? x[idx] : 0.0f;
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

torch::Tensor reduce_launcher(torch::Tensor x) {
    auto out = torch::zeros({}, x.options());
    int N = (int)x.numel();
    if (N == 0) return out;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), N);
    return out;
}
"""

cpp_source = "torch::Tensor reduce_launcher(torch::Tensor x);"

ext = load_inline(
    name="reduce_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["reduce_launcher"],
    verbose=False,
)

def run(x: torch.Tensor) -> torch.Tensor:
    return ext.reduce_launcher(x)
