# Model: DeepSeek (latest), Kernel: 2. Softmax (row-wise, numerically stable)
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ out, int N) {
    int row = blockIdx.x;
    const float* xr = x + (long)row * N;
    float* orow = out + (long)row * N;
    extern __shared__ float sdata[];

    // 1) row max
    float lmax = -INFINITY;
    for (int i = threadIdx.x; i < N; i += blockDim.x) lmax = fmaxf(lmax, xr[i]);
    sdata[threadIdx.x] = lmax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // 2) sum of exp(x - max)
    float lsum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) lsum += __expf(xr[i] - row_max);
    sdata[threadIdx.x] = lsum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float row_sum = sdata[0];

    // 3) normalize
    float inv = 1.0f / row_sum;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        orow[i] = __expf(xr[i] - row_max) * inv;
    }
}

torch::Tensor softmax_launcher(torch::Tensor x) {
    int batch = (int)x.size(0);
    int N = (int)x.size(1);
    auto out = torch::empty_like(x);
    int threads = 256;
    dim3 blocks(batch);
    softmax_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), N);
    return out;
}
"""

cpp_source = "torch::Tensor softmax_launcher(torch::Tensor x);"

ext = load_inline(
    name="softmax_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["softmax_launcher"],
    verbose=False,
)

def run(x: torch.Tensor) -> torch.Tensor:
    return ext.softmax_launcher(x)
