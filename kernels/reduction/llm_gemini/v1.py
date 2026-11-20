# Model: Gemini
# Kernel: 1. Parallel 1D Sum Reduction (shared-memory tree reduction + warp __shfl_down_sync + atomicAdd)
import torch
from torch.utils.cpp_extension import load_inline

cpp_source = """
#include <torch/extension.h>
torch::Tensor launch_reduction(torch::Tensor x);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

__global__ void reduce_sum_kernel(const float* __restrict__ x,
                                   float* __restrict__ out, int64_t N) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    float sum = 0.0f;
    for (int64_t i = idx; i < N; i += stride) {
        sum += x[i];
    }

    smem[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        float val = (blockDim.x >= 32) ? smem[tid] : ((tid < blockDim.x) ? smem[tid] : 0.0f);
        val = warp_reduce_sum(val);
        if (tid == 0) {
            atomicAdd(out, val);
        }
    }
}

torch::Tensor launch_reduction(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int64_t N = x.numel();
    auto out = torch::zeros({}, x.options());
    if (N == 0) {
        return out;
    }

    const int threads = 256;
    int blocks = static_cast<int>(std::min<int64_t>((N + threads - 1) / threads, 65535));
    size_t smem_bytes = threads * sizeof(float);

    reduce_sum_kernel<<<blocks, threads, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), N
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

_ext = load_inline(
    name="cuda_reduction_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["launch_reduction"],
    verbose=False,
)

def run(x: torch.Tensor) -> torch.Tensor:
    """
    x: 1D float32 tensor, shape (N,), N not guaranteed to be a power of 2 or a
       multiple of any particular block size.
    returns: 0-dimensional (scalar) tensor containing sum(x).
    """
    return _ext.launch_reduction(x.reshape(-1))
