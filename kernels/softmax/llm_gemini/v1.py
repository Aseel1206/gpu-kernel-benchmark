# Model: Gemini
# Kernel: 2. Row-Wise Numerically-Stable Softmax (one block per row, warp shuffles + shared memory)
import torch
from torch.utils.cpp_extension import load_inline

cpp_source = """
#include <torch/extension.h>
torch::Tensor launch_softmax(torch::Tensor x);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, offset));
    }
    return val;
}

__inline__ __device__ float warp_reduce_sum_f(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

__global__ void row_softmax_kernel(const float* __restrict__ x,
                                    float* __restrict__ y, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int num_warps = (blockDim.x + 31) / 32;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    extern __shared__ float smem[];
    float* warp_buf = smem;
    float* shared_scalar = smem + num_warps;

    const float* row_x = x + static_cast<int64_t>(row) * cols;
    float* row_y = y + static_cast<int64_t>(row) * cols;

    float local_max = -FLT_MAX;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, row_x[col]);
    }
    float w_max = warp_reduce_max(local_max);
    if (lane == 0) {
        warp_buf[warp_id] = w_max;
    }
    __syncthreads();

    if (tid == 0) {
        float b_max = -FLT_MAX;
        for (int w = 0; w < num_warps; ++w) {
            b_max = fmaxf(b_max, warp_buf[w]);
        }
        shared_scalar[0] = b_max;
    }
    __syncthreads();
    float row_max = shared_scalar[0];

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += expf(row_x[col] - row_max);
    }
    float w_sum = warp_reduce_sum_f(local_sum);
    if (lane == 0) {
        warp_buf[warp_id] = w_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float b_sum = 0.0f;
        for (int w = 0; w < num_warps; ++w) {
            b_sum += warp_buf[w];
        }
        shared_scalar[1] = b_sum;
    }
    __syncthreads();
    float inv_sum = 1.0f / shared_scalar[1];

    for (int col = tid; col < cols; col += blockDim.x) {
        row_y[col] = expf(row_x[col] - row_max) * inv_sum;
    }
}

torch::Tensor launch_softmax(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int rows = static_cast<int>(x.size(0));
    int cols = static_cast<int>(x.size(1));
    auto y = torch::empty_like(x);
    if (rows == 0 || cols == 0) {
        return y;
    }

    int threads = 256;
    if (cols <= 128) threads = 128;
    else if (cols <= 256) threads = 256;
    else if (cols <= 1024) threads = 512;
    else threads = 1024;

    int num_warps = (threads + 31) / 32;
    size_t smem_bytes = (num_warps + 2) * sizeof(float);

    row_softmax_kernel<<<rows, threads, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), rows, cols
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""

_ext = load_inline(
    name="cuda_softmax_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["launch_softmax"],
    verbose=False,
)

def run(x: torch.Tensor) -> torch.Tensor:
    """
    x: 2D float32 tensor, shape (batch, N). N varies across test cases (e.g. 128 to 8192),
       not guaranteed to be a power of 2.
    returns: tensor of the same shape, softmax applied along dim=-1 (each row sums to 1).
    """
    return _ext.launch_softmax(x)
