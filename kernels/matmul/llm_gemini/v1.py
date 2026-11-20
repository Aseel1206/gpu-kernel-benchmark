# Model: Gemini
# Kernel: 3. Dense Matrix Multiplication (shared-memory tiled matmul with boundary checks)
import torch
from torch.utils.cpp_extension import load_inline

cpp_source = """
#include <torch/extension.h>
torch::Tensor launch_matmul(torch::Tensor a, torch::Tensor b);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

__global__ void tiled_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y;

    int row = blockIdx.y * TILE_M + ty;
    int col = blockIdx.x * TILE_N + tx;

    float acc = 0.0f;

    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        for (int idx = tid; idx < TILE_M * TILE_K; idx += num_threads) {
            int lr = idx / TILE_K;
            int lc = idx % TILE_K;
            int gr = blockIdx.y * TILE_M + lr;
            int gc = k_tile + lc;
            As[lr][lc] = (gr < M && gc < K) ? A[static_cast<int64_t>(gr) * K + gc] : 0.0f;
        }
        for (int idx = tid; idx < TILE_K * TILE_N; idx += num_threads) {
            int lr = idx / TILE_N;
            int lc = idx % TILE_N;
            int gr = k_tile + lr;
            int gc = blockIdx.x * TILE_N + lc;
            Bs[lr][lc] = (gr < K && gc < N) ? B[static_cast<int64_t>(gr) * N + gc] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[static_cast<int64_t>(row) * N + col] = acc;
    }
}

torch::Tensor launch_matmul(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "Inputs must be float32");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "Incompatible dimensions for matmul");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "Inputs must be contiguous");

    int M = static_cast<int>(a.size(0));
    int K = static_cast<int>(a.size(1));
    int N = static_cast<int>(b.size(1));

    auto c = torch::empty({M, N}, a.options());
    if (M == 0 || K == 0 || N == 0) {
        c.zero_();
        return c;
    }

    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    tiled_matmul_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, K, N
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}
"""

_ext = load_inline(
    name="cuda_matmul_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["launch_matmul"],
    verbose=False,
)

def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: 2D float32 tensor, shape (M, K).
    b: 2D float32 tensor, shape (K, N).
    returns: 2D float32 tensor, shape (M, N), equal to a @ b.
    """
    return _ext.launch_matmul(a, b)
