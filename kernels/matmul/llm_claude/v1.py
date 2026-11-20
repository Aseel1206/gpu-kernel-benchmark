# Model used for generation: Claude Sonnet 5.5
# Kernel: Tiled Matrix Multiplication
# Strategy: 16x16 shared-memory tiling with boundary guards for arbitrary M, K, N

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define TILE 16

__global__ void matmul_kernel(const float* A, const float* B, float* C,
                               int M, int K, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[(size_t)row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[(size_t)b_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32,
                "inputs must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "inputs must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "inner dimensions must match");
    A = A.contiguous();
    B = B.contiguous();

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());
    if (M == 0 || N == 0 || K == 0) return C;

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    return C;
}
"""

cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

module = load_inline(
    name="matmul_module",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_cuda"],
    verbose=False,
)

def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return module.matmul_cuda(a, b)
