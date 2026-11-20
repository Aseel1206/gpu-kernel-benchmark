# Model used for generation: GPT-6
# Kernel: Tiled CUDA Matrix Multiplication
# Strategy: 16x16 shared-memory tiling

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE 16

__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;

    int tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < tiles; t++) {

        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;

        // Load A tile
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B tile
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Multiply accumulated tiles
        for (int i = 0; i < TILE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(
    torch::Tensor A,
    torch::Tensor B
) {

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty(
        {M, N},
        A.options()
    );

    if (M == 0 || N == 0) {
        return C;
    }

    dim3 threads(TILE, TILE);
    dim3 blocks(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    return C;
}
"""

cpp_source = """
torch::Tensor matmul_cuda(
    torch::Tensor A,
    torch::Tensor B
);
"""

module = load_inline(
    name="matmul_module",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_cuda"],
    verbose=False
)

def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return module.matmul_cuda(a, b)
