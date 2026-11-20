# Model: Grok (xAI)
# Kernel: Matmul

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE 16

__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                               int M, int K, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;

        if (row < M && a_col < K)
            As[ty][tx] = A[row * K + a_col];
        else
            As[ty][tx] = 0.0f;

        if (b_row < K && col < N)
            Bs[ty][tx] = B[b_row * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.dim() == 2 && a.scalar_type() == torch::kFloat32, "a invalid");
    TORCH_CHECK(b.is_cuda() && b.is_contiguous() && b.dim() == 2 && b.scalar_type() == torch::kFloat32, "b invalid");
    TORCH_CHECK(a.size(1) == b.size(0), "inner dims must match");

    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    auto c = torch::empty({M, N}, a.options());

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    matmul_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, K, N);
    return c;
}
"""

cpp_source = r"""
torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b);
"""

module = load_inline(
    name="matmul_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_cuda"],
    verbose=False,
)

def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return module.matmul_cuda(a, b)
