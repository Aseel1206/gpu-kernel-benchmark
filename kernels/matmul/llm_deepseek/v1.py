# Model: DeepSeek (latest), Kernel: 3. Matmul (tiled shared-memory)
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE 16

__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B,
                               float* __restrict__ C, int M, int K, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        int aCol = k0 + threadIdx.x;
        int bRow = k0 + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * (long)K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * (long)N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) C[row * (long)N + col] = acc;
}

torch::Tensor matmul_launcher(torch::Tensor a, torch::Tensor b) {
    int M = (int)a.size(0);
    int K = (int)a.size(1);
    int N = (int)b.size(1);
    auto c = torch::empty({M, N}, a.options());
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_kernel<<<grid, block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, K, N);
    return c;
}
"""

cpp_source = "torch::Tensor matmul_launcher(torch::Tensor a, torch::Tensor b);"

ext = load_inline(
    name="matmul_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_launcher"],
    verbose=False,
)

def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ext.matmul_launcher(a, b)
