# Model used for generation: Claude Sonnet 5.5
# Kernel: Fused Scaled Dot-Product Attention
# Strategy: One CUDA block per (batch, head, query); scores kept in dynamic shared memory,
#           stable softmax via block reductions, then weighted sum of V (no global attention matrix)

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <float.h>

#define BLOCK 128

// Q: (B,H,Sq,D) K,V: (B,H,Sk,D) O: (B,H,Sq,D)
// Dynamic shared memory layout: scores[Sk] followed by reduction[BLOCK]
__global__ void attention_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                  const float* __restrict__ V,
                                  float* __restrict__ O, int H, int Sq, int Sk,
                                  int D) {
    extern __shared__ float smem[];
    float* scores = smem;
    float* reduction = smem + Sk;

    int tid = threadIdx.x;
    size_t qi = blockIdx.x;                       // flattened (b, h, q)
    int q = qi % Sq;
    size_t bh = qi / Sq;                           // flattened (b, h)

    const float* qp = Q + (bh * Sq + q) * D;
    const float* kp = K + bh * (size_t)Sk * D;
    const float* vp = V + bh * (size_t)Sk * D;
    float* op = O + (bh * Sq + q) * D;

    float scale = rsqrtf((float)D);

    // Step 1: scores = (Q . K^T) * scale, track local max
    float local_max = -FLT_MAX;
    for (int k = tid; k < Sk; k += BLOCK) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += qp[d] * kp[(size_t)k * D + d];
        float s = dot * scale;
        scores[k] = s;
        local_max = fmaxf(local_max, s);
    }

    // Step 2: max reduction
    reduction[tid] = local_max;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) reduction[tid] = fmaxf(reduction[tid], reduction[tid + s]);
        __syncthreads();
    }
    float max_score = reduction[0];
    __syncthreads();

    // Step 3: exponentials and local sum
    float local_sum = 0.0f;
    for (int k = tid; k < Sk; k += BLOCK) {
        float e = expf(scores[k] - max_score);
        scores[k] = e;
        local_sum += e;
    }

    // Step 4: sum reduction
    reduction[tid] = local_sum;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) reduction[tid] += reduction[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / reduction[0];
    __syncthreads();

    // Step 5: O = softmax(scores) @ V
    for (int d = tid; d < D; d += BLOCK) {
        float acc = 0.0f;
        for (int k = 0; k < Sk; ++k) {
            acc += scores[k] * vp[(size_t)k * D + d];
        }
        op[d] = acc * inv_sum;
    }
}

torch::Tensor attention_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32 && K.scalar_type() == torch::kFloat32 &&
                V.scalar_type() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4, "inputs must be (B,H,S,D)");
    TORCH_CHECK(K.sizes() == V.sizes(), "K and V must have the same shape");
    TORCH_CHECK(Q.size(0) == K.size(0) && Q.size(1) == K.size(1) && Q.size(3) == K.size(3),
                "Q and K must agree on B, H, D");
    Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous();

    int B = Q.size(0), H = Q.size(1), Sq = Q.size(2), D = Q.size(3);
    int Sk = K.size(2);

    auto O = torch::zeros_like(Q);
    if (B == 0 || H == 0 || Sq == 0 || D == 0) return O;
    TORCH_CHECK(Sk > 0, "key sequence length must be > 0");

    size_t smem = ((size_t)Sk + BLOCK) * sizeof(float);
    int device;
    cudaGetDevice(&device);
    int max_optin = 0;
    cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    TORCH_CHECK(smem <= (size_t)max_optin,
                "Sk too large: needs ", smem, " bytes of shared memory, device allows ", max_optin);
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(attention_kernel,
                              cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    }

    size_t grid = (size_t)B * H * Sq;
    attention_kernel<<<(unsigned int)grid, BLOCK, smem, at::cuda::getCurrentCUDAStream()>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), H, Sq, Sk, D);
    return O;
}
"""

cpp_source = "torch::Tensor attention_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V);"

module = load_inline(
    name="attention_module",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["attention_cuda"],
    verbose=False,
)

def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return module.attention_cuda(q, k, v)
