# Model used for generation: Claude Sonnet 5.5
# Kernel: Direct 2D Convolution (valid, stride 1, no bias; cross-correlation like torch.nn.functional.conv2d)
# Strategy: One thread per output element, accumulating over input channels and kernel window

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define BLOCK 256

__global__ void conv2d_kernel(const float* X, const float* W, float* Y,
                               int B, int IC, int H, int Wd,
                               int OC, int KH, int KW, int OH, int OW) {
    size_t total = (size_t)B * OC * OH * OW;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int ow = idx % OW;
    int oh = (idx / OW) % OH;
    int oc = (idx / ((size_t)OW * OH)) % OC;
    int b = idx / ((size_t)OW * OH * OC);

    float acc = 0.0f;
    for (int ic = 0; ic < IC; ++ic) {
        const float* xp = X + (((size_t)b * IC + ic) * H) * Wd;
        const float* wp = W + (((size_t)oc * IC + ic) * KH) * KW;
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                acc += xp[(size_t)(oh + kh) * Wd + (ow + kw)] * wp[kh * KW + kw];
            }
        }
    }
    Y[idx] = acc;
}

torch::Tensor conv2d_cuda(torch::Tensor X, torch::Tensor W) {
    TORCH_CHECK(X.is_cuda() && W.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 && W.scalar_type() == torch::kFloat32,
                "inputs must be float32");
    TORCH_CHECK(X.dim() == 4 && W.dim() == 4, "X must be (B,IC,H,W), W must be (OC,IC,KH,KW)");
    TORCH_CHECK(X.size(1) == W.size(1), "channel mismatch");
    X = X.contiguous();
    W = W.contiguous();

    int B = X.size(0), IC = X.size(1), H = X.size(2), Wd = X.size(3);
    int OC = W.size(0), KH = W.size(2), KW = W.size(3);
    TORCH_CHECK(H >= KH && Wd >= KW, "Kernel dimensions exceed input dimensions");
    int OH = H - KH + 1;
    int OW = Wd - KW + 1;

    auto Y = torch::empty({B, OC, OH, OW}, X.options());
    size_t total = (size_t)B * OC * OH * OW;
    if (total == 0) return Y;

    int blocks = (int)((total + BLOCK - 1) / BLOCK);
    conv2d_kernel<<<blocks, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        X.data_ptr<float>(), W.data_ptr<float>(), Y.data_ptr<float>(),
        B, IC, H, Wd, OC, KH, KW, OH, OW);
    return Y;
}
"""

cpp_source = "torch::Tensor conv2d_cuda(torch::Tensor X, torch::Tensor W);"

module = load_inline(
    name="conv2d_module",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv2d_cuda"],
    verbose=False,
)

def run(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return module.conv2d_cuda(x, weight)
