# Model: Grok (xAI)
# Kernel: Conv2D

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
                               int B, int IC, int H, int W, int OC, int KH, int KW, int OH, int OW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * OC * OH * OW;
    if (idx >= total) return;

    int ow = idx % OW;
    int tmp = idx / OW;
    int oh = tmp % OH;
    tmp = tmp / OH;
    int oc = tmp % OC;
    int b = tmp / OC;

    float sum = 0.0f;
    for (int ic = 0; ic < IC; ++ic) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                int ih = oh + kh;
                int iw = ow + kw;
                float xv = x[((b * IC + ic) * H + ih) * W + iw];
                float wv = w[((oc * IC + ic) * KH + kh) * KW + kw];
                sum += xv * wv;
            }
        }
    }
    y[((b * OC + oc) * OH + oh) * OW + ow] = sum;
}

torch::Tensor conv2d_cuda(torch::Tensor x, torch::Tensor weight) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 4 && x.scalar_type() == torch::kFloat32, "x invalid");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.dim() == 4 && weight.scalar_type() == torch::kFloat32, "weight invalid");
    TORCH_CHECK(x.size(1) == weight.size(1), "in_channels mismatch");

    int B = x.size(0);
    int IC = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int OC = weight.size(0);
    int KH = weight.size(2);
    int KW = weight.size(3);
    int OH = H - KH + 1;
    int OW = W - KW + 1;

    TORCH_CHECK(OH > 0 && OW > 0, "kernel larger than input");

    auto y = torch::empty({B, OC, OH, OW}, x.options());
    int total = B * OC * OH * OW;
    if (total == 0) return y;

    const int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv2d_kernel<<<blocks, threads>>>(x.data_ptr<float>(), weight.data_ptr<float>(), y.data_ptr<float>(),
                                        B, IC, H, W, OC, KH, KW, OH, OW);
    return y;
}
"""

cpp_source = r"""
torch::Tensor conv2d_cuda(torch::Tensor x, torch::Tensor weight);
"""

module = load_inline(
    name="conv2d_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv2d_cuda"],
    verbose=False,
)

def run(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return module.conv2d_cuda(x, weight)
