# Model used for generation: GPT-6
# Kernel: Direct 2D Convolution
# Strategy: Parallel output calculation with register accumulation

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK 256

__global__ void conv2d_kernel(
    const float* X,
    const float* W,
    float* Y,
    int B,
    int IC,
    int H,
    int Width,
    int OC,
    int KH,
    int KW,
    int OH,
    int OW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * OC * OH * OW;

    if (idx >= total) return;

    int ow = idx % OW;
    int oh = (idx / OW) % OH;
    int oc = (idx / (OW * OH)) % OC;
    int b = idx / (OW * OH * OC);

    float sum = 0.0f;

    // Iterate over input channels
    for (int ic = 0; ic < IC; ic++) {

        // Iterate over convolution kernel
        for (int kh = 0; kh < KH; kh++) {

            for (int kw = 0; kw < KW; kw++) {

                int ih = oh + kh;
                int iw = ow + kw;

                int x_idx =
                    ((b * IC + ic) * H + ih) * Width + iw;

                int w_idx =
                    ((oc * IC + ic) * KH + kh) * KW + kw;

                sum += X[x_idx] * W[w_idx];
            }
        }
    }

    Y[idx] = sum;
}

torch::Tensor conv2d_cuda(
    torch::Tensor X,
    torch::Tensor W
) {

    int B = X.size(0);
    int IC = X.size(1);
    int H = X.size(2);
    int Width = X.size(3);

    int OC = W.size(0);
    int KH = W.size(2);
    int KW = W.size(3);

    int OH = H - KH + 1;
    int OW = Width - KW + 1;

    TORCH_CHECK(OH >= 0 && OW >= 0,
                "Kernel dimensions exceed input dimensions");

    auto Y = torch::empty(
        {B, OC, OH, OW},
        X.options()
    );

    int total = B * OC * OH * OW;

    if (total == 0) {
        return Y;
    }

    int blocks = (total + BLOCK - 1) / BLOCK;

    conv2d_kernel<<<blocks, BLOCK>>>(
        X.data_ptr<float>(),
        W.data_ptr<float>(),
        Y.data_ptr<float>(),
        B, IC, H, Width,
        OC, KH, KW, OH, OW
    );

    return Y;
}
"""

cpp_source = """
torch::Tensor conv2d_cuda(
    torch::Tensor X,
    torch::Tensor W
);
"""

module = load_inline(
    name="conv2d_module",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv2d_cuda"],
    verbose=False
)

def run(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return module.conv2d_cuda(x, weight)
