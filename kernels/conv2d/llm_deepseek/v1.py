# Model: DeepSeek (latest), Kernel: 4. Conv2D (direct, no padding, stride 1)
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* __restrict__ x, const float* __restrict__ w,
                               float* __restrict__ y, int B, int C, int H, int W,
                               int OC, int KH, int KW, int OH, int OW) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)B * OC * OH * OW;
    if (idx >= total) return;

    int ow = (int)(idx % OW);
    int oh = (int)((idx / OW) % OH);
    int oc = (int)((idx / ((long)OW * OH)) % OC);
    int b = (int)(idx / ((long)OW * OH * OC));

    const float* xb = x + (long)b * C * H * W;
    const float* woc = w + (long)oc * C * KH * KW;

    float acc = 0.0f;
    for (int c = 0; c < C; c++) {
        const float* xc = xb + (long)c * H * W;
        const float* wc = woc + (long)c * KH * KW;
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                acc += xc[(oh + kh) * W + (ow + kw)] * wc[kh * KW + kw];
            }
        }
    }
    y[idx] = acc;
}

torch::Tensor conv2d_launcher(torch::Tensor x, torch::Tensor weight) {
    int B = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int OC = (int)weight.size(0);
    int KH = (int)weight.size(2);
    int KW = (int)weight.size(3);
    int OH = H - KH + 1;
    int OW = W - KW + 1;
    auto y = torch::empty({B, OC, OH, OW}, x.options());

    long total = (long)B * OC * OH * OW;
    int threads = 256;
    long blocks = (total + threads - 1) / threads;
    conv2d_kernel<<<(int)blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), y.data_ptr<float>(),
        B, C, H, W, OC, KH, KW, OH, OW);
    return y;
}
"""

cpp_source = "torch::Tensor conv2d_launcher(torch::Tensor x, torch::Tensor weight);"

ext = load_inline(
    name="conv2d_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv2d_launcher"],
    verbose=False,
)

def run(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return ext.conv2d_launcher(x, weight)
