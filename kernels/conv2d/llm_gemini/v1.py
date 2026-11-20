# Model: Gemini
# Kernel: 4. Direct 2D Convolution (stride 1, no padding, no bias, register accumulation)
import torch
from torch.utils.cpp_extension import load_inline

cpp_source = """
#include <torch/extension.h>
torch::Tensor launch_conv2d(torch::Tensor x, torch::Tensor weight);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void direct_conv2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int batch, int in_c, int H, int W,
    int out_c, int kh, int kw,
    int out_h, int out_w
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    int oc = z % out_c;
    int b = z / out_c;

    if (b >= batch || oh >= out_h || ow >= out_w) return;

    float acc = 0.0f;
    for (int ic = 0; ic < in_c; ++ic) {
        const float* x_plane = x + (((static_cast<int64_t>(b) * in_c + ic) * H) * W);
        const float* w_plane = weight + (((static_cast<int64_t>(oc) * in_c + ic) * kh) * kw);
        for (int r = 0; r < kh; ++r) {
            int ih = oh + r;
            const float* x_row = x_plane + static_cast<int64_t>(ih) * W;
            const float* w_row = w_plane + static_cast<int64_t>(r) * kw;
            for (int c = 0; c < kw; ++c) {
                int iw = ow + c;
                acc += x_row[iw] * w_row[c];
            }
        }
    }

    int64_t out_idx = (((static_cast<int64_t>(b) * out_c + oc) * out_h + oh) * out_w + ow);
    out[out_idx] = acc;
}

torch::Tensor launch_conv2d(torch::Tensor x, torch::Tensor weight) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && weight.dtype() == torch::kFloat32, "Inputs must be float32");
    TORCH_CHECK(x.dim() == 4 && weight.dim() == 4, "Inputs must be 4D");
    TORCH_CHECK(x.size(1) == weight.size(1), "in_channels mismatch");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "Inputs must be contiguous");

    int batch = static_cast<int>(x.size(0));
    int in_c = static_cast<int>(x.size(1));
    int H = static_cast<int>(x.size(2));
    int W = static_cast<int>(x.size(3));

    int out_c = static_cast<int>(weight.size(0));
    int kh = static_cast<int>(weight.size(2));
    int kw = static_cast<int>(weight.size(3));

    TORCH_CHECK(H >= kh && W >= kw, "Input spatial size must be >= kernel size for valid conv");

    int out_h = H - kh + 1;
    int out_w = W - kw + 1;

    auto out = torch::empty({batch, out_c, out_h, out_w}, x.options());
    if (batch == 0 || out_c == 0 || out_h == 0 || out_w == 0) {
        return out;
    }

    dim3 block(16, 16, 1);
    dim3 grid(
        (out_w + block.x - 1) / block.x,
        (out_h + block.y - 1) / block.y,
        batch * out_c
    );

    direct_conv2d_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(),
        batch, in_c, H, W, out_c, kh, kw, out_h, out_w
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

_ext = load_inline(
    name="cuda_conv2d_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["launch_conv2d"],
    verbose=False,
)

def run(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    x:      4D float32 tensor, shape (batch, in_channels, H, W).
    weight: 4D float32 tensor, shape (out_channels, in_channels, kh, kw).
    returns: 4D float32 tensor, shape (batch, out_channels, H-kh+1, W-kw+1) — valid convolution,
             stride 1, no padding, no bias.
    """
    return _ext.launch_conv2d(x, weight)
