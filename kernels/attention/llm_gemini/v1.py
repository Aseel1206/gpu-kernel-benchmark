# Model: Gemini
# Kernel: 5. Scaled Dot-Product Attention (flash-attention-lite tiled online-softmax CUDA kernel)
import torch
from torch.utils.cpp_extension import load_inline

cpp_source = """
#include <torch/extension.h>
torch::Tensor launch_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

template <int MAX_D>
__global__ void flash_attn_lite_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq_len,
    int head_dim,
    float scale
) {
    constexpr int Br = 32;
    constexpr int Bc = 32;

    int bh = blockIdx.x;
    int q_tile = blockIdx.y;
    int tid = threadIdx.x;

    int q_row = q_tile * Br + tid;
    bool valid_q = (q_row < seq_len);

    int64_t bh_offset = static_cast<int64_t>(bh) * seq_len * head_dim;
    const float* Q_bh = Q + bh_offset;
    const float* K_bh = K + bh_offset;
    const float* V_bh = V + bh_offset;
    float* O_bh = O + bh_offset;

    extern __shared__ float smem[];
    float* K_s = smem;
    float* V_s = K_s + Bc * head_dim;

    float q_reg[MAX_D];
    float o_reg[MAX_D];
    #pragma unroll
    for (int d = 0; d < MAX_D; ++d) {
        q_reg[d] = 0.0f;
        o_reg[d] = 0.0f;
    }

    if (valid_q) {
        for (int d = 0; d < head_dim; ++d) {
            q_reg[d] = Q_bh[static_cast<int64_t>(q_row) * head_dim + d];
        }
    }

    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    int num_k_tiles = (seq_len + Bc - 1) / Bc;
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int k_start = kt * Bc;
        int tile_k_len = min(Bc, seq_len - k_start);

        int total_elems = Bc * head_dim;
        for (int idx = tid; idx < total_elems; idx += Br) {
            int r = idx / head_dim;
            int c = idx % head_dim;
            if (r < tile_k_len) {
                int64_t g_idx = static_cast<int64_t>(k_start + r) * head_dim + c;
                K_s[r * head_dim + c] = K_bh[g_idx];
                V_s[r * head_dim + c] = V_bh[g_idx];
            } else {
                K_s[r * head_dim + c] = 0.0f;
                V_s[r * head_dim + c] = 0.0f;
            }
        }
        __syncthreads();

        if (valid_q) {
            float s_row[Bc];
            float m_ij = -FLT_MAX;
            for (int j = 0; j < Bc; ++j) {
                if (j < tile_k_len) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += q_reg[d] * K_s[j * head_dim + d];
                    }
                    float val = dot * scale;
                    s_row[j] = val;
                    m_ij = fmaxf(m_ij, val);
                } else {
                    s_row[j] = -FLT_MAX;
                }
            }

            float m_new = fmaxf(m_i, m_ij);
            float exp_scale_old = (l_i > 0.0f) ? expf(m_i - m_new) : 0.0f;

            float p_sum = 0.0f;
            float p_row[Bc];
            for (int j = 0; j < Bc; ++j) {
                if (j < tile_k_len) {
                    float p = expf(s_row[j] - m_new);
                    p_row[j] = p;
                    p_sum += p;
                } else {
                    p_row[j] = 0.0f;
                }
            }

            float l_new = exp_scale_old * l_i + p_sum;

            for (int d = 0; d < head_dim; ++d) {
                float pv = 0.0f;
                for (int j = 0; j < tile_k_len; ++j) {
                    pv += p_row[j] * V_s[j * head_dim + d];
                }
                o_reg[d] = o_reg[d] * exp_scale_old + pv;
            }

            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();
    }

    if (valid_q) {
        float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            O_bh[static_cast<int64_t>(q_row) * head_dim + d] = o_reg[d] * inv_l;
        }
    }
}

torch::Tensor launch_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(
        q.dtype() == torch::kFloat32 && k.dtype() == torch::kFloat32 && v.dtype() == torch::kFloat32,
        "Inputs must be float32"
    );
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "Inputs must be 4D");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "Q, K, V must have identical shapes");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "Inputs must be contiguous");

    int batch = static_cast<int>(q.size(0));
    int heads = static_cast<int>(q.size(1));
    int seq_len = static_cast<int>(q.size(2));
    int head_dim = static_cast<int>(q.size(3));

    TORCH_CHECK(head_dim <= 128, "head_dim > 128 not supported in this kernel");

    auto out = torch::empty_like(q);
    if (batch == 0 || heads == 0 || seq_len == 0 || head_dim == 0) {
        return out;
    }

    constexpr int Br = 32;
    constexpr int Bc = 32;
    int bh_total = batch * heads;
    int q_tiles = (seq_len + Br - 1) / Br;
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    size_t smem_bytes = static_cast<size_t>(2 * Bc * head_dim) * sizeof(float);
    dim3 grid(bh_total, q_tiles);

    if (head_dim <= 32) {
        flash_attn_lite_kernel<32><<<grid, Br, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<float>(),
            seq_len, head_dim, scale
        );
    } else if (head_dim <= 64) {
        flash_attn_lite_kernel<64><<<grid, Br, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<float>(),
            seq_len, head_dim, scale
        );
    } else {
        flash_attn_lite_kernel<128><<<grid, Br, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<float>(),
            seq_len, head_dim, scale
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

_ext = load_inline(
    name="cuda_attention_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["launch_attention"],
    verbose=False,
)

def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    q, k, v: 4D float32 tensors, shape (batch, heads, seq_len, head_dim). seq_len and head_dim
             vary across test cases (e.g. seq_len in {128, 512, 2048}, head_dim in {32, 64}).
    returns: 4D float32 tensor, shape (batch, heads, seq_len, head_dim), equal to
             softmax(q @ k.transpose(-1, -2) / sqrt(head_dim)) @ v, no mask, no dropout.
    """
    return _ext.launch_attention(q, k, v)
