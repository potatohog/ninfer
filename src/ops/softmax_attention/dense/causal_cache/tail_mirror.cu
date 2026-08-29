// ninfer::ops - causal precision-tail ring mirror for the prompt route. The fused small-T
// kernels mirror the ring inside their append; the prompt route (body append + body
// attention) instead runs this dedicated pass between the two. It copies the current tail
// rows [mirror_start, window) of each live row into the BF16 ring and advances the
// ring-validity watermark to the ring horizon, so the following decode rounds read the
// tail through the ring exactly as the fused route would have. The prompt route is
// single-row (batch size 1); the table row comes from the KV table row binding.
#include "ops/softmax_attention/dense/causal_cache/launch.h"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"
#include "ops/common/math.h"
#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/softmax_attention.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kMirrorRowsPerCta = 8;

// grid.x covers the worst-case mirror row count (at most the ring rows C); each CTA handles
// kMirrorRowsPerCta rows. Rows outside [mirror_start, window) do no work. Every live row's
// input column is (position - first_pos); the ring destination is formulaic, so the mirror
// needs no tail block table.
template <typename Geometry>
__launch_bounds__(128) __global__ void causal_attention_tail_mirror_kernel(
    const __nv_bfloat16* k, const __nv_bfloat16* v, const std::int32_t* positions,
    const std::int32_t* table_rows, __nv_bfloat16* ring_k, __nv_bfloat16* ring_v,
    std::int32_t* watermark, std::int32_t width, std::int32_t tail_rows,
    std::int32_t ring_pages) {
    constexpr int D         = kCausalHeadDim;
    constexpr int Hkv       = Geometry::KVHeads;
    constexpr int Rows      = kMirrorRowsPerCta;
    constexpr int Threads   = 128;

    const int row_lo    = static_cast<int>(blockIdx.x) * Rows;
    const int tid       = static_cast<int>(threadIdx.x);
    const int table_row = table_rows[0];

    const std::int32_t first_pos = positions[0];
    const std::int32_t last_pos  = positions[width - 1];
    const int window             = last_pos + 1;
    const int body_end           =
        causal_small_t_body_end(window, watermark[table_row], tail_rows);
    const int mirror_start = causal_small_t_mirror_start(window, first_pos, body_end, tail_rows);

    for (int r = 0; r < Rows; ++r) {
        const int p = row_lo + r;
        if (p < mirror_start || p >= window) { continue; }
        const int tok       = p - first_pos;
        const int ring_page = table_row * ring_pages + ((p >> kPagedKVPageShift) % ring_pages);
        for (int idx = tid; idx < 2 * D * Hkv; idx += Threads) {
            const bool is_v = idx >= D * Hkv;
            const int rest  = is_v ? idx - D * Hkv : idx;
            const int h     = rest / D;
            const int d     = rest % D;
            const std::int64_t src =
                d + static_cast<std::int64_t>(D) * (h + static_cast<std::int64_t>(Hkv) * tok);
            // input is [D, Hkv, width], batch 1
            const std::int64_t dst =
                causal_cache_index<Geometry>(ring_page, h, d, p & kPagedKVPageMask);
            if (is_v) { ring_v[dst] = v[src]; } else { ring_k[dst] = k[src]; }
        }
    }
    if (blockIdx.x == 0 && tid == 0) {
        const int horizon = window - causal_small_t_ring_rows(tail_rows);
        if (horizon > watermark[table_row]) {
            watermark[table_row] = horizon;
        }
    }
}

} // namespace

void causal_attention_tail_mirror_launch(const Tensor& k, const Tensor& v,
                                         const Tensor& positions, const Tensor& table_rows,
                                         const CausalTailDescriptor& tail, cudaStream_t stream) {
    const auto width    = static_cast<std::int32_t>(positions.ne[0]);
    const auto ring_rows = static_cast<std::uint32_t>(tail.tail_rows) + 64u;
    const auto ring_pages = static_cast<std::int32_t>(ring_rows / 64u);
    // The mirror never exceeds one ring of rows; the host does not read device positions, so
    // the grid is sized to the ring and idle CTAs exit early.
    const dim3 grid(div_up(static_cast<std::int32_t>(ring_rows), kMirrorRowsPerCta), 1, 1);
    const bool wide_geometry = k.ne[1] == 4;
    if (wide_geometry) {
        causal_attention_tail_mirror_kernel<CausalD256H24Kv4>
            <<<grid, 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(k.data),
                static_cast<const __nv_bfloat16*>(v.data),
                static_cast<const std::int32_t*>(positions.data),
                static_cast<const std::int32_t*>(table_rows.data),
                static_cast<__nv_bfloat16*>(tail.k_ring.data),
                static_cast<__nv_bfloat16*>(tail.v_ring.data), tail.watermark, width,
                tail.tail_rows, ring_pages);
    } else {
        causal_attention_tail_mirror_kernel<CausalD256H16Kv2>            <<<grid, 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(k.data),
                static_cast<const __nv_bfloat16*>(v.data),
                static_cast<const std::int32_t*>(positions.data),
                static_cast<const std::int32_t*>(table_rows.data),
                static_cast<__nv_bfloat16*>(tail.k_ring.data),
                static_cast<__nv_bfloat16*>(tail.v_ring.data), tail.watermark, width,
                tail.tail_rows, ring_pages);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
