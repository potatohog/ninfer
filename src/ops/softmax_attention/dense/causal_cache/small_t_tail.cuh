#pragma once

// ninfer::ops - split-KV causal small-T attention, BF16 precision-tail partial kernel.
// Standalone from the body kernels (small_t_bf16.cuh / small_t_i8.cuh): the tail reads the
// newest rows through the BF16 ring in original (unrotated) coordinates, so the body's per-dtype
// codec and rotation are absent here by construction. The ring is a per-table-row pool of
// C = tail_rows + 64 rows in the same [D, 64, Hkv, pages] plane layout as the body cache; row p
// of table row t sits in physical page t * ring_pages + (p >> 6) % ring_pages at page offset
// p & 63, where ring_pages = C / 64.
//
// The kernel owns the tail key range [body_end, window) with body_end = causal_small_t_body_end
// (page-aligned tail base, ring-validity watermark, ring horizon). Its split partition covers
// that range; the append mirror writes the current tail rows [mirror_start, split_end) into the
// ring and advances the watermark to the ring horizon. Current-step tokens are read directly
// from the input, so no split depends on another split's ring write (the ring rows older than
// the first live position were mirrored by a previous, stream-ordered round).

#include <cuda_bf16.h>
#include <math_constants.h>

#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"

#include <cstdint>
#include <limits>

namespace ninfer::ops {

template <typename PartialAcc, typename Geometry, int TokenTile, int WarpsPerCta,
          bool MultiBatch, bool Masked, typename CacheInput>
__launch_bounds__(128, 2) __global__ void causal_attention_small_t_tc_partial_tail_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos,
    __nv_bfloat16* ring_k, __nv_bfloat16* ring_v, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t tokens, std::int32_t full_width,
    std::int32_t column_begin, std::int32_t logical_capacity, std::int32_t* watermark,
    std::int32_t tail_rows, std::int32_t ring_pages, float scale,
    PartialAcc* partial_acc, float* partial_m, float* partial_l, std::int32_t slot_stride) {
    static_assert(TokenTile >= 1 && TokenTile <= 6);
    static_assert(WarpsPerCta >= 1 && WarpsPerCta <= 4);

    constexpr int Wc      = WarpsPerCta;
    constexpr int Br      = Wc * 16;
    constexpr int Bc      = 32;
    constexpr int D       = kCausalHeadDim;
    constexpr int Threads = Wc * 32;
    constexpr int QKNt    = Bc / 8;
    constexpr int QKKs    = D / 16;
    constexpr int PVNt    = D / 8;
    constexpr int PVKs    = Bc / 16;
    // The tail range is at most C = tail_rows + 64 keys and the split policy keeps at least
    // 4 * SmallTSplitScale splits, so a single split spans far fewer pages than the body's
    // 49-page envelope bound.
    constexpr int PageIds       = 64;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    constexpr int QkvRows       = 2 * Bc;

    static_assert(QkvRows >= Br);

    __shared__ __align__(16) __nv_bfloat16 qkv_s[QkvRows * D];
    __shared__ __align__(16) __nv_bfloat16 p_s[Wc * 16 * Bc];
    __shared__ std::int32_t physical_pages_s[PageIds];
    __nv_bfloat16* k_s = qkv_s;
    __nv_bfloat16* v_s = qkv_s + Bc * D;

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    int valid_tokens      = tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
    }
    const int row_count = tokens * Geometry::GroupSize;

    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(kCausalHeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(kCausalHeadDim) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(kCausalHeadDim) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * kCausalHeadDim * Geometry::QHeads *
                       tokens * slot_stride;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * slot_stride;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * slot_stride;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                    -CUDART_INF_F;
                partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                    0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[causal_partial_acc_index<Geometry>(q_head, d, token, split,
                                                               tokens)] =
                    static_cast<PartialAcc>(0.0f);
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || tokens < 1 || tokens > TokenTile ||
        row_count > Br || split_count <= 0 || ring_pages <= 0) {
        return;
    }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[tokens - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int window = last_pos + 1;
    // The watermark row is the ring prime point for this table row: fresh sequences prime from
    // position 0, restored histories from the restored length, and the append refreshes it to
    // the ring horizon each round.
    const int body_end = causal_small_t_body_end(window, watermark[table_row], tail_rows);
    const int key_begin = (body_end < window) ? body_end : window;
    const int range_len = window - key_begin;
    if (range_len <= 0) {
        write_neutral();
        return;
    }
    const int active_split_count =
        causal_small_t_active_splits<Geometry, false>(range_len, split_count, TokenTile);
    if (split >= active_split_count) { return; }

    const int logical_tiles = div_up(range_len, Bc);
    const bool tile_split   = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(range_len, active_split_count);
    const int split_start = key_begin + split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);
    const int first_page = first_tile >> kPagedKVPageShift;
    const int page_count = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int page = tid; page < page_count; page += Threads) {
        // Formulaic ring addressing: absolute page first_page + page wraps inside this table
        // row's R-page ring, so no tail block table exists.
        physical_pages_s[page] =
            table_row * ring_pages + ((first_page + page) % ring_pages);
    }

    if constexpr (CacheInput::writes_cache) {
        if (ring_k != nullptr) {
            // Mirror this split's tail rows [mirror_start, split_end) into the ring. Everything
            // before mirror_start is already ring-resident (previous round) or starts at the
            // prime point, so the mirror never rewrites it.
            const int mirror_start =
                causal_small_t_mirror_start(window, first_pos, body_end, tail_rows);
            for (int idx = tid; idx < valid_tokens * D; idx += Threads) {
                const int token = idx / D;
                const int d     = idx - token * D;
                const int p     = pos[token];
                if (p < mirror_start || p >= split_end) { continue; }
                const std::int64_t src = kv_cache_int8_new_index<Geometry>(kv_head, d, token);
                const int ring_page =
                    table_row * ring_pages + ((p >> kPagedKVPageShift) % ring_pages);
                const std::int64_t dst = causal_cache_index<Geometry>(
                    ring_page, kv_head, d, p & kPagedKVPageMask);
                ring_k[dst] = input.k[src];
                ring_v[dst] = input.v[src];
            }
            if (split == 0 && tid == 0) {
                const int horizon = window - causal_small_t_ring_rows(tail_rows);
                if (horizon > watermark[table_row]) {
                    watermark[table_row] = horizon;
                }
            }
            __syncthreads();
        }
    }

    for (int idx = tid; idx < Br * D; idx += Threads) {
        const int row = idx / D;
        const int d   = idx - row * D;
        int q_head    = 0;
        int token     = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (row < row_count && causal_valid_q_head<Geometry>(kv_head, q_head)) {
            value = q[causal_q_index<Geometry>(q_head, d, token)];
        }
        qkv_s[row * D + causal_small_t_tc_swz(row, d)] = value;
    }
    __syncthreads();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    const int warp_row0 = warp * 16;
    __nv_bfloat16* p_sw = &p_s[warp * 16 * Bc];

    unsigned af_q[QKKs][4];
#pragma unroll
    for (int k = 0; k < QKKs; ++k) {
        const int arow = warp_row0 + a_rowoff;
        const int acol = k * 16 + a_coloff;
        ldmatrix_x4(af_q[k][0], af_q[k][1], af_q[k][2], af_q[k][3],
                    smem_addr(&qkv_s[arow * D + causal_small_t_tc_swz(arow, acol)]));
    }
    __syncthreads();
    int physical_page = physical_pages_s[0];
    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;
        if (kb != 0 && (k0 & kPagedKVPageMask) == 0) {
            physical_page = physical_pages_s[(k0 >> kPagedKVPageShift) - first_page];
        }
        // Stage the bf16 K/V key tile with one cp.async wave (16B/thread, high MLP).
        // Current-step tokens come from the input; out-of-range slots are zeroed.
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
            const int key_l      = chunk / (D / 8);
            const int d          = (chunk - key_l * (D / 8)) * 8;
            const int key        = k0 + key_l;
            __nv_bfloat16* k_dst = &k_s[key_l * D + causal_small_t_tc_swz(key_l, d)];
            __nv_bfloat16* v_dst = &v_s[key_l * D + causal_small_t_tc_swz(key_l, d)];
            if (key >= split_start && key < split_end) {
                if constexpr (CacheInput::writes_cache) {
                    const int new_token = key - first_pos;
                    const bool from_new =
                        new_token >= 0 && new_token < valid_tokens && key >= first_pos;
                    if (from_new) {
                        const std::int64_t off =
                            kv_cache_int8_new_index<Geometry>(kv_head, d, new_token);
                        ninfer::ops::cp_async<16>(k_dst, &input.k[off]);
                        ninfer::ops::cp_async<16>(v_dst, &input.v[off]);
                    } else {
                        const std::int64_t off = causal_cache_index<Geometry>(
                            physical_page, kv_head, d, key & kPagedKVPageMask);
                        ninfer::ops::cp_async<16>(k_dst, &ring_k[off]);
                        ninfer::ops::cp_async<16>(v_dst, &ring_v[off]);
                    }
                } else {
                    const std::int64_t off = causal_cache_index<Geometry>(
                        physical_page, kv_head, d, key & kPagedKVPageMask);
                    ninfer::ops::cp_async<16>(k_dst, &ring_k[off]);
                    ninfer::ops::cp_async<16>(v_dst, &ring_v[off]);
                }
            } else {
                store_vec(k_dst, make_int4(0, 0, 0, 0));
                store_vec(v_dst, make_int4(0, 0, 0, 0));
            }
        }
        ninfer::ops::cp_commit();
        ninfer::ops::cp_wait<0>();
        __syncthreads();

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
                unsigned bf[2];
                const int brow = nt * 8 + b_rin;
                const int bcol = k * 16 + b_koff;
                ldmatrix_x2(bf[0], bf[1],
                            smem_addr(&k_s[brow * D + causal_small_t_tc_swz(brow, bcol)]));
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af_q[k][0],
                         af_q[k][1], af_q[k][2], af_q[k][3], bf[0], bf[1]);
            }
        }

        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head0, token0);
        causal_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head1, token1);
        const int qabs0 = (row0 < row_count) ? pos[token0] : -1;
        const int qabs1 = (row1 < row_count) ? pos[token1] : -1;

        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0 = nt * 8 + 2 * lid;
            const int col1 = col0 + 1;
            const int key0 = k0 + col0;
            const int key1 = col1 + k0;
            score[nt][0] =
                (row0 < row_count && key0 >= split_start && key0 < split_end && key0 <= qabs0)
                    ? score[nt][0] * scale
                    : -CUDART_INF_F;
            score[nt][1] =
                (row0 < row_count && key1 >= split_start && key1 < split_end && key1 <= qabs0)
                    ? score[nt][1] * scale
                    : -CUDART_INF_F;
            score[nt][2] =
                (row1 < row_count && key0 >= split_start && key0 < split_end && key0 <= qabs1)
                    ? score[nt][2] * scale
                    : -CUDART_INF_F;
            score[nt][3] =
                (row1 < row_count && key1 >= split_start && key1 < split_end && key1 <= qabs1)
                    ? score[nt][3] * scale
                    : -CUDART_INF_F;
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0    = fmaxf(m0, bm0);
        const float nm1    = fmaxf(m1, bm1);
        const float alpha0 = (m0 == -CUDART_INF_F) ? 0.0f : exp2_approx((m0 - nm0) * Log2E);
        const float alpha1 = (m1 == -CUDART_INF_F) ? 0.0f : exp2_approx((m1 - nm1) * Log2E);

        float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0  = nt * 8 + 2 * lid;
            const int col1  = col0 + 1;
            const float p00 = (nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][0] - nm0) * Log2E)
                                  : 0.0f;
            const float p01 = (nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][1] - nm0) * Log2E)
                                  : 0.0f;
            const float p10 = (nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][2] - nm1) * Log2E)
                                  : 0.0f;
            const float p11 = (nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][3] - nm1) * Log2E)
                                  : 0.0f;
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            p_sw[gid * Bc + causal_small_t_tc_swz32(gid, col0)]           = __float2bfloat16(p00);
            p_sw[gid * Bc + causal_small_t_tc_swz32(gid, col1)]           = __float2bfloat16(p01);
            p_sw[(gid + 8) * Bc + causal_small_t_tc_swz32(gid + 8, col0)] = __float2bfloat16(p10);
            p_sw[(gid + 8) * Bc + causal_small_t_tc_swz32(gid + 8, col1)] = __float2bfloat16(p11);
        }
        bl0 = warp_sum<4>(bl0, FullMask);
        bl1 = warp_sum<4>(bl1, FullMask);

        l0 = l0 * alpha0 + bl0;
        l1 = l1 * alpha1 + bl1;
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }
        __syncwarp();

#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned pf[4];
                const int pcol = k * 16 + a_coloff;
                ldmatrix_x4(
                    pf[0], pf[1], pf[2], pf[3],
                    smem_addr(
                        &p_sw[a_rowoff * Bc + causal_small_t_tc_swz32(a_rowoff, pcol)]));
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_s[vrow * D + causal_small_t_tc_swz(vrow, vcol)]));
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                         vf[0], vf[1]);
            }
        }
        __syncthreads();
    }

    if (lid == 0) {
        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        if (row0 < row_count) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head, token);
            partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m0;
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l0;
        }
        if (row1 < row_count) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head, token);
            partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m1;
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l1;
        }
    }

    // MMA fragments hold each row in four-lane groups. For a BF16 partial accumulator the
    // split-local accumulator is staged through shared memory and written as contiguous
    // d-vector stores; for an FP32 partial accumulator the FP32 fragments are stored directly
    // (no BF16 rounding), matching the body kernel's per-profile accumulator dtype.
    if constexpr (std::is_same_v<PartialAcc, __nv_bfloat16>) {
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            const int d0   = n * 8 + 2 * lid;
            const int d1   = d0 + 1;
            const int row0 = warp_row0 + gid;
            const int row1 = row0 + 8;
            if (row0 < row_count) {
                qkv_s[row0 * D + d0] = __float2bfloat16(acc[n][0]);
                qkv_s[row0 * D + d1] = __float2bfloat16(acc[n][1]);
            }
            if (row1 < row_count) {
                qkv_s[row1 * D + d0] = __float2bfloat16(acc[n][2]);
                qkv_s[row1 * D + d1] = __float2bfloat16(acc[n][3]);
            }
        }
        __syncthreads();

        for (int chunk = tid; chunk < row_count * (D / 8); chunk += Threads) {
            const int row = chunk / (D / 8);
            const int d   = (chunk - row * (D / 8)) * 8;
            int q_head    = 0;
            int token     = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                const std::int64_t dst =
                    causal_partial_acc_index<Geometry>(q_head, d, token, split, tokens);
                store_vec(&partial_acc[dst], load_vec<int4>(&qkv_s[row * D + d]));
            }
        }
    } else {
        static_assert(std::is_same_v<PartialAcc, float>);
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            const int d0   = n * 8 + 2 * lid;
            const int d1   = d0 + 1;
            const int row0 = warp_row0 + gid;
            const int row1 = row0 + 8;
            if (row0 < row_count) {
                int q_head = 0, token = 0;
                causal_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head, token);
                if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                    partial_acc[causal_partial_acc_index<Geometry>(q_head, d0, token, split,
                                                                   tokens)] = acc[n][0];
                    partial_acc[causal_partial_acc_index<Geometry>(q_head, d1, token, split,
                                                                   tokens)] = acc[n][1];
                }
            }
            if (row1 < row_count) {
                int q_head = 0, token = 0;
                causal_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head, token);
                if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                    partial_acc[causal_partial_acc_index<Geometry>(q_head, d0, token, split,
                                                                   tokens)] = acc[n][2];
                    partial_acc[causal_partial_acc_index<Geometry>(q_head, d1, token, split,
                                                                   tokens)] = acc[n][3];
                }
            }
        }
    }
}

} // namespace ninfer::ops
