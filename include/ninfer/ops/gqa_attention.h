#pragma once

#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr std::uint32_t kGqaAttentionMaximumVisibleKeys = 262144;

struct GqaExecutionEnvelope {
    std::uint32_t min_visible_keys = 0;
    std::uint32_t max_visible_keys = 0;
};

/**
 * Shared numerical contract for A1/A2/A3.
 *
 * Public q/k/v inputs and BF16 cache values are interpreted after their BF16 storage boundary.
 * The INT8 route additionally operates in a fixed Hadamard rotation domain. For one contiguous
 * 64-element group, let H64 be the 64x64 Sylvester Walsh-Hadamard matrix and rot(x) = H64 x / 8,
 * evaluated by the iterative butterfly (stages h = 1, 2, 4, 8, 16, 32; pair (j, j+h) with
 * j&h == 0: (a, b) -> (a + b, a - b)) in FP32. H64 is orthonormal and self-inverse
 * (H64 H64 = 64 I), so rot is its own inverse and preserves Euclidean dot products. Every K/V
 * group is rotated before INT8-G64 encoding, every query row is rotated before Q8-G64 encoding,
 * and the attention output is un-rotated after the value reduction. The rotation is a property
 * of the INT8 cache dtype: BF16 cache rows, BF16 Q, and the BF16 compute path carry no
 * rotation, and an INT8 cache row written by A1 or A2 is only readable through this Op's INT8
 * route.
 *
 * INT8-G64 cache rows use one FP16 scale for each contiguous 64-element group of rotated source
 * values r = rot(x). For BF16 source values x, their exact observable encoding is:
 *
 *   a          = max_i abs(r[i])
 *   scale_bits = FP16_RNE(a / 127)
 *   s          = FP32(scale_bits)
 *   inv        = s == 0 ? 0 : FP32(1 / s)
 *   code[i]    = s == 0 ? 0 : I8(clamp(RNE_even(FP32(r[i]) * inv), -127, 127))
 *   decode[i]  = FP32(code[i]) * s        (rotated domain)
 *
 * The scale granularity is a property of the registered target, and the scale plane's per-head
 * extent reflects it. The INT8-G64/S32 variant (Qwen 3.6/3.8 27B) keeps one FP16 scale per
 * contiguous 32-element half of a rotated 64-group: the group is the same WHT rot, and the first
 * half (r[0:32]) and second half (r[32:64]) each carry their own (a, scale_bits, s, inv) with a
 * taken over that half. Its scale plane therefore holds head_dim/32 FP16 slots per head (8 for
 * head_dim 256) rather than head_dim/64 (4). Targets that keep one scale per 64-group use the
 * per-64 rule above unchanged. The Q8-G64 query rule is per-64 in every registered target.
 *
 * A1 and A2 produce identical code and scale bits. The common ideal attention oracle uses BF16 Q
 * and logical cache values (BF16 values for a BF16 cache, rotated-domain FP32 decode above for
 * INT8-G64), then evaluates score dot products, stable softmax, and value reduction in FP64. For
 * an INT8 cache the oracle rotates BF16 Q and the logical K/V groups with the same rot and
 * un-rotates the reduced value before comparison, so every dot product runs over
 * rotation-equivalent operands. The BF16 Op output is promoted to FP64 for comparison with that
 * result.
 *
 * The registered INT8 implementation defines Q8-G64, paired with INT8-G64 K, as its native query
 * compute profile: it encodes the rotated query with the same per-64-group rule and reads the
 * rotated-domain codes. Its profile-defined query quantization and any narrower staging do not
 * replace BF16 Q in the ideal oracle. BF16-cache and INT8-cache compute profiles therefore have
 * separate named numerical criteria owned by the GQA conformance test. Those envelopes apply to
 * the registered geometries, tested token extents, conformance matrix, and target-representative
 * activation range; they are not a universal error bound for arbitrary adversarial BF16 tensors.
 * A1 and A3 are each qualified directly against the ideal oracle. A1-versus-A3 parity is only an
 * additional consistency check.
 */

/**
 * Returns the transient arena capacity required for every W in the inclusive interval at one
 * exact logical batch size. Head geometry, cache dtype, and execution envelope are the fixed
 * implementation profile. Invalid profiles or intervals throw; a legal B=1 prompt route may
 * return zero.
 */
[[nodiscard]] std::size_t
gqa_attention_workspace_capacity_bytes(std::int32_t q_heads, DType cache_dtype,
                                       GqaExecutionEnvelope envelope, std::int32_t batch_size,
                                       std::int32_t min_width, std::int32_t max_width);

/**
 * A1: append K/V for B independent sequences and compute causal grouped-query attention. Let
 * Vb=W when valid_columns is empty and Vb=valid_columns[b] otherwise. For row b, query head h,
 * kvh=floor(h/group), 0<=j<Vb, p=positions[j,b], and that row's populated cache history [0,p]:
 *
 *   score[x]      = scale * dot(q[:,h,j,b], K_cache[b][:,x,kvh]), 0 <= x <= p
 *   probability   = softmax_x(score)
 *   ideal[:,h,j,b] = sum_x probability[x] * V_cache[b][:,x,kvh].
 *
 * The registered geometries are `[256,24|4,W,B]` group 6 and `[256,16|2,W,B]` group 8.
 * q/k/v/out are contiguous BF16 in request-major order, positions is contiguous I32 [W,B], and
 * kv_table_rows is contiguous I32 [B]. valid_columns is either contiguous I32 [B], or an empty
 * Tensor meaning every row has exactly W valid columns. This dense/masked choice is part of the
 * call topology; it is not inferred by copying device metadata to the host. B=1 accepts every
 * positive W in the current prefill/decode domain; B=2..8 accepts W=1..16. Cache storage is BF16
 * or INT8-G64 under the shared numerical contract above. PagedKVBatchLayerView supplies shared
 * planes and the complete block-table matrix; kv_table_rows[b] selects one row for sequence b.
 *
 * In masked form, every row's valid columns are the prefix [0,valid_columns[b]); positions in that
 * prefix are sequential and address populated causal histories. Each nonempty row repeats its
 * final valid position through the invalid tail; an empty row uses zero positions. Other
 * invalid-tail inputs contain safe dummy values. A1 does not modify cache for invalid columns and
 * writes exact BF16 zero to their output. The caller guarantees that the maximum final valid
 * position plus one over nonempty rows lies in the declared execution envelope. The envelope is a
 * host launch-resource promise over that batch maximum; it does not alter any row's causal mask.
 *
 * q/k/v/positions/valid_columns/kv_table_rows/out, every cache plane/table, and live workspace
 * suballocations are pairwise non-overlapping. The Op overwrites every addressed cache row but
 * owns no persistent frontier, allocation, request identity, or commit authority.
 */
void gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   const Tensor& valid_columns, const Tensor& kv_table_rows, float scale,
                   PagedKVBatchLayerView cache, GqaExecutionEnvelope envelope,
                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

/**
 * A2: perform only the cache-write part of A1. k/v are contiguous BF16 `[256,4|2,T]`, positions is
 * contiguous sequential I32 [T], and every addressed code and INT8 scale is overwritten. It reads
 * no unrelated cache row, receives no execution envelope, and owns no persistent frontier.
 */
void gqa_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                   PagedKVLayerView cache, cudaStream_t stream);

/**
 * A3: compute causal attention from an already populated cache without accepting new K/V or
 * mutating any cache plane. q/out are contiguous BF16 `[256,24|16,T]`, positions is contiguous
 * sequential I32 [T], and the mathematical formula and execution-envelope contract are identical
 * to A1. Caller workspace is reported by gqa_attention_workspace_capacity_bytes().
 */
void gqa_attention_cached(const Tensor& q, const Tensor& positions, float scale,
                          const PagedKVLayerView& cache, GqaExecutionEnvelope envelope,
                          WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
