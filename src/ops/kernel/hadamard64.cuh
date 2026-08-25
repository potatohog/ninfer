#pragma once

// ninfer::ops - fixed per-64 Sylvester Walsh-Hadamard rotation shared by the INT8 GQA
// route. The INT8-G64 KV codec and the native Q8-G64 profile operate in the Hadamard
// domain: each 64-element K/V group is rotated before INT8-G64 encoding, each query row
// is rotated before Q8-G64 encoding, and the attention output is un-rotated after the
// value reduction. The rotation is a property of the INT8 cache dtype; the BF16 route
// carries no rotation.
//
// The transform applied to one 64-group is y = H64 x / 8, where H64 is the 64x64
// Sylvester Hadamard matrix built by the iterative butterfly (stages h = 1, 2, 4, 8,
// 16, 32; pair (j, j+h) with j&h == 0: (a, b) -> (a + b, a - b)). H64 is orthonormal
// and self-inverse (H64 H64 = 64 I), so the same transform undoes the rotation and the
// transform preserves Euclidean dot products.
//
// Device layout: one warp owns one 64-group; lane i holds the group values at local
// indices i and i+32. Stages h = 1..16 mix lanes with shuffles; the final h = 32 stage
// mixes the two in-lane registers. All 32 lanes must participate (full-warp mask).

#include <cuda_runtime.h>

namespace ninfer::ops {

// Apply y = H64 x / 8 in place to the 64-group whose local-index i and i+32 values
// are [lo, hi] on lane i. The evaluation order (one FP32 add/sub per stage per pair,
// stages ascending) is part of the INT8 codec's FP32 semantics and is mirrored
// bit-for-bit by the host codec oracle.
__device__ __forceinline__ void hadamard64_warp_pair(float& lo, float& hi, int lane,
                                                     unsigned mask) {
#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        const float lo2 = __shfl_xor_sync(mask, lo, h);
        const float hi2 = __shfl_xor_sync(mask, hi, h);
        if ((lane & h) == 0) {
            lo += lo2;
            hi += hi2;
        } else {
            lo = lo2 - lo;
            hi = hi2 - hi;
        }
    }
    const float s = lo + hi;
    const float d = lo - hi;
    lo            = s * 0.125f;
    hi            = d * 0.125f;
}

} // namespace ninfer::ops