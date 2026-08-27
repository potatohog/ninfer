#pragma once

// Exact grouped-query head geometries served by the Qwen3.6 GQA kernels. Head
// dimension, cache format, and tile policy are shared; head mapping remains a
// compile-time property so each registered shape gets an independent kernel.

namespace ninfer::ops {

template <int QHeadsValue, int KVHeadsValue, int DecodeSplitScaleValue,
          int KvScaleSlotsValue = 4>
struct GqaGeometry {
    static_assert(QHeadsValue > 0 && KVHeadsValue > 0);
    static_assert(QHeadsValue % KVHeadsValue == 0);
    static_assert(DecodeSplitScaleValue > 0);

    static constexpr int QHeads           = QHeadsValue;
    static constexpr int KVHeads          = KVHeadsValue;
    static constexpr int GroupSize        = QHeads / KVHeads;
    static constexpr int DecodeSplitScale = DecodeSplitScaleValue;
    static constexpr int DecodeSplits     = 85 * DecodeSplitScale;
    // Number of FP16 scale slots per KV head (the scale-plane leading extent). 4 is the
    // per-64 INT8-G64 default (one scale per rotated 64-group); 8 is the per-32
    // INT8-G64/S32 codec (one scale per 32-half of each 64-group). This is a compile-time
    // per-target codec choice: the 27B route enables per-32, the 35B route stays per-64.
    static constexpr int KvScaleSlots     = KvScaleSlotsValue;
};

using Gqa27Geometry = GqaGeometry<24, 4, 1, 8>;
using Gqa35Geometry = GqaGeometry<16, 2, 2>;

} // namespace ninfer::ops
