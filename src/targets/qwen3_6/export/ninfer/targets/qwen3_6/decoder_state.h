#pragma once

#include "core/layout.h"
#include "core/paged_kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer::targets::qwen3_6 {

inline constexpr std::int32_t kKvInt8QuantGroup = 64;
inline constexpr std::int32_t kKvFp8QuantGroup  = 256;

struct DecoderStateSpec {
    std::uint32_t full_attention_layers     = 0;
    std::uint32_t mtp_layers                = 0;
    std::uint32_t capacity                  = 0;
    std::int32_t kv_heads                   = 0;
    std::int32_t attention_head_dim         = 0;
    DType kv_dtype                          = DType::BF16;
    std::int32_t kv_quant_group             = 0;
    // KV precision tail: exact BF16 ring of the newest kv_tail_tokens rows (multiple of 64) of
    // every full-attention layer, mirrored by the text attention route. 0 disables the ring.
    std::uint32_t kv_tail_tokens            = 0;
    bool enable_mtp                         = false;
    std::int32_t kv_table_rows              = 1;
    std::uint32_t text_physical_page_groups = 0;
    std::uint32_t mtp_physical_page_groups  = 0;
};

// KV precision-tail ring: per-table-row BF16 rings of the newest rows, one plane per
// full-attention layer and per K/V, plus the shared per-table-row ring-validity watermark.
// Row p of table row t sits in physical ring page t * ring_pages + (p / 64) % ring_pages at
// page offset p % 64, where ring_pages = (tail_rows + 64) / 64; every plane has the same
// [head_dim, 64, kv_heads, table_rows * ring_pages] layout as the body cache pages.
struct KvTailRingLayout {
    std::vector<TensorRegion> k_planes; // [full_attention_layers]
    std::vector<TensorRegion> v_planes; // [full_attention_layers]
    TensorRegion watermark;             // I32 [table_rows]
    std::int32_t tail_rows              = 0;
    std::int32_t ring_pages             = 0;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

// Bound per-layer view of the precision-tail ring, ready for the causal attention Op.
struct PagedKVTailView {
    Tensor k;
    Tensor v;
    std::int32_t* watermark = nullptr; // ring-validity watermark; mutated by the Op
    std::int32_t tail_rows        = 0;
};

struct PagedKVCacheLayout {
    DeviceKVPagePoolLayout pages;
    KVExecutionTableLayout execution_tables;
    std::uint32_t layers      = 0;
    std::uint32_t max_context = 0;
    std::int32_t kv_heads     = 0;
    std::int32_t head_dim     = 0;
    DType dtype               = DType::BF16;
    std::int32_t quant_group  = 0;
    std::optional<KvTailRingLayout> kv_tail;

    [[nodiscard]] std::size_t payload_bytes() const noexcept {
        return pages.payload_bytes() + (kv_tail ? kv_tail->payload_bytes() : 0);
    }
};

class PagedKVCache;

class PagedKVCacheView {
public:
    PagedKVCacheView() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return cache_ != nullptr; }

    [[nodiscard]] std::uint32_t max_context() const noexcept;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer) const;

private:
    friend class PagedKVCache;
    PagedKVCacheView(const PagedKVCache& cache, Tensor block_table) noexcept;

    const PagedKVCache* cache_ = nullptr;
    Tensor block_table_;
};

class PagedKVCache {
public:
    PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout);

    PagedKVCache(const PagedKVCache&)            = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;
    PagedKVCache(PagedKVCache&&)                 = delete;
    PagedKVCache& operator=(PagedKVCache&&)      = delete;

    [[nodiscard]] std::uint32_t max_context() const noexcept { return max_context_; }

    [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }

    [[nodiscard]] DeviceKVPagePool& page_pool() noexcept { return pages_; }

    [[nodiscard]] const DeviceKVPagePool& page_pool() const noexcept { return pages_; }

    [[nodiscard]] KVExecutionTablePool& execution_tables() noexcept { return execution_tables_; }

    [[nodiscard]] const KVExecutionTablePool& execution_tables() const noexcept {
        return execution_tables_;
    }

    [[nodiscard]] PagedKVCacheView execution_view(const KVExecutionRowLease& row) const;

    [[nodiscard]] PagedKVBatchLayerView batch_layer_view(std::uint32_t layer) const;

    [[nodiscard]] bool has_tail() const noexcept { return !tail_k_.empty(); }

    [[nodiscard]] std::int32_t tail_rows() const noexcept {
        return tail_watermark_.data != nullptr ? tail_rows_ : 0;
    }

    [[nodiscard]] PagedKVTailView tail_layer_view(std::uint32_t layer) const;

    [[nodiscard]] Tensor tail_watermark() const;

private:
    friend class PagedKVCacheView;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer, Tensor block_table) const;

    DeviceKVPagePool pages_;
    KVExecutionTablePool execution_tables_;
    std::uint32_t layers_      = 0;
    std::uint32_t max_context_ = 0;
    std::int32_t kv_heads_     = 0;
    std::int32_t head_dim_     = 0;
    DType dtype_               = DType::BF16;
    std::int32_t quant_group_  = 0;
    std::vector<Tensor> tail_k_;
    std::vector<Tensor> tail_v_;
    Tensor tail_watermark_;
    std::int32_t tail_rows_ = 0;
};

struct DecoderStateLayout {
    PagedKVCacheLayout text_kv;
    std::optional<PagedKVCacheLayout> mtp_kv;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept {
        return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0);
    }
};

[[nodiscard]] DecoderStateLayout plan_decoder_state(LayoutBuilder& builder,
                                                    const DecoderStateSpec& spec);

struct DecoderState {
    PagedKVCache text_kv;
    std::optional<PagedKVCache> mtp_kv;

    DecoderState(DeviceSpan backing, const DecoderStateLayout& layout);

    [[nodiscard]] PagedKVCache* mtp_cache() noexcept;
    [[nodiscard]] const PagedKVCache* mtp_cache() const noexcept;
};

} // namespace ninfer::targets::qwen3_6
