#include <ninfer/targets/qwen3_6/decoder_state.h>

#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6 {
namespace {

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

PagedKVCacheLayout plan_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                              std::int32_t kv_heads, std::int32_t head_dim, DType dtype,
                              std::int32_t quant_group, std::int32_t table_rows,
                              std::uint32_t physical_page_groups) {
    if (layers == 0 ||
        layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        kv_heads <= 0 || head_dim <= 0 || table_rows <= 0) {
        throw std::invalid_argument("Paged KV cache geometry is invalid");
    }
    const bool scaled = dtype == DType::I8 || dtype == DType::FP8_E4M3FN;
    const bool valid_profile =
        (dtype == DType::BF16 && quant_group == 0) ||
        (dtype == DType::I8 && quant_group == kKvInt8QuantGroup && head_dim % quant_group == 0) ||
        (dtype == DType::FP8_E4M3FN && head_dim == kKvFp8QuantGroup &&
         quant_group == kKvFp8QuantGroup);
    if (!valid_profile) {
        throw std::invalid_argument("Paged KV cache dtype or quantization is invalid");
    }

    const std::uint32_t logical_pages = page_count(capacity);
    if (physical_page_groups < logical_pages) {
        throw std::invalid_argument("Paged KV physical pages are below logical capacity");
    }

    KVPageGeometry geometry;
    geometry.planes.reserve(static_cast<std::size_t>(layers) * (scaled ? 4ULL : 2ULL));
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        geometry.planes.push_back({dtype, head_dim, kv_heads, 256});
        geometry.planes.push_back({dtype, head_dim, kv_heads, 256});
        if (scaled) {
            geometry.planes.push_back({DType::FP16, head_dim / quant_group, kv_heads, 256});
            geometry.planes.push_back({DType::FP16, head_dim / quant_group, kv_heads, 256});
        }
    }
    return PagedKVCacheLayout{
        .pages = plan_device_kv_page_pool(
            builder, DeviceKVPagePoolSpec{.page_group_count = physical_page_groups,
                                          .geometry         = std::move(geometry)}),
        .execution_tables = plan_kv_execution_tables(
            builder,
            KVExecutionTableSpec{.logical_page_capacity = logical_pages, .table_rows = table_rows}),
        .layers      = layers,
        .max_context = capacity,
        .kv_heads    = kv_heads,
        .head_dim    = head_dim,
        .dtype       = dtype,
        .quant_group = quant_group,
    };
}

} // namespace

std::size_t KvTailRingLayout::payload_bytes() const noexcept {
    std::size_t bytes = 0;
    for (const TensorRegion& plane : k_planes) { bytes += plane.region.bytes; }
    for (const TensorRegion& plane : v_planes) { bytes += plane.region.bytes; }
    return bytes + watermark.region.bytes;
}

std::optional<KvTailRingLayout> plan_kv_tail_ring(LayoutBuilder& builder,
                                                  const DecoderStateSpec& spec) {
    if (spec.kv_tail_tokens == 0) { return std::nullopt; }
    if (spec.kv_dtype == DType::FP8_E4M3FN) {
        throw std::invalid_argument("KV precision tail requires BF16 or INT8 KV storage");
    }
    const std::uint32_t tail_rows = spec.kv_tail_tokens;
    if (tail_rows < 64 || tail_rows > 16384 || tail_rows % 64 != 0 || spec.full_attention_layers == 0 ||
        spec.kv_heads <= 0 || spec.attention_head_dim != 256 || spec.kv_table_rows <= 0) {
        throw std::invalid_argument("KV precision tail geometry is invalid");
    }
    const std::uint32_t ring_pages = (tail_rows + 64u) / 64u;
    const std::uint32_t ring_rows  = spec.kv_table_rows * static_cast<std::uint32_t>(ring_pages);
    if (ring_rows == 0 || ring_rows > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("KV precision tail ring pages exceed int32");
    }
    KvTailRingLayout layout;
    layout.tail_rows  = static_cast<std::int32_t>(tail_rows);
    layout.ring_pages = static_cast<std::int32_t>(ring_pages);
    layout.k_planes.reserve(spec.full_attention_layers);
    layout.v_planes.reserve(spec.full_attention_layers);
    for (std::uint32_t layer = 0; layer < spec.full_attention_layers; ++layer) {
        layout.k_planes.push_back(builder.add_tensor(
            DType::BF16, {spec.attention_head_dim, 64, spec.kv_heads, static_cast<std::int32_t>(ring_rows)},
            256, "kv tail ring k"));
        layout.v_planes.push_back(builder.add_tensor(
            DType::BF16, {spec.attention_head_dim, 64, spec.kv_heads, static_cast<std::int32_t>(ring_rows)},
            256, "kv tail ring v"));
    }
    layout.watermark =
        builder.add_tensor(DType::I32, {spec.kv_table_rows, 1, 1, 1}, 256, "kv tail watermark");
    return layout;
}

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    DecoderStateLayout layout;
    layout.text_kv = plan_cache(builder, spec.full_attention_layers, spec.capacity, spec.kv_heads,
                                spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                spec.kv_table_rows, spec.text_physical_page_groups);
    layout.text_kv.kv_tail = plan_kv_tail_ring(builder, spec);
    if (spec.enable_mtp) {
        layout.mtp_kv = plan_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                                   spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                   spec.kv_table_rows, spec.mtp_physical_page_groups);
    }
    return layout;
}

PagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)
    : pages_(backing, layout.pages), execution_tables_(backing, layout.execution_tables, pages_),
      layers_(layout.layers), max_context_(layout.max_context), kv_heads_(layout.kv_heads),
      head_dim_(layout.head_dim), dtype_(layout.dtype), quant_group_(layout.quant_group) {
    if (layout.kv_tail) {
        const auto& tail = *layout.kv_tail;
        tail_k_.reserve(tail.k_planes.size());
        tail_v_.reserve(tail.v_planes.size());
        for (const TensorRegion& plane : tail.k_planes) { tail_k_.push_back(plane.bind(backing)); }
        for (const TensorRegion& plane : tail.v_planes) { tail_v_.push_back(plane.bind(backing)); }
        tail_watermark_ = tail.watermark.bind(backing);
        tail_rows_      = tail.tail_rows;
    }
}

PagedKVTailView PagedKVCache::tail_layer_view(std::uint32_t layer) const {
    if (layer >= tail_k_.size() || layer >= layers_) {
        throw std::out_of_range("KV precision tail layer is out of range");
    }
    return PagedKVTailView{
        .k         = tail_k_[layer],
        .v         = tail_v_[layer],
        .watermark = static_cast<std::int32_t*>(tail_watermark_.data),
        .tail_rows = tail_rows_,
    };
}

Tensor PagedKVCache::tail_watermark() const {
    if (tail_watermark_.data == nullptr) { throw std::logic_error("KV precision tail is absent"); }
    return tail_watermark_;
}

PagedKVCacheView::PagedKVCacheView(const PagedKVCache& cache, Tensor block_table) noexcept
    : cache_(&cache), block_table_(block_table) {}

std::uint32_t PagedKVCacheView::max_context() const noexcept {
    return cache_ == nullptr ? 0 : cache_->max_context();
}

PagedKVLayerView PagedKVCacheView::layer_view(std::uint32_t layer) const {
    if (cache_ == nullptr) { throw std::logic_error("Paged KV execution view is empty"); }
    return cache_->layer_view(layer, block_table_);
}

PagedKVCacheView PagedKVCache::execution_view(const KVExecutionRowLease& row) const {
    if (!row.belongs_to(execution_tables_)) {
        throw std::invalid_argument("Paged KV execution row belongs to another cache");
    }
    return PagedKVCacheView(*this, execution_tables_.row(row.handle()));
}

PagedKVLayerView PagedKVCache::layer_view(std::uint32_t layer, Tensor block_table) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const bool scaled        = dtype_ == DType::I8 || dtype_ == DType::FP8_E4M3FN;
    const std::size_t stride = scaled ? 4ULL : 2ULL;
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    return PagedKVLayerView{
        .k_pages       = pages_.plane(base),
        .v_pages       = pages_.plane(base + 1),
        .k_scale_pages = scaled ? pages_.plane(base + 2) : Tensor(),
        .v_scale_pages = scaled ? pages_.plane(base + 3) : Tensor(),
        .block_table   = block_table,
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .dtype         = dtype_,
        .quant_group   = quant_group_,
    };
}

PagedKVBatchLayerView PagedKVCache::batch_layer_view(std::uint32_t layer) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const bool scaled        = dtype_ == DType::I8 || dtype_ == DType::FP8_E4M3FN;
    const std::size_t stride = scaled ? 4ULL : 2ULL;
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    return PagedKVBatchLayerView{
        .k_pages       = pages_.plane(base),
        .v_pages       = pages_.plane(base + 1),
        .k_scale_pages = scaled ? pages_.plane(base + 2) : Tensor(),
        .v_scale_pages = scaled ? pages_.plane(base + 3) : Tensor(),
        .block_tables  = execution_tables_.matrix(),
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .dtype         = dtype_,
        .quant_group   = quant_group_,
    };
}

DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : text_kv(backing, layout.text_kv) {
    if (layout.mtp_kv) { mtp_kv.emplace(backing, *layout.mtp_kv); }
}

PagedKVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const PagedKVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

} // namespace ninfer::targets::qwen3_6
