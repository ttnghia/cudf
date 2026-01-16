/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace cudf {
namespace detail {
namespace {

/**
 * @brief Checks if the given data type is supported by batch_concatenate.
 *
 * Supported types are: fixed-width types (plain types) and struct types.
 * Lists, strings, and dictionaries are NOT supported.
 */
bool is_batch_concat_supported_type(data_type type)
{
  return cudf::is_fixed_width(type) || type.id() == type_id::STRUCT;
}

/**
 * @brief Recursively checks if a column and all its children are supported by batch_concatenate.
 */
bool is_column_batch_concat_supported(column_view const& col)
{
  if (!is_batch_concat_supported_type(col.type())) { return false; }

  if (col.type().id() == type_id::STRUCT) {
    for (size_type i = 0; i < col.num_children(); ++i) {
      if (!is_column_batch_concat_supported(col.child(i))) { return false; }
    }
  }
  return true;
}

/**
 * @brief Checks if all columns are supported by batch_concatenate.
 */
bool all_columns_batch_concat_supported(host_span<column_view const> cols)
{
  return std::all_of(
    cols.begin(), cols.end(), [](auto const& c) { return is_column_batch_concat_supported(c); });
}

/**
 * @brief Consolidated data structure for a single nesting level.
 *
 * Contains all information needed for both data copy and mask concatenation.
 */
struct level_info {
  // Data copy info
  std::vector<void const*> src_data_ptrs;  // Source data pointers (empty for struct levels)
  std::vector<std::size_t> data_sizes;     // Size of each data buffer in bytes

  // Mask concatenation info
  std::vector<bitmask_type const*> mask_ptrs;  // Source mask pointers (can be null)
  std::vector<size_type> mask_offsets;         // Bit offset for each mask
  std::vector<size_type> col_sizes;            // Number of rows in each column

  // Level metadata
  data_type dtype;         // Data type at this level
  size_type total_rows;    // Total number of rows
  size_type num_children;  // Number of children (for struct types)
  bool has_nulls;          // Whether any column has nulls
  bool is_struct;          // Whether this level is a struct type
};

/**
 * @brief Recursively collects all level information in a single pass.
 *
 * This consolidates what was previously done in multiple passes.
 */
void collect_level_info_recursive(host_span<column_view const> cols,
                                  std::vector<level_info>& levels,
                                  rmm::cuda_stream_view stream)
{
  if (cols.empty()) return;

  level_info info;
  auto const& first_col = cols.front();
  info.dtype            = first_col.type();
  info.is_struct        = (info.dtype.id() == type_id::STRUCT);
  info.num_children     = first_col.num_children();
  info.has_nulls        = false;
  info.total_rows       = 0;

  // Compute element size once for fixed-width types
  std::size_t const element_size = info.is_struct ? 0 : cudf::size_of(info.dtype);

  // Reserve space to avoid reallocations
  auto const num_cols = cols.size();
  info.mask_ptrs.reserve(num_cols);
  info.mask_offsets.reserve(num_cols);
  info.col_sizes.reserve(num_cols);
  if (!info.is_struct) {
    info.src_data_ptrs.reserve(num_cols);
    info.data_sizes.reserve(num_cols);
  }

  // Single pass over columns to collect all info
  for (auto const& col : cols) {
    auto const col_size = col.size();
    info.total_rows += col_size;
    info.col_sizes.push_back(col_size);

    // Mask info
    info.mask_ptrs.push_back(col.null_mask());
    info.mask_offsets.push_back(col.offset());
    if (col.has_nulls()) { info.has_nulls = true; }

    // Data info (only for non-struct types)
    if (!info.is_struct) {
      info.src_data_ptrs.push_back(col.head<char>() + (col.offset() * element_size));
      info.data_sizes.push_back(col_size * element_size);
    }
  }

  levels.push_back(std::move(info));

  // Recurse into struct children
  if (first_col.type().id() == type_id::STRUCT) {
    auto const num_children = first_col.num_children();
    std::vector<column_view> child_cols;
    child_cols.reserve(num_cols);

    for (size_type child_idx = 0; child_idx < num_children; ++child_idx) {
      child_cols.clear();
      for (auto const& col : cols) {
        structs_column_view scv(col);
        child_cols.push_back(scv.get_sliced_child(child_idx, stream));
      }
      collect_level_info_recursive(child_cols, levels, stream);
    }
  }
}

/**
 * @brief Kernel to concatenate bitmasks for multiple levels in a single launch.
 *
 * Each warp processes one complete 32-bit word (aligned to warp boundaries within each level).
 * The work is distributed across all levels and words, with each warp handling a complete word
 * to ensure no cross-level corruption.
 *
 * Optimizations:
 * - Uses shared memory to cache level metadata for frequently accessed levels
 * - Uses block-level reduction before atomic updates to reduce contention
 * - Processes words in contiguous chunks for better memory coalescing
 *
 * @param mask_ptrs Source mask pointers for all columns across all levels
 * @param mask_offsets Bit offset for each source mask (from column offset)
 * @param level_column_offsets Prefix sum of columns per level [num_levels + 1]
 * @param level_word_offsets Prefix sum of words per level [num_levels + 1]
 * @param within_level_bit_offsets Prefix sum of bits within each level [total_columns + 1]
 * @param dest_masks Output mask pointers for each level [num_levels]
 * @param level_total_bits Total bits for each level [num_levels]
 * @param num_levels Number of levels to process
 * @param total_words Total number of 32-bit words across all levels
 * @param valid_counts Output valid counts per level [num_levels]
 */
template <size_type block_size>
CUDF_KERNEL void batch_concatenate_masks_kernel(
  bitmask_type const* const* __restrict__ mask_ptrs,
  size_type const* __restrict__ mask_offsets,
  size_type const* __restrict__ level_column_offsets,
  size_type const* __restrict__ level_word_offsets,
  size_type const* __restrict__ within_level_bit_offsets,
  bitmask_type* const* __restrict__ dest_masks,
  size_type const* __restrict__ level_total_bits,
  size_type num_levels,
  size_type total_words,
  size_type* __restrict__ valid_counts)
{
  // Shared memory for per-level valid counts within this block
  // Maximum 32 levels supported in shared memory (covers most practical cases)
  constexpr size_type max_cached_levels = 32;
  __shared__ size_type block_valid_counts[max_cached_levels];

  // Initialize shared memory valid counts to 0
  if (threadIdx.x < max_cached_levels) { block_valid_counts[threadIdx.x] = 0; }
  __syncthreads();

  // Each warp processes one word
  auto const warp_idx  = cudf::detail::grid_1d::global_thread_id<block_size>() / warp_size;
  auto const lane_idx  = threadIdx.x % warp_size;
  auto const num_warps = cudf::detail::grid_1d::grid_stride<block_size>() / warp_size;

  for (size_type word_idx = warp_idx; word_idx < total_words; word_idx += num_warps) {
    // Find which level this word belongs to using binary search
    size_type const level_idx =
      thrust::upper_bound(
        thrust::seq, level_word_offsets, level_word_offsets + num_levels + 1, word_idx) -
      level_word_offsets - 1;

    // Get the word index within this level
    size_type const word_in_level = word_idx - level_word_offsets[level_idx];

    // Calculate the bit index for this thread within the level
    size_type const bit_in_level = word_in_level * warp_size + lane_idx;

    // Check if this bit is within the valid range for this level
    bool bit_is_valid = false;
    if (bit_in_level < level_total_bits[level_idx]) {
      // Find which column within the level this bit belongs to
      size_type const col_start         = level_column_offsets[level_idx];
      size_type const* level_bit_starts = within_level_bit_offsets + col_start;
      size_type const num_cols_in_level = level_column_offsets[level_idx + 1] - col_start;

      size_type const col_in_level =
        thrust::upper_bound(
          thrust::seq, level_bit_starts, level_bit_starts + num_cols_in_level + 1, bit_in_level) -
        level_bit_starts - 1;

      size_type const global_col_idx = col_start + col_in_level;
      size_type const bit_in_col     = bit_in_level - level_bit_starts[col_in_level];

      // Read the validity bit from source
      bitmask_type const* src_mask = mask_ptrs[global_col_idx];
      if (src_mask == nullptr) {
        bit_is_valid = true;  // No mask means all valid
      } else {
        size_type const src_offset  = mask_offsets[global_col_idx];
        size_type const src_bit_idx = src_offset + bit_in_col;
        bit_is_valid = (src_mask[src_bit_idx / warp_size] >> (src_bit_idx % warp_size)) & 1;
      }
    }

    // Collect validity bits from all threads in warp
    bitmask_type const new_word = __ballot_sync(0xFFFF'FFFFu, bit_is_valid);

    // First thread in warp writes the result and accumulates valid count
    if (lane_idx == 0) {
      dest_masks[level_idx][word_in_level] = new_word;
      size_type const word_valid_count     = __popc(new_word);

      // Use shared memory for levels that fit, otherwise atomic directly
      if (level_idx < max_cached_levels) {
        atomicAdd(&block_valid_counts[level_idx], word_valid_count);
      } else {
        atomicAdd(&valid_counts[level_idx], word_valid_count);
      }
    }
  }

  // Sync before reducing shared memory counts to global
  __syncthreads();

  // First warp reduces shared memory counts to global memory
  if (threadIdx.x < num_levels && threadIdx.x < max_cached_levels) {
    atomicAdd(&valid_counts[threadIdx.x], block_valid_counts[threadIdx.x]);
  }
}

/**
 * @brief Performs all batch operations: data copy and mask concatenation.
 *
 * Returns the output data buffers and mask buffers with null counts.
 */
std::pair<std::vector<rmm::device_buffer>, std::vector<std::pair<rmm::device_buffer, size_type>>>
batch_process_levels(std::vector<level_info> const& levels,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
{
  size_type const num_levels = static_cast<size_type>(levels.size());

  // ========== PHASE 1: Allocate output buffers ==========
  std::vector<rmm::device_buffer> data_buffers;
  std::vector<rmm::device_buffer> null_masks;
  data_buffers.reserve(num_levels);
  null_masks.reserve(num_levels);

  for (auto const& level : levels) {
    // Data buffer
    if (level.is_struct) {
      data_buffers.emplace_back(0, stream, mr);
    } else {
      std::size_t const total_bytes =
        std::accumulate(level.data_sizes.begin(), level.data_sizes.end(), std::size_t{0});
      data_buffers.emplace_back(total_bytes, stream, mr);
    }

    // Null mask buffer
    if (level.has_nulls) {
      null_masks.emplace_back(
        cudf::detail::create_null_mask(level.total_rows, mask_state::UNINITIALIZED, stream, mr));
    } else {
      null_masks.emplace_back(0, stream, mr);
    }
  }

  // ========== PHASE 2: Batch data copy ==========
  // Pre-calculate total number of copy operations for efficient allocation
  size_t total_copy_ops = 0;
  for (auto const& level : levels) {
    if (!level.is_struct) { total_copy_ops += level.src_data_ptrs.size(); }
  }

  if (total_copy_ops > 0) {
    // Reserve upfront to avoid reallocations
    std::vector<void const*> all_src_ptrs;
    std::vector<void*> all_dst_ptrs;
    std::vector<std::size_t> all_sizes;
    all_src_ptrs.reserve(total_copy_ops);
    all_dst_ptrs.reserve(total_copy_ops);
    all_sizes.reserve(total_copy_ops);

    for (size_type level_idx = 0; level_idx < num_levels; ++level_idx) {
      auto const& level = levels[level_idx];
      if (level.is_struct) continue;

      char* dst_ptr = static_cast<char*>(data_buffers[level_idx].data());
      for (size_t col = 0; col < level.src_data_ptrs.size(); ++col) {
        all_src_ptrs.push_back(level.src_data_ptrs[col]);
        all_dst_ptrs.push_back(dst_ptr);
        all_sizes.push_back(level.data_sizes[col]);
        dst_ptr += level.data_sizes[col];
      }
    }

    auto d_src_ptrs =
      make_device_uvector_async(all_src_ptrs, stream, cudf::get_current_device_resource_ref());
    auto d_dst_ptrs =
      make_device_uvector_async(all_dst_ptrs, stream, cudf::get_current_device_resource_ref());
    auto d_sizes =
      make_device_uvector_async(all_sizes, stream, cudf::get_current_device_resource_ref());

    batched_memcpy_async(
      d_src_ptrs.data(), d_dst_ptrs.data(), d_sizes.data(), total_copy_ops, stream);
  }

  // ========== PHASE 3: Batch mask concatenation ==========
  // Collect levels with nulls
  std::vector<size_type> levels_with_nulls;
  for (size_type i = 0; i < num_levels; ++i) {
    if (levels[i].has_nulls) { levels_with_nulls.push_back(i); }
  }

  // Initialize null counts to 0
  std::vector<size_type> null_counts(num_levels, 0);

  if (!levels_with_nulls.empty()) {
    auto const num_null_levels = levels_with_nulls.size();

    // Pre-calculate total sizes for efficient allocation
    size_t total_columns = 0;
    for (size_type level_idx : levels_with_nulls) {
      total_columns += levels[level_idx].mask_ptrs.size();
    }

    // Reserve space upfront to avoid reallocations
    std::vector<bitmask_type const*> all_mask_ptrs;
    std::vector<size_type> all_mask_offsets;
    std::vector<size_type> level_column_offsets;
    std::vector<size_type> level_word_offsets;
    std::vector<size_type> within_level_bit_offsets;
    std::vector<bitmask_type*> dest_mask_ptrs;
    std::vector<size_type> level_total_bits;

    all_mask_ptrs.reserve(total_columns);
    all_mask_offsets.reserve(total_columns);
    level_column_offsets.reserve(num_null_levels + 1);
    level_word_offsets.reserve(num_null_levels + 1);
    within_level_bit_offsets.reserve(total_columns + num_null_levels);
    dest_mask_ptrs.reserve(num_null_levels);
    level_total_bits.reserve(num_null_levels);

    level_column_offsets.push_back(0);
    level_word_offsets.push_back(0);

    size_type total_words = 0;

    for (size_type level_idx : levels_with_nulls) {
      auto const& level = levels[level_idx];

      dest_mask_ptrs.push_back(static_cast<bitmask_type*>(null_masks[level_idx].data()));
      level_total_bits.push_back(level.total_rows);

      size_type level_bit_count = 0;
      for (size_t col = 0; col < level.mask_ptrs.size(); ++col) {
        all_mask_ptrs.push_back(level.mask_ptrs[col]);
        all_mask_offsets.push_back(level.mask_offsets[col]);
        within_level_bit_offsets.push_back(level_bit_count);
        level_bit_count += level.col_sizes[col];
      }
      within_level_bit_offsets.push_back(level_bit_count);

      level_column_offsets.push_back(static_cast<size_type>(all_mask_ptrs.size()));

      // Calculate number of words for this level (rounded up to warp size)
      size_type const level_words = cudf::util::div_rounding_up_safe(level.total_rows, warp_size);
      total_words += level_words;
      level_word_offsets.push_back(total_words);
    }

    // Upload to device - use temporary memory resource for intermediate allocations
    auto d_mask_ptrs =
      make_device_uvector_async(all_mask_ptrs, stream, cudf::get_current_device_resource_ref());
    auto d_mask_offsets =
      make_device_uvector_async(all_mask_offsets, stream, cudf::get_current_device_resource_ref());
    auto d_level_column_offsets = make_device_uvector_async(
      level_column_offsets, stream, cudf::get_current_device_resource_ref());
    auto d_level_word_offsets = make_device_uvector_async(
      level_word_offsets, stream, cudf::get_current_device_resource_ref());
    auto d_within_level_bit_offsets = make_device_uvector_async(
      within_level_bit_offsets, stream, cudf::get_current_device_resource_ref());
    auto d_dest_masks =
      make_device_uvector_async(dest_mask_ptrs, stream, cudf::get_current_device_resource_ref());
    auto d_level_total_bits =
      make_device_uvector_async(level_total_bits, stream, cudf::get_current_device_resource_ref());
    auto d_valid_counts = make_zeroed_device_uvector_async<size_type>(
      num_null_levels, stream, cudf::get_current_device_resource_ref());

    // Launch kernel - one warp per word
    constexpr size_type block_size{256};
    size_type const num_threads = total_words * warp_size;
    cudf::detail::grid_1d config(num_threads, block_size);

    batch_concatenate_masks_kernel<block_size>
      <<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
        d_mask_ptrs.data(),
        d_mask_offsets.data(),
        d_level_column_offsets.data(),
        d_level_word_offsets.data(),
        d_within_level_bit_offsets.data(),
        d_dest_masks.data(),
        d_level_total_bits.data(),
        static_cast<size_type>(num_null_levels),
        total_words,
        d_valid_counts.data());

    // Copy valid counts back using pinned memory for efficient transfer
    auto h_valid_counts = make_pinned_vector_async<size_type>(num_null_levels, stream);
    cuda_memcpy_async<size_type>(
      host_span<size_type>{h_valid_counts}, device_span<size_type const>{d_valid_counts}, stream);
    stream.synchronize();

    for (size_t i = 0; i < num_null_levels; ++i) {
      size_type const level_idx   = levels_with_nulls[i];
      size_type const valid_count = h_valid_counts[i];
      null_counts[level_idx]      = levels[level_idx].total_rows - valid_count;
    }
  }

  // Build mask results
  std::vector<std::pair<rmm::device_buffer, size_type>> mask_results;
  mask_results.reserve(num_levels);
  for (size_type i = 0; i < num_levels; ++i) {
    mask_results.emplace_back(std::move(null_masks[i]), null_counts[i]);
  }

  return {std::move(data_buffers), std::move(mask_results)};
}

/**
 * @brief Reconstructs the column hierarchy from the processed buffers.
 */
std::unique_ptr<column> reconstruct_column(
  std::vector<level_info> const& levels,
  std::vector<rmm::device_buffer>& data_buffers,
  std::vector<std::pair<rmm::device_buffer, size_type>>& mask_results,
  size_type& level_idx,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const& level             = levels[level_idx];
  auto& [null_mask, null_count] = mask_results[level_idx];
  size_type const current_level = level_idx;
  ++level_idx;

  if (level.is_struct) {
    std::vector<std::unique_ptr<column>> children;
    children.reserve(level.num_children);

    for (size_type i = 0; i < level.num_children; ++i) {
      children.push_back(
        reconstruct_column(levels, data_buffers, mask_results, level_idx, stream, mr));
    }

    return make_structs_column(
      level.total_rows, std::move(children), null_count, std::move(null_mask), stream, mr);
  } else {
    return std::make_unique<column>(level.dtype,
                                    level.total_rows,
                                    std::move(data_buffers[current_level]),
                                    std::move(null_mask),
                                    null_count);
  }
}

/**
 * @brief Verifies bounds and type compatibility for batch concatenation.
 */
void batch_bounds_and_type_check(host_span<column_view const> cols, rmm::cuda_stream_view stream)
{
  // Check total row count doesn't exceed size_type limit
  size_t const total_row_count =
    std::accumulate(cols.begin(), cols.end(), std::size_t{}, [](size_t a, auto const& b) {
      return a + static_cast<size_t>(b.size());
    });
  CUDF_EXPECTS(total_row_count <= static_cast<size_t>(std::numeric_limits<size_type>::max()),
               "Total number of concatenated rows exceeds the column size limit",
               std::overflow_error);

  // Handle EMPTY type columns
  if (std::any_of(cols.begin(), cols.end(), [](column_view const& c) {
        return c.type().id() == cudf::type_id::EMPTY;
      })) {
    CUDF_EXPECTS(
      std::all_of(cols.begin(),
                  cols.end(),
                  [](column_view const& c) { return c.type().id() == cudf::type_id::EMPTY; }),
      "Mismatch in columns to concatenate.",
      cudf::data_type_error);
    return;
  }

  // Check all types match
  CUDF_EXPECTS(cudf::all_have_same_types(cols.begin(), cols.end()),
               "Type mismatch in columns to concatenate.",
               cudf::data_type_error);

  // Recursively check struct children
  if (cols.front().type().id() == type_id::STRUCT) {
    auto const num_children = cols.front().num_children();
    std::vector<column_view> nth_children;
    nth_children.reserve(cols.size());

    for (size_type child_idx = 0; child_idx < num_children; ++child_idx) {
      nth_children.clear();
      for (auto const& col : cols) {
        structs_column_view scv(col);
        nth_children.push_back(scv.get_sliced_child(child_idx, stream));
      }
      batch_bounds_and_type_check(nth_children, stream);
    }
  }
}

}  // anonymous namespace

std::unique_ptr<column> batch_concatenate(host_span<column_view const> columns_to_concat,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!columns_to_concat.empty(), "Unexpected empty list of columns to concatenate.");

  // Check that all columns are supported types (plain types or struct types)
  CUDF_EXPECTS(all_columns_batch_concat_supported(columns_to_concat),
               "batch_concatenate only supports fixed-width and struct types. "
               "Lists, strings, and dictionaries are not supported.",
               cudf::logic_error);

  // Verify bounds and type compatibility
  batch_bounds_and_type_check(columns_to_concat, stream);

  // Handle all-empty case
  if (std::all_of(columns_to_concat.begin(), columns_to_concat.end(), [](column_view const& c) {
        return c.is_empty();
      })) {
    return empty_like(columns_to_concat.front());
  }

  // Handle EMPTY type
  if (columns_to_concat.front().type().id() == cudf::type_id::EMPTY) {
    auto length = std::accumulate(
      columns_to_concat.begin(), columns_to_concat.end(), 0, [](auto a, auto const& b) {
        return a + b.size();
      });
    return std::make_unique<column>(
      data_type(type_id::EMPTY), length, rmm::device_buffer{}, rmm::device_buffer{}, length);
  }

  // Step 1: Collect all level information in a single recursive pass
  std::vector<level_info> levels;
  collect_level_info_recursive(columns_to_concat, levels, stream);

  // Step 2: Batch process all data copy and mask concatenation
  auto [data_buffers, mask_results] = batch_process_levels(levels, stream, mr);

  // Step 3: Reconstruct the column hierarchy
  size_type level_idx = 0;
  return reconstruct_column(levels, data_buffers, mask_results, level_idx, stream, mr);
}

bool can_use_batch_concatenate(host_span<column_view const> columns)
{
  if (columns.empty()) return false;
  return all_columns_batch_concat_supported(columns);
}

}  // namespace detail
}  // namespace cudf
