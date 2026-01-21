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
#include <cudf/detail/nvtx/ranges.hpp>
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
  data_type dtype;             // Data type at this level
  size_type total_rows;        // Total number of rows
  size_type total_null_count;  // Sum of null counts from all columns
  size_type num_children;      // Number of children (for struct types)
  bool has_nulls;              // Whether any column has nulls
  bool is_struct;              // Whether this level is a struct type
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
  info.total_null_count = 0;

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
    info.total_null_count += col.null_count();
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
 * @brief Concatenates bitmasks for a single level.
 *
 * Follows the same pattern as concatenate_masks_kernel but without computing null counts
 * (they are pre-computed on the host from input columns).
 *
 * @param mask_ptrs Source mask pointers for columns in this level
 * @param mask_offsets Bit offset for each source mask (from column offset)
 * @param output_offsets Prefix sum of column sizes [num_columns + 1]
 * @param num_columns Number of columns being concatenated
 * @param dest_mask Output mask buffer
 * @param num_mask_bits Total number of bits in the output mask
 */
template <size_type block_size>
CUDF_KERNEL void batch_concatenate_masks_kernel(bitmask_type const* const* __restrict__ mask_ptrs,
                                                size_type const* __restrict__ mask_offsets,
                                                size_type const* __restrict__ output_offsets,
                                                size_type num_columns,
                                                bitmask_type* __restrict__ dest_mask,
                                                size_type num_mask_bits)
{
  auto tidx         = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();
  auto active_mask  = __ballot_sync(0xFFFF'FFFFu, tidx < num_mask_bits);

  while (tidx < num_mask_bits) {
    auto const bit_idx = tidx;

    // Find which column this bit belongs to
    size_type const col_idx =
      thrust::upper_bound(thrust::seq, output_offsets, output_offsets + num_columns + 1, bit_idx) -
      output_offsets - 1;

    bool bit_is_valid = true;
    if (col_idx < num_columns) {
      size_type const bit_in_col   = bit_idx - output_offsets[col_idx];
      bitmask_type const* src_mask = mask_ptrs[col_idx];
      if (src_mask != nullptr) {
        size_type const src_bit_idx = mask_offsets[col_idx] + bit_in_col;
        bit_is_valid                = cudf::bit_is_set(src_mask, src_bit_idx);
      }
    }

    bitmask_type const new_word = __ballot_sync(active_mask, bit_is_valid);

    if (threadIdx.x % warp_size == 0) { dest_mask[word_index(bit_idx)] = new_word; }

    tidx += stride;
    active_mask = __ballot_sync(active_mask, tidx < num_mask_bits);
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
  CUDF_FUNC_RANGE();

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
    cudf::scoped_range r{"concatenate masks"};

    // Process each level with nulls
    for (size_type level_idx : levels_with_nulls) {
      auto const& level        = levels[level_idx];
      auto const num_columns   = level.mask_ptrs.size();
      auto const num_mask_bits = level.total_rows;

      // Build output offsets (prefix sum of column sizes)
      std::vector<size_type> output_offsets(num_columns + 1);
      output_offsets[0] = 0;
      for (size_t col = 0; col < num_columns; ++col) {
        output_offsets[col + 1] = output_offsets[col] + level.col_sizes[col];
      }

      // Upload to device
      auto d_mask_ptrs =
        make_device_uvector_async(level.mask_ptrs, stream, cudf::get_current_device_resource_ref());
      auto d_mask_offsets = make_device_uvector_async(
        level.mask_offsets, stream, cudf::get_current_device_resource_ref());
      auto d_output_offsets =
        make_device_uvector_async(output_offsets, stream, cudf::get_current_device_resource_ref());

      // Launch kernel
      constexpr size_type block_size{256};
      cudf::detail::grid_1d config(num_mask_bits, block_size);

      batch_concatenate_masks_kernel<block_size>
        <<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
          d_mask_ptrs.data(),
          d_mask_offsets.data(),
          d_output_offsets.data(),
          static_cast<size_type>(num_columns),
          static_cast<bitmask_type*>(null_masks[level_idx].data()),
          num_mask_bits);

      // Use pre-computed null count
      null_counts[level_idx] = level.total_null_count;
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
  CUDF_FUNC_RANGE();

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
  CUDF_FUNC_RANGE();

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
