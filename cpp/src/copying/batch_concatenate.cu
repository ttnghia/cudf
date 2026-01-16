/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
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
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>

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
 *
 * @param type The data type to check
 * @return true if the type is supported, false otherwise
 */
bool is_batch_concat_supported_type(data_type type)
{
  return cudf::is_fixed_width(type) || type.id() == type_id::STRUCT;
}

/**
 * @brief Recursively checks if a column and all its children are supported by batch_concatenate.
 *
 * @param col The column view to check
 * @return true if the column and all children are supported types
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
 *
 * @param cols Span of columns to check
 * @return true if all columns are supported
 */
bool all_columns_batch_concat_supported(host_span<column_view const> cols)
{
  return std::all_of(
    cols.begin(), cols.end(), [](auto const& c) { return is_column_batch_concat_supported(c); });
}

/**
 * @brief Verifies bounds and type compatibility for batch concatenation.
 *
 * Similar to the existing bounds_and_type_check but specific to batch concatenate requirements.
 *
 * @param cols Span of columns to check
 * @param stream CUDA stream
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

/**
 * @brief Data structure to hold collected column information at each nesting level.
 */
struct level_column_info {
  std::vector<void const*> data_ptrs;   // Source data pointers
  std::vector<std::size_t> data_sizes;  // Size of each data buffer in bytes
  std::size_t total_size;               // Total size for this level
  size_type total_rows;                 // Total number of rows at this level
  data_type dtype;                      // Data type at this level
  bool has_nulls;                       // Whether any column at this level has nulls
};

/**
 * @brief Data structure to hold bitmask information for concatenation.
 */
struct level_mask_info {
  std::vector<bitmask_type const*> mask_ptrs;  // Source mask pointers (can be null)
  std::vector<size_type> mask_sizes;           // Number of bits in each mask
  std::vector<size_type> mask_offsets;         // Bit offset for each mask (from column offset)
  size_type total_bits;                        // Total number of bits
  bool has_nulls;                              // Whether nulls exist at this level
};

/**
 * @brief Recursively collects all columns at each nesting level.
 *
 * For struct columns, this traverses into children. At each level, it collects
 * all columns that need to be concatenated together.
 *
 * @param cols Span of columns at the current level
 * @param concat_list_plain_types Output vector of vectors of column views per level
 * @param mask_info_list Output vector of mask info per level
 * @param stream CUDA stream
 */
void collect_columns_recursive(host_span<column_view const> cols,
                               std::vector<std::vector<column_view>>& concat_list_plain_types,
                               std::vector<level_mask_info>& mask_info_list,
                               rmm::cuda_stream_view stream)
{
  // Add current level columns
  std::vector<column_view> current_level(cols.begin(), cols.end());
  concat_list_plain_types.push_back(std::move(current_level));

  // Collect mask info for current level
  level_mask_info mask_info;
  mask_info.has_nulls = false;
  mask_info.total_bits =
    std::accumulate(cols.begin(), cols.end(), size_type{0}, [](size_type sum, auto const& col) {
      return sum + col.size();
    });

  for (auto const& col : cols) {
    mask_info.mask_ptrs.push_back(col.null_mask());
    mask_info.mask_sizes.push_back(col.size());
    mask_info.mask_offsets.push_back(col.offset());
    if (col.has_nulls()) { mask_info.has_nulls = true; }
  }
  mask_info_list.push_back(std::move(mask_info));

  // If struct type, recurse into children
  if (!cols.empty() && cols.front().type().id() == type_id::STRUCT) {
    auto const num_children = cols.front().num_children();
    std::vector<column_view> child_cols;
    child_cols.reserve(cols.size());

    for (size_type child_idx = 0; child_idx < num_children; ++child_idx) {
      child_cols.clear();
      for (auto const& col : cols) {
        structs_column_view scv(col);
        child_cols.push_back(scv.get_sliced_child(child_idx, stream));
      }
      collect_columns_recursive(child_cols, concat_list_plain_types, mask_info_list, stream);
    }
  }
}

/**
 * @brief Kernel to concatenate bitmasks for multiple levels in a single launch.
 *
 * This kernel processes all bitmask concatenations across all nesting levels
 * in a single kernel launch. Each thread processes one bit position across
 * the flattened work space of all levels.
 *
 * @tparam block_size The number of threads per block
 * @param mask_ptrs Source mask pointers for all columns across all levels [total_columns]
 * @param mask_offsets Bit offset for each source mask [total_columns]
 * @param level_column_offsets Prefix sum of columns per level [num_levels + 1]
 * @param level_bit_offsets Prefix sum of bits per level (global work distribution) [num_levels + 1]
 * @param within_level_bit_offsets Prefix sum of bits within each level [total_columns + 1]
 * @param dest_masks Output mask pointers for each level [num_levels]
 * @param num_levels Number of levels to process
 * @param total_bits Total number of bits across all levels
 * @param valid_counts Output valid counts per level [num_levels]
 */
template <size_type block_size>
CUDF_KERNEL void batch_concatenate_masks_kernel(
  bitmask_type const* const* mask_ptrs,       // [total_columns] source mask pointers
  size_type const* mask_offsets,              // [total_columns] bit offset in each mask
  size_type const* level_column_offsets,      // [num_levels + 1] prefix sum of columns per level
  size_type const* level_bit_offsets,         // [num_levels + 1] prefix sum of bits per level
  size_type const* within_level_bit_offsets,  // [total_columns + 1] prefix sum of bits within level
  bitmask_type* const* dest_masks,            // [num_levels] output mask per level
  size_type num_levels,
  size_type total_bits,
  size_type* valid_counts)  // [num_levels] output valid count per level
{
  auto tidx         = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();
  auto active_mask  = __ballot_sync(0xFFFF'FFFFu, tidx < total_bits);

  // Per-level valid counts accumulated in this warp
  // We use a simple approach: each warp leader tracks and atomically updates
  size_type warp_valid_count = 0;
  size_type last_level_idx   = -1;

  while (tidx < total_bits) {
    // Find which level this bit belongs to using binary search
    size_type const level_idx =
      thrust::upper_bound(
        thrust::seq, level_bit_offsets, level_bit_offsets + num_levels + 1, tidx) -
      level_bit_offsets - 1;

    // Get the bit index within this level
    size_type const bit_in_level = tidx - level_bit_offsets[level_idx];

    // Find which column within the level this bit belongs to
    size_type const col_start         = level_column_offsets[level_idx];
    size_type const* level_bit_starts = within_level_bit_offsets + col_start;
    size_type const num_cols_in_level = level_column_offsets[level_idx + 1] - col_start;

    size_type const col_in_level =
      thrust::upper_bound(
        thrust::seq, level_bit_starts, level_bit_starts + num_cols_in_level + 1, bit_in_level) -
      level_bit_starts - 1;

    size_type const global_col_idx = col_start + col_in_level;

    // Get the bit within this column
    size_type const bit_in_col = bit_in_level - level_bit_starts[col_in_level];

    // Read the validity bit from source
    bitmask_type const* src_mask = mask_ptrs[global_col_idx];
    size_type const src_offset   = mask_offsets[global_col_idx];
    bool bit_is_set              = true;
    if (src_mask != nullptr) {
      size_type const src_bit_idx = src_offset + bit_in_col;
      bit_is_set                  = (src_mask[src_bit_idx / 32] >> (src_bit_idx % 32)) & 1;
    }

    // Use ballot to collect bits from the warp
    bitmask_type const new_word = __ballot_sync(active_mask, bit_is_set);

    // First thread in warp writes the result and tracks valid count
    if (threadIdx.x % warp_size == 0) {
      bitmask_type* dest             = dest_masks[level_idx];
      dest[word_index(bit_in_level)] = new_word;

      // Track valid counts - when level changes or at end, flush accumulated count
      if (last_level_idx != level_idx && last_level_idx >= 0) {
        atomicAdd(&valid_counts[last_level_idx], warp_valid_count);
        warp_valid_count = 0;
      }
      warp_valid_count += __popc(new_word);
      last_level_idx = level_idx;
    }

    tidx += stride;
    active_mask = __ballot_sync(active_mask, tidx < total_bits);
  }

  // Flush remaining valid count
  if (threadIdx.x % warp_size == 0 && last_level_idx >= 0) {
    atomicAdd(&valid_counts[last_level_idx], warp_valid_count);
  }
}

/**
 * @brief Concatenates masks for all levels using a single batched kernel.
 *
 * This function processes all levels with nulls in a single kernel launch,
 * reducing kernel launch overhead for deeply nested structures.
 *
 * @param mask_info_list Vector of mask info per level
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return Vector of (null_mask_buffer, null_count) pairs per level
 */
std::vector<std::pair<rmm::device_buffer, size_type>> batch_concatenate_masks(
  std::vector<level_mask_info> const& mask_info_list,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  size_type const num_levels = static_cast<size_type>(mask_info_list.size());

  // Pre-allocate result vector with empty buffers
  std::vector<std::pair<rmm::device_buffer, size_type>> results;
  results.reserve(num_levels);

  // Collect levels that have nulls and need processing
  std::vector<size_type> levels_with_nulls;
  for (size_type i = 0; i < num_levels; ++i) {
    if (mask_info_list[i].has_nulls) { levels_with_nulls.push_back(i); }
  }

  // If no levels have nulls, return empty buffers for all
  if (levels_with_nulls.empty()) {
    for (size_type i = 0; i < num_levels; ++i) {
      results.emplace_back(rmm::device_buffer{0, stream, mr}, 0);
    }
    return results;
  }

  // Allocate output masks for all levels (empty for non-null levels)
  std::vector<rmm::device_buffer> null_masks;
  null_masks.reserve(num_levels);
  for (size_type i = 0; i < num_levels; ++i) {
    if (mask_info_list[i].has_nulls) {
      null_masks.emplace_back(cudf::detail::create_null_mask(
        mask_info_list[i].total_bits, mask_state::UNINITIALIZED, stream, mr));
    } else {
      null_masks.emplace_back(0, stream, mr);
    }
  }

  // Build flattened arrays for all columns across all levels (only for levels with nulls)
  std::vector<bitmask_type const*> all_mask_ptrs;
  std::vector<size_type> all_mask_offsets;
  std::vector<size_type> level_column_offsets;      // prefix sum of columns per level
  std::vector<size_type> level_bit_offsets;         // prefix sum of bits per level (global work)
  std::vector<size_type> within_level_bit_offsets;  // prefix sum of bits within each level
  std::vector<bitmask_type*> dest_mask_ptrs;

  level_column_offsets.push_back(0);
  level_bit_offsets.push_back(0);

  size_type total_columns = 0;
  size_type total_bits    = 0;

  for (size_type level_idx : levels_with_nulls) {
    auto const& mask_info = mask_info_list[level_idx];

    // Add destination mask pointer
    dest_mask_ptrs.push_back(static_cast<bitmask_type*>(null_masks[level_idx].data()));

    // Add column data for this level
    size_type level_bit_count = 0;
    for (size_t col = 0; col < mask_info.mask_ptrs.size(); ++col) {
      all_mask_ptrs.push_back(mask_info.mask_ptrs[col]);
      all_mask_offsets.push_back(mask_info.mask_offsets[col]);
      within_level_bit_offsets.push_back(level_bit_count);
      level_bit_count += mask_info.mask_sizes[col];
    }
    // Add final offset for this level
    within_level_bit_offsets.push_back(level_bit_count);

    total_columns += static_cast<size_type>(mask_info.mask_ptrs.size());
    level_column_offsets.push_back(total_columns);

    total_bits += mask_info.total_bits;
    level_bit_offsets.push_back(total_bits);
  }

  // Upload arrays to device
  auto d_mask_ptrs =
    make_device_uvector_async(all_mask_ptrs, stream, cudf::get_current_device_resource_ref());
  auto d_mask_offsets =
    make_device_uvector_async(all_mask_offsets, stream, cudf::get_current_device_resource_ref());
  auto d_level_column_offsets = make_device_uvector_async(
    level_column_offsets, stream, cudf::get_current_device_resource_ref());
  auto d_level_bit_offsets =
    make_device_uvector_async(level_bit_offsets, stream, cudf::get_current_device_resource_ref());
  auto d_within_level_bit_offsets = make_device_uvector_async(
    within_level_bit_offsets, stream, cudf::get_current_device_resource_ref());
  auto d_dest_masks =
    make_device_uvector_async(dest_mask_ptrs, stream, cudf::get_current_device_resource_ref());

  // Allocate and zero-initialize valid counts for levels with nulls
  auto d_valid_counts = make_zeroed_device_uvector_async<size_type>(
    levels_with_nulls.size(), stream, cudf::get_current_device_resource_ref());

  // Launch single kernel for all levels
  constexpr size_type block_size{256};
  cudf::detail::grid_1d config(total_bits, block_size);

  batch_concatenate_masks_kernel<block_size>
    <<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
      d_mask_ptrs.data(),
      d_mask_offsets.data(),
      d_level_column_offsets.data(),
      d_level_bit_offsets.data(),
      d_within_level_bit_offsets.data(),
      d_dest_masks.data(),
      static_cast<size_type>(levels_with_nulls.size()),
      total_bits,
      d_valid_counts.data());

  // Copy valid counts back to host (synchronous)
  auto h_valid_counts = make_host_vector(d_valid_counts, stream);

  // Build result vector
  size_type null_level_idx = 0;
  for (size_type i = 0; i < num_levels; ++i) {
    if (mask_info_list[i].has_nulls) {
      size_type const valid_count = h_valid_counts[null_level_idx];
      size_type const null_count  = mask_info_list[i].total_bits - valid_count;
      results.emplace_back(std::move(null_masks[i]), null_count);
      ++null_level_idx;
    } else {
      results.emplace_back(std::move(null_masks[i]), 0);
    }
  }

  return results;
}

/**
 * @brief Performs batched data copy for all levels using cub::DeviceMemcpy::Batched.
 *
 * @param concat_list_plain_types Vector of column views per level
 * @param dest_buffers Output: vector of allocated destination buffers
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return Vector of (buffer_ptr, buffer_size) for each source column
 */
std::vector<rmm::device_buffer> batch_copy_data(
  std::vector<std::vector<column_view>> const& concat_list_plain_types,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // First pass: collect all source/dest info and allocate destination buffers
  std::vector<void const*> all_src_ptrs;
  std::vector<void*> all_dst_ptrs;
  std::vector<std::size_t> all_sizes;
  std::vector<rmm::device_buffer> dest_buffers;

  for (auto const& level_cols : concat_list_plain_types) {
    if (level_cols.empty()) continue;

    auto const& first_col = level_cols.front();

    // Skip struct columns - they have no data buffer of their own
    if (first_col.type().id() == type_id::STRUCT) {
      // Add empty placeholder buffer for struct level
      dest_buffers.emplace_back(0, stream, mr);
      continue;
    }

    // Calculate total size for this level
    std::size_t const element_size = cudf::size_of(first_col.type());
    std::size_t total_rows         = 0;
    for (auto const& col : level_cols) {
      total_rows += col.size();
    }

    // Allocate destination buffer
    std::size_t const total_bytes = total_rows * element_size;
    dest_buffers.emplace_back(total_bytes, stream, mr);

    // Collect source pointers and sizes
    char* dst_ptr = static_cast<char*>(dest_buffers.back().data());
    for (auto const& col : level_cols) {
      void const* src_ptr = col.head<char>() + (col.offset() * element_size);
      std::size_t size    = col.size() * element_size;

      all_src_ptrs.push_back(src_ptr);
      all_dst_ptrs.push_back(dst_ptr);
      all_sizes.push_back(size);

      dst_ptr += size;
    }
  }

  // Perform batched copy if there are any buffers to copy
  if (!all_src_ptrs.empty()) {
    auto d_src_ptrs =
      make_device_uvector_async(all_src_ptrs, stream, cudf::get_current_device_resource_ref());
    auto d_dst_ptrs =
      make_device_uvector_async(all_dst_ptrs, stream, cudf::get_current_device_resource_ref());
    auto d_sizes =
      make_device_uvector_async(all_sizes, stream, cudf::get_current_device_resource_ref());

    batched_memcpy_async(
      d_src_ptrs.data(), d_dst_ptrs.data(), d_sizes.data(), all_src_ptrs.size(), stream);
  }

  return dest_buffers;
}

/**
 * @brief Reconstructs the column hierarchy from the concatenated data and masks.
 *
 * @param concat_list_plain_types The collected column views per level
 * @param data_buffers The concatenated data buffers per level
 * @param mask_results The concatenated mask buffers and null counts per level
 * @param level_idx Current level index (for recursion)
 * @param buffer_idx Index into data_buffers (tracks struct levels)
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return The reconstructed column
 */
std::unique_ptr<column> reconstruct_column(
  std::vector<std::vector<column_view>> const& concat_list_plain_types,
  std::vector<rmm::device_buffer>& data_buffers,
  std::vector<std::pair<rmm::device_buffer, size_type>>& mask_results,
  size_type& level_idx,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const& level_cols = concat_list_plain_types[level_idx];
  auto const& first_col  = level_cols.front();
  auto const dtype       = first_col.type();

  // Calculate total rows at this level
  size_type const total_rows = std::accumulate(
    level_cols.begin(), level_cols.end(), size_type{0}, [](size_type sum, auto const& col) {
      return sum + col.size();
    });

  // Get mask and null count for this level
  auto& [null_mask, null_count] = mask_results[level_idx];
  size_type current_level       = level_idx;
  ++level_idx;

  if (dtype.id() == type_id::STRUCT) {
    // Reconstruct struct column
    size_type const num_children = first_col.num_children();
    std::vector<std::unique_ptr<column>> children;
    children.reserve(num_children);

    for (size_type i = 0; i < num_children; ++i) {
      children.push_back(reconstruct_column(
        concat_list_plain_types, data_buffers, mask_results, level_idx, stream, mr));
    }

    return make_structs_column(
      total_rows, std::move(children), null_count, std::move(null_mask), stream, mr);
  } else {
    // Fixed-width column: use the data buffer
    auto& data_buffer = data_buffers[current_level];

    return std::make_unique<column>(
      dtype, total_rows, std::move(data_buffer), std::move(null_mask), null_count);
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

  // Step 1: Recursively collect all columns at each nesting level
  std::vector<std::vector<column_view>> concat_list_plain_types;
  std::vector<level_mask_info> mask_info_list;
  collect_columns_recursive(columns_to_concat, concat_list_plain_types, mask_info_list, stream);

  // Step 2: Batch copy all data using cub::DeviceMemcpy::Batched
  auto data_buffers = batch_copy_data(concat_list_plain_types, stream, mr);

  // Step 3: Batch concatenate all masks
  auto mask_results = batch_concatenate_masks(mask_info_list, stream, mr);

  // Step 4: Reconstruct the column hierarchy
  size_type level_idx = 0;
  return reconstruct_column(
    concat_list_plain_types, data_buffers, mask_results, level_idx, stream, mr);
}

bool can_use_batch_concatenate(host_span<column_view const> columns)
{
  if (columns.empty()) return false;
  return all_columns_batch_concat_supported(columns);
}

}  // namespace detail
}  // namespace cudf
