/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Set CONCAT_SEPARATE_COLUMN=1 to concatenate each column position separately
// (calls batch_concatenate(column_view) per column instead of single batched_memcpy for all)
#ifndef CONCAT_SEPARATE_COLUMN
#define CONCAT_SEPARATE_COLUMN 0
#endif

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/detail/concatenate.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/std/iterator>
#include <thrust/binary_search.h>

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace {

/**
 * @brief Checks if the given data type is supported by batch_concatenate.
 *
 * Supported types are: fixed-width types (plain types), struct types, string types,
 * list types, and dictionary types.
 */
bool is_batch_concat_supported_type(data_type type)
{
  return cudf::is_fixed_width(type) || type.id() == type_id::STRUCT ||
         type.id() == type_id::STRING || type.id() == type_id::LIST ||
         type.id() == type_id::DICTIONARY32;
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
  // LIST type: recursively check child column
  if (col.type().id() == type_id::LIST) {
    if (col.size() > 0) {
      lists_column_view lcv(col);
      if (!is_column_batch_concat_supported(lcv.child())) { return false; }
    }
  }
  // STRING and DICTIONARY types: no need to check children - handled specially
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

  // String/List offset concatenation info
  std::vector<void const*> src_offsets_ptrs;         // Source offsets pointers
  std::vector<size_type> src_offsets_offsets;        // Offset within each source offsets column
  std::vector<std::size_t> chars_partition_offsets;  // Cumulative child offsets for each column
  std::vector<int64_t> first_src_offsets;  // First offset value for each column (for adjustment)
  data_type offsets_dtype;                 // Type of offsets column (INT32 or INT64)
  std::size_t total_child_size;            // Total chars bytes (STRING) or child rows (LIST)

  // LIST-specific info
  data_type child_dtype;  // Type of child column (for empty list reconstruction)
  std::optional<column_view> empty_child_like;  // Child column view for empty_like (nested types)

  // DICTIONARY delegation info
  bool delegate_to_specialized;               // If true, skip batch processing and delegate
  std::vector<column_view> original_columns;  // Store column views for delegation

  // Level metadata
  data_type dtype;             // Data type at this level
  int64_t total_rows;          // Total number of rows (int64_t to avoid overflow)
  size_type total_null_count;  // Sum of null counts from all columns
  size_type num_children;      // Number of children (for struct types)
  bool has_nulls;              // Whether any column has nulls
  bool is_struct;              // Whether this level is a struct type
  bool is_string;              // Whether this level is a string type
  bool is_list;                // Whether this level is a list type
  bool is_dictionary;          // Whether this level is a dictionary type
};

/**
 * @brief Groups string/list levels by total_rows for 2D kernel launch.
 *
 * This structure aggregates data from multiple columns that have the same total_rows,
 * allowing them to be processed with a single 2D kernel launch.
 */
struct offsets_kernel_group {
  int64_t total_rows{0};  // All levels in this group have same total_rows

  // Flattened arrays across all columns in group
  std::vector<void const*> all_src_offsets_ptrs;
  std::vector<size_type> all_src_offsets_offsets;
  std::vector<size_t> all_input_offsets;
  std::vector<size_t> all_chars_partition_offsets;
  std::vector<int64_t> all_first_src_offsets;

  // Per-column metadata (indexed by blockIdx.y in 2D kernel)
  std::vector<size_type> col_src_ptr_offsets;        // Boundaries into all_src_* arrays
  std::vector<size_type> col_input_offsets_offsets;  // Boundaries into all_input_offsets
  std::vector<size_type> col_num_src_columns;
  std::vector<cudf::detail::output_offsetalator> col_output_data;
  std::vector<size_t> col_total_child_size;
  std::vector<size_type> col_offsets_type_size;

  // Track which (col_idx, level_idx) pairs are in this group
  std::vector<std::pair<size_type, size_type>> col_level_indices;
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
  auto const& first_col        = cols.front();
  info.dtype                   = first_col.type();
  info.is_struct               = (info.dtype.id() == type_id::STRUCT);
  info.is_string               = (info.dtype.id() == type_id::STRING);
  info.is_list                 = (info.dtype.id() == type_id::LIST);
  info.is_dictionary           = (info.dtype.id() == type_id::DICTIONARY32);
  info.delegate_to_specialized = info.is_dictionary;
  info.num_children            = first_col.num_children();
  info.has_nulls               = false;
  info.total_rows              = 0;
  info.total_null_count        = 0;
  info.total_child_size        = 0;

  // Compute element size once for fixed-width types
  std::size_t const element_size =
    (info.is_struct || info.is_string || info.is_list || info.is_dictionary)
      ? 0
      : cudf::size_of(info.dtype);

  // Reserve space to avoid reallocations
  auto const num_cols = cols.size();
  info.mask_ptrs.reserve(num_cols);
  info.mask_offsets.reserve(num_cols);
  info.col_sizes.reserve(num_cols);

  // Handle DICTIONARY delegation - collect columns and return early
  if (info.delegate_to_specialized) {
    // Collect basic mask info for all columns first
    for (auto const& col : cols) {
      auto const col_size = col.size();
      info.total_rows += col_size;
      info.col_sizes.push_back(col_size);
      info.mask_ptrs.push_back(col.null_mask());
      info.mask_offsets.push_back(col.offset());
      info.total_null_count += col.null_count();
      if (col.has_nulls()) { info.has_nulls = true; }
      info.original_columns.push_back(col);
    }
    levels.push_back(std::move(info));
    return;  // Don't recurse - specialized implementation handles everything
  }

  if (!info.is_struct && !info.is_string && !info.is_list) {
    info.src_data_ptrs.reserve(num_cols);
    info.data_sizes.reserve(num_cols);
  }
  if (info.is_string) {
    info.src_offsets_ptrs.reserve(num_cols);
    info.src_offsets_offsets.reserve(num_cols);
    info.chars_partition_offsets.reserve(num_cols + 1);
    info.chars_partition_offsets.push_back(0);  // First offset is 0
    info.first_src_offsets.reserve(num_cols);   // Pre-computed first offsets
    info.src_data_ptrs.reserve(num_cols);       // For chars data
    info.data_sizes.reserve(num_cols);          // For chars sizes
    // Get offsets type from the first non-empty column's offsets child
    // Empty string columns have no children
    info.offsets_dtype = data_type{type_id::INT32};  // Default
    for (auto const& col : cols) {
      if (col.size() > 0) {
        strings_column_view scv(col);
        info.offsets_dtype = scv.offsets().type();
        break;
      }
    }
  }
  if (info.is_list) {
    info.src_offsets_ptrs.reserve(num_cols);
    info.src_offsets_offsets.reserve(num_cols);
    info.chars_partition_offsets.reserve(num_cols + 1);
    info.chars_partition_offsets.push_back(0);  // First offset is 0
    info.first_src_offsets.reserve(num_cols);
    info.offsets_dtype = data_type{type_id::INT32};  // LIST offsets are always INT32

    // Store child info for empty list reconstruction
    // Find first non-empty column to get child type and view
    for (auto const& col : cols) {
      if (col.size() > 0) {
        lists_column_view lcv(col);
        info.child_dtype      = lcv.child().type();
        info.empty_child_like = lcv.child();  // Store for empty_like on nested types
        break;
      }
    }
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

    // Data info based on type
    if (info.is_string) {
      // String column: collect offsets and chars info
      // Note: empty string columns have no children, so we must check size first
      if (col_size > 0) {
        strings_column_view scv(col);
        auto const offsets_col = scv.offsets();

        // Offsets info
        info.src_offsets_ptrs.push_back(offsets_col.head());
        info.src_offsets_offsets.push_back(col.offset());

        // Chars info - need to get the actual char bytes for this column
        auto const [first_offset, last_offset] =
          cudf::strings::detail::get_first_and_last_offset(scv, stream);
        auto const chars_bytes = last_offset - first_offset;
        // For sliced columns, chars start at first_offset from the beginning
        info.src_data_ptrs.push_back(scv.chars_begin(stream) + first_offset);
        info.data_sizes.push_back(chars_bytes);
        info.total_child_size += chars_bytes;
        info.chars_partition_offsets.push_back(info.total_child_size);
        // Store first offset for kernel optimization (avoid redundant reads)
        info.first_src_offsets.push_back(first_offset);
      } else {
        // Empty column - push placeholders
        info.src_offsets_ptrs.push_back(nullptr);
        info.src_offsets_offsets.push_back(0);
        info.src_data_ptrs.push_back(nullptr);
        info.data_sizes.push_back(0);
        info.chars_partition_offsets.push_back(info.total_child_size);
        info.first_src_offsets.push_back(0);
      }
    } else if (info.is_list) {
      // List column: collect offsets info (child data handled via recursion)
      if (col_size > 0) {
        lists_column_view lcv(col);
        auto const offsets_col = lcv.offsets();

        // Offsets info
        info.src_offsets_ptrs.push_back(offsets_col.head());
        info.src_offsets_offsets.push_back(col.offset());

        // Get child boundaries (requires GPU read for sliced columns)
        size_type child_start = 0;
        size_type child_end   = 0;
        if (col.offset() > 0) {
          child_start = cudf::detail::get_value<size_type>(offsets_col, col.offset(), stream);
        }
        child_end =
          cudf::detail::get_value<size_type>(offsets_col, col.offset() + col_size, stream);

        auto const child_rows = child_end - child_start;
        info.first_src_offsets.push_back(child_start);
        info.total_child_size += child_rows;
        info.chars_partition_offsets.push_back(info.total_child_size);
      } else {
        // Empty column - push placeholders
        info.src_offsets_ptrs.push_back(nullptr);
        info.src_offsets_offsets.push_back(0);
        info.first_src_offsets.push_back(0);
        info.chars_partition_offsets.push_back(info.total_child_size);
      }
    } else if (!info.is_struct) {
      // Fixed-width type
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

  // Recurse into list children
  if (first_col.type().id() == type_id::LIST) {
    std::vector<column_view> child_cols;
    child_cols.reserve(num_cols);

    for (auto const& col : cols) {
      if (col.size() > 0) {
        lists_column_view lcv(col);
        child_cols.push_back(lcv.get_sliced_child(stream));
      }
    }

    // Only recurse if there are non-empty columns
    // If all columns are empty, total_child_size will be 0 and
    // the child level will have 0 rows, which is handled correctly
    if (!child_cols.empty()) { collect_level_info_recursive(child_cols, levels, stream); }
  }

  // Note: STRING type children (offsets, chars) are handled specially, not recursively
  // Note: DICTIONARY type is handled via delegation, not recursively
}

/**
 * @brief Concatenates bitmasks for a single level using word-level operations.
 *
 * Each thread processes one output word (32 bits). The algorithm:
 * 1. Find which column(s) contribute to this output word
 * 2. Read one or two source words into a uint64_t
 * 3. Shift the bits to align with the output position
 * 4. Write the output word (with special handling for column boundaries)
 *
 * For boundary handling: when an output word spans two columns, the thread processing
 * the first word of the later column will merge the trailing bits from the previous
 * column with its own bits.
 *
 * @param mask_ptrs Source mask pointers for columns in this level
 * @param mask_offsets Bit offset for each source mask (from column offset)
 * @param output_offsets Prefix sum of column sizes (in bits) [num_columns + 1]
 * @param num_columns Number of columns being concatenated
 * @param dest_mask Output mask buffer
 * @param num_output_words Total number of words in the output mask
 */
// Maximum number of columns that can fit in shared memory (num_columns + 1 offsets)
constexpr size_type max_smem_columns = 1024;

template <size_type block_size>
CUDF_KERNEL void batch_concatenate_masks_kernel(bitmask_type const* const* __restrict__ mask_ptrs,
                                                size_type const* __restrict__ mask_offsets,
                                                size_type const* __restrict__ output_offsets,
                                                size_type num_columns,
                                                bitmask_type* __restrict__ dest_mask,
                                                size_type num_output_words)
{
  constexpr size_type bits_per_word = detail::size_in_bits<bitmask_type>();

  // Load output_offsets into shared memory if it fits
  __shared__ size_type smem_output_offsets[max_smem_columns + 1];
  bool const use_smem = (num_columns <= max_smem_columns);

  if (use_smem) {
    // Cooperatively load output_offsets into shared memory
    for (size_type i = threadIdx.x; i <= num_columns; i += block_size) {
      smem_output_offsets[i] = output_offsets[i];
    }
    __syncthreads();
  }

  // Pointer to use for output_offsets lookups
  size_type const* offsets_ptr = use_smem ? smem_output_offsets : output_offsets;

  auto word_idx     = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();

  while (word_idx < num_output_words) {
    // Bit range for this output word
    size_type const out_bit_start = word_idx * bits_per_word;
    size_type const out_bit_end   = out_bit_start + bits_per_word;  // exclusive

    // Find which column contains the start bit of this word
    size_type col_idx =
      thrust::upper_bound(thrust::seq, offsets_ptr, offsets_ptr + num_columns + 1, out_bit_start) -
      offsets_ptr - 1;

    bitmask_type output_word = 0;
    size_type bits_filled    = 0;

    // Process columns that contribute to this output word
    while (bits_filled < bits_per_word && col_idx < num_columns) {
      size_type const col_start_bit = offsets_ptr[col_idx];
      size_type const col_end_bit   = offsets_ptr[col_idx + 1];
      size_type const col_size      = col_end_bit - col_start_bit;

      // Calculate bit positions within this column
      size_type const current_out_bit = out_bit_start + bits_filled;
      size_type const bit_in_col      = current_out_bit - col_start_bit;

      // How many bits can we take from this column?
      size_type const bits_remaining_in_col = col_size - bit_in_col;
      size_type const bits_needed           = bits_per_word - bits_filled;
      size_type const bits_to_copy          = min(bits_remaining_in_col, bits_needed);

      bitmask_type const* src_mask = mask_ptrs[col_idx];

      if (src_mask == nullptr) {
        // No mask means all valid (all 1s)
        bitmask_type const all_ones_mask = (bits_to_copy == bits_per_word)
                                             ? ~bitmask_type{0}
                                             : ((bitmask_type{1} << bits_to_copy) - 1);
        output_word |= (all_ones_mask << bits_filled);
      } else {
        // Calculate source bit position
        size_type const src_bit_idx   = mask_offsets[col_idx] + bit_in_col;
        size_type const src_word_idx  = src_bit_idx / bits_per_word;
        size_type const src_bit_shift = src_bit_idx % bits_per_word;

        // Read source bits using funnel shift for unaligned access
        // __funnelshift_r(lo, hi, shift) extracts bits [shift, shift+32) from {hi, lo}
        // When shift=0, it returns lo unchanged, so we can use it unconditionally
        bitmask_type const lo = src_mask[src_word_idx];
        // Only read next word if we actually need bits from it
        bitmask_type const hi = (src_bit_shift > 0) ? src_mask[src_word_idx + 1] : 0;
        bitmask_type src_bits = __funnelshift_r(lo, hi, src_bit_shift);

        // Mask to get only the bits we need
        bitmask_type const bits_mask      = (bits_to_copy == bits_per_word)
                                              ? ~bitmask_type{0}
                                              : ((bitmask_type{1} << bits_to_copy) - 1);
        bitmask_type const extracted_bits = src_bits & bits_mask;

        // Place the extracted bits at the correct position in the output word
        output_word |= (extracted_bits << bits_filled);
      }

      bits_filled += bits_to_copy;
      col_idx++;
    }

    // Write the output word
    dest_mask[word_idx] = output_word;

    word_idx += stride;
  }
}

/**
 * @brief Concatenates offsets from multiple string or list columns.
 *
 * Each thread processes one output offset element. The algorithm:
 * 1. Find which source column contributes this element
 * 2. Read the source offset value
 * 3. Adjust the offset: subtract the first offset of the source column
 *    and add the cumulative child offset for that column
 * 4. Write to output using output_offsetalator
 *
 * Optimizations:
 * - Uses shared memory to cache input_offsets for binary search
 * - Pre-computes first offset values on host to avoid redundant device reads
 *
 * @param src_offsets_ptrs Array of source offsets column data pointers
 * @param src_offsets_offsets Array of offsets within each source offsets column (parent offset)
 * @param input_offsets Prefix sum of row counts [num_columns + 1] - for finding source column
 * @param chars_partition_offsets Cumulative child offsets [num_columns + 1]
 * @param first_src_offsets Pre-computed first offset value for each column (for adjustment)
 * @param num_columns Number of columns being concatenated
 * @param output_size Number of rows (not including final offset element)
 * @param output_data Output offsetalator
 * @param total_child_size Total child bytes/rows (for final offset)
 * @param offsets_type_size Size of offset type in bytes (4 for INT32, 8 for INT64)
 */
template <size_type block_size>
CUDF_KERNEL void batch_concatenate_offsets_kernel(
  void const* const* __restrict__ src_offsets_ptrs,
  size_type const* __restrict__ src_offsets_offsets,
  size_t const* __restrict__ input_offsets,
  size_t const* __restrict__ chars_partition_offsets,
  int64_t const* __restrict__ first_src_offsets,
  size_type num_columns,
  int64_t output_size,
  cudf::detail::output_offsetalator output_data,
  size_t total_child_size,
  size_type offsets_type_size)
{
  // Use shared memory to cache input_offsets for binary search
  __shared__ size_t smem_input_offsets[max_smem_columns + 1];
  bool const use_smem = (num_columns <= max_smem_columns);

  if (use_smem) {
    // Cooperatively load input_offsets into shared memory
    for (size_type i = threadIdx.x; i <= num_columns; i += block_size) {
      smem_input_offsets[i] = input_offsets[i];
    }
    __syncthreads();
  }

  size_t const* offsets_ptr = use_smem ? smem_input_offsets : input_offsets;

  auto output_index   = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride   = cudf::detail::grid_1d::grid_stride<block_size>();
  bool const is_int64 = (offsets_type_size == 8);

  while (output_index < output_size) {
    // Find which source column contains this output index
    auto const offset_it = cuda::std::prev(
      thrust::upper_bound(thrust::seq, offsets_ptr, offsets_ptr + num_columns, output_index));
    size_type const col_idx = offset_it - offsets_ptr;

    // Index within this source column
    auto const idx_in_col    = output_index - *offset_it;
    auto const parent_offset = src_offsets_offsets[col_idx];
    void const* src_ptr      = src_offsets_ptrs[col_idx];

    // Read source offset value
    int64_t src_offset_value;
    if (is_int64) {
      auto const* typed_ptr = static_cast<int64_t const*>(src_ptr);
      src_offset_value      = typed_ptr[idx_in_col + parent_offset];
    } else {
      auto const* typed_ptr = static_cast<int32_t const*>(src_ptr);
      src_offset_value      = typed_ptr[idx_in_col + parent_offset];
    }

    // Compute output offset: subtract first offset and add cumulative char offset
    // first_src_offsets[col_idx] is pre-computed on host to avoid redundant device reads
    output_data[output_index] =
      src_offset_value - first_src_offsets[col_idx] + chars_partition_offsets[col_idx];

    output_index += stride;
  }

  // Handle the final offset element (total chars bytes)
  if (output_index == output_size) { output_data[output_size] = total_child_size; }
}

/**
 * @brief 2D kernel for batch offsets concatenation.
 *
 * Processes multiple output columns with the same total_rows in a single launch.
 * Grid: (ceil((total_rows+1)/block_size), num_cols_in_group)
 * Each column in Y dimension (blockIdx.y) processes independently.
 *
 * The flattened arrays contain data for all columns in the group concatenated together.
 * Each column's portion is accessed via boundary arrays (col_src_ptr_offsets, etc.).
 *
 * @param all_src_offsets_ptrs Flattened source offsets pointers for all columns
 * @param all_src_offsets_offsets Flattened offsets within each source offsets column
 * @param all_input_offsets Flattened prefix sums of row counts for all columns
 * @param all_chars_partition_offsets Flattened cumulative child offsets for all columns
 * @param all_first_src_offsets Flattened first offset values for all columns
 * @param col_src_ptr_offsets Boundaries into all_src_* arrays [num_cols+1]
 * @param col_input_offsets_offsets Boundaries into all_input_offsets [num_cols+1]
 * @param col_num_src_columns Number of source columns per output column [num_cols]
 * @param col_output_data Output offsetalators for each column [num_cols]
 * @param col_total_child_size Total child size for each column [num_cols]
 * @param col_offsets_type_size Offset type size for each column [num_cols]
 * @param output_size Number of rows (same for all columns in group)
 */
template <size_type block_size>
CUDF_KERNEL void batch_concatenate_offsets_kernel_2d(
  void const* const* __restrict__ all_src_offsets_ptrs,
  size_type const* __restrict__ all_src_offsets_offsets,
  size_t const* __restrict__ all_input_offsets,
  size_t const* __restrict__ all_chars_partition_offsets,
  int64_t const* __restrict__ all_first_src_offsets,
  size_type const* __restrict__ col_src_ptr_offsets,
  size_type const* __restrict__ col_input_offsets_offsets,
  size_type const* __restrict__ col_num_src_columns,
  output_offsetalator const* __restrict__ col_output_data,
  size_t const* __restrict__ col_total_child_size,
  size_type const* __restrict__ col_offsets_type_size,
  int64_t output_size)
{
  auto const col_in_group = blockIdx.y;

  // Get this column's metadata
  auto const src_ptr_start     = col_src_ptr_offsets[col_in_group];
  auto const input_off_start   = col_input_offsets_offsets[col_in_group];
  auto const num_columns       = col_num_src_columns[col_in_group];
  auto output_data             = col_output_data[col_in_group];
  auto const total_child_size  = col_total_child_size[col_in_group];
  auto const offsets_type_size = col_offsets_type_size[col_in_group];

  // Use shared memory to cache input_offsets for binary search
  __shared__ size_t smem_input_offsets[max_smem_columns + 1];
  bool const use_smem = (num_columns <= max_smem_columns);

  if (use_smem) {
    // Cooperatively load input_offsets into shared memory
    auto const input_offsets = all_input_offsets + input_off_start;
    for (size_type i = threadIdx.x; i <= num_columns; i += block_size) {
      smem_input_offsets[i] = input_offsets[i];
    }
    __syncthreads();
  }

  // Get this column's slice of the flattened arrays
  auto const src_offsets_ptrs    = all_src_offsets_ptrs + src_ptr_start;
  auto const src_offsets_offsets = all_src_offsets_offsets + src_ptr_start;
  auto const input_offsets = use_smem ? smem_input_offsets : (all_input_offsets + input_off_start);
  auto const chars_partition_offsets = all_chars_partition_offsets + input_off_start;
  auto const first_src_offsets       = all_first_src_offsets + src_ptr_start;

  bool const is_int64 = (offsets_type_size == 8);

  auto output_index = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();

  while (output_index <= output_size) {
    if (output_index < output_size) {
      // Binary search to find which source column contains this output index
      auto const offset_it = cuda::std::prev(
        thrust::upper_bound(thrust::seq, input_offsets, input_offsets + num_columns, output_index));
      size_type const col_idx = offset_it - input_offsets;

      // Index within this source column
      auto const idx_in_col    = output_index - *offset_it;
      auto const parent_offset = src_offsets_offsets[col_idx];
      void const* src_ptr      = src_offsets_ptrs[col_idx];

      // Read source offset value
      int64_t src_offset_value;
      if (is_int64) {
        auto const* typed_ptr = static_cast<int64_t const*>(src_ptr);
        src_offset_value      = typed_ptr[idx_in_col + parent_offset];
      } else {
        auto const* typed_ptr = static_cast<int32_t const*>(src_ptr);
        src_offset_value      = typed_ptr[idx_in_col + parent_offset];
      }

      // Compute output offset: subtract first offset and add cumulative child offset
      output_data[output_index] =
        src_offset_value - first_src_offsets[col_idx] + chars_partition_offsets[col_idx];
    }

    // Handle the final offset element
    if (output_index == output_size) { output_data[output_size] = total_child_size; }

    output_index += stride;
  }
}

/**
 * @brief Get maximum active blocks per SM for batch_concatenate_offsets_kernel.
 */
inline int32_t max_active_blocks_offsets_kernel()
{
  int32_t max_active_blocks{-1};
  constexpr size_type block_size{256};
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_blocks, batch_concatenate_offsets_kernel<block_size>, block_size, 0));
  return max_active_blocks;
}

/**
 * @brief Get maximum active blocks per SM for batch_concatenate_offsets_kernel_2d.
 */
inline int32_t max_active_blocks_offsets_kernel_2d()
{
  int32_t max_active_blocks{-1};
  constexpr size_type block_size{256};
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_blocks, batch_concatenate_offsets_kernel_2d<block_size>, block_size, 0));
  return max_active_blocks;
}

/**
 * @brief Result structure for batch_process_levels.
 */
struct batch_process_result {
  std::vector<rmm::device_buffer> data_buffers;  // Data (or chars for strings)
  std::vector<std::pair<rmm::device_buffer, size_type>> mask_results;  // Null masks with counts
  std::vector<std::unique_ptr<column>> offsets_columns;  // Offsets columns for strings
};

/**
 * @brief Performs all batch operations: data copy, mask concatenation, and string offset handling.
 *
 * Returns the output data buffers, mask buffers with null counts, and offsets columns for strings.
 */
batch_process_result batch_process_levels(std::vector<level_info> const& levels,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  size_type const num_levels = static_cast<size_type>(levels.size());

  // ========== PHASE 1: Allocate output buffers ==========
  std::vector<rmm::device_buffer> data_buffers;
  std::vector<rmm::device_buffer> null_masks;
  std::vector<std::unique_ptr<column>> offsets_columns;
  data_buffers.reserve(num_levels);
  null_masks.reserve(num_levels);
  offsets_columns.resize(num_levels);  // Initialize with nullptrs

  for (size_type level_idx = 0; level_idx < num_levels; ++level_idx) {
    auto const& level = levels[level_idx];

    // Skip allocation for delegated levels (e.g., DICTIONARY)
    if (level.delegate_to_specialized) {
      data_buffers.emplace_back(0, stream, mr);
      null_masks.emplace_back(0, stream, mr);
      continue;
    }

    // Data buffer
    if (level.is_struct) {
      data_buffers.emplace_back(0, stream, mr);
    } else if (level.is_string) {
      // For strings, data_buffer holds chars
      data_buffers.emplace_back(level.total_child_size, stream, mr);
      // Allocate offsets column
      auto const offsets_count   = level.total_rows + 1;
      offsets_columns[level_idx] = cudf::strings::detail::create_offsets_child_column(
        static_cast<int64_t>(level.total_child_size), offsets_count, stream, mr);
    } else if (level.is_list) {
      // For lists, no data buffer at this level (child data handled via recursion)
      data_buffers.emplace_back(0, stream, mr);
      // Allocate INT32 offsets column
      auto const offsets_count   = level.total_rows + 1;
      offsets_columns[level_idx] = cudf::make_numeric_column(
        data_type{type_id::INT32}, offsets_count, mask_state::UNALLOCATED, stream, mr);
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
  // Skip struct, list, and delegated levels (they have no data to copy at this level)
  size_t total_copy_ops = 0;
  for (auto const& level : levels) {
    if (level.delegate_to_specialized) continue;
    if (!level.is_struct && !level.is_list && !level.src_data_ptrs.empty()) {
      total_copy_ops += level.src_data_ptrs.size();
    }
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
      if (level.delegate_to_specialized) continue;
      if (level.is_struct || level.is_list || level.src_data_ptrs.empty()) continue;

      char* dst_ptr = static_cast<char*>(data_buffers[level_idx].data());
      for (size_t col = 0; col < level.src_data_ptrs.size(); ++col) {
        if (level.src_data_ptrs[col] != nullptr && level.data_sizes[col] > 0) {
          all_src_ptrs.push_back(level.src_data_ptrs[col]);
          all_dst_ptrs.push_back(dst_ptr);
          all_sizes.push_back(level.data_sizes[col]);
        }
        dst_ptr += level.data_sizes[col];
      }
    }

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
  }

  // ========== PHASE 3: String/List offsets concatenation ==========
  for (size_type level_idx = 0; level_idx < num_levels; ++level_idx) {
    auto const& level = levels[level_idx];
    if (!level.is_string && !level.is_list) continue;

    cudf::scoped_range r{level.is_string ? "concatenate string offsets"
                                         : "concatenate list offsets"};

    auto const num_columns = level.src_offsets_ptrs.size();

    // Build input offsets (prefix sum of row counts for finding source column)
    std::vector<size_t> input_offsets(num_columns + 1);
    input_offsets[0] = 0;
    for (size_t col = 0; col < num_columns; ++col) {
      input_offsets[col + 1] = input_offsets[col] + level.col_sizes[col];
    }

    // Upload to device
    auto d_src_offsets_ptrs = make_device_uvector_async(
      level.src_offsets_ptrs, stream, cudf::get_current_device_resource_ref());
    auto d_src_offsets_offsets = make_device_uvector_async(
      level.src_offsets_offsets, stream, cudf::get_current_device_resource_ref());
    auto d_input_offsets =
      make_device_uvector_async(input_offsets, stream, cudf::get_current_device_resource_ref());
    auto d_chars_partition_offsets = make_device_uvector_async(
      level.chars_partition_offsets, stream, cudf::get_current_device_resource_ref());
    auto d_first_src_offsets = make_device_uvector_async(
      level.first_src_offsets, stream, cudf::get_current_device_resource_ref());

    // Get output offsetalator
    auto output_offsets_view = offsets_columns[level_idx]->mutable_view();
    auto output_offsetalator = offsetalator_factory::make_output_iterator(output_offsets_view);

    // Determine offset type size
    size_type const offsets_type_size = cudf::size_of(level.offsets_dtype);

    // Launch kernel with occupancy-based grid sizing
    constexpr size_type block_size{256};
    auto const max_blocks     = max_active_blocks_offsets_kernel();
    auto const num_sms        = cudf::detail::num_multiprocessors();
    auto const occupancy_grid = max_blocks * num_sms;
    auto const num_elements   = level.total_rows + 1;  // +1 for final offset
    // Use smaller of: occupancy-based grid or element count (avoid over-launching)
    auto const num_blocks =
      std::min(occupancy_grid, cudf::util::div_rounding_up_safe(num_elements, block_size));

    batch_concatenate_offsets_kernel<block_size>
      <<<num_blocks, block_size, 0, stream.value()>>>(d_src_offsets_ptrs.data(),
                                                      d_src_offsets_offsets.data(),
                                                      d_input_offsets.data(),
                                                      d_chars_partition_offsets.data(),
                                                      d_first_src_offsets.data(),
                                                      static_cast<size_type>(num_columns),
                                                      level.total_rows,
                                                      output_offsetalator,
                                                      level.total_child_size,
                                                      offsets_type_size);
  }

  // ========== PHASE 4: Batch mask concatenation ==========
  // Collect levels with nulls (skip delegated levels - their masks are handled separately)
  std::vector<size_type> levels_with_nulls;
  for (size_type i = 0; i < num_levels; ++i) {
    if (levels[i].delegate_to_specialized) continue;
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

      // Launch kernel - one thread per output word
      constexpr size_type block_size{256};
      auto const num_output_words = cudf::util::div_rounding_up_safe(num_mask_bits, 32);
      cudf::detail::grid_1d config(num_output_words, block_size);

      batch_concatenate_masks_kernel<block_size>
        <<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
          d_mask_ptrs.data(),
          d_mask_offsets.data(),
          d_output_offsets.data(),
          static_cast<size_type>(num_columns),
          static_cast<bitmask_type*>(null_masks[level_idx].data()),
          num_output_words);

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

  return {std::move(data_buffers), std::move(mask_results), std::move(offsets_columns)};
}

/**
 * @brief Reconstructs the column hierarchy from the processed buffers.
 */
std::unique_ptr<column> reconstruct_column(
  std::vector<level_info> const& levels,
  std::vector<rmm::device_buffer>& data_buffers,
  std::vector<std::pair<rmm::device_buffer, size_type>>& mask_results,
  std::vector<std::unique_ptr<column>>& offsets_columns,
  size_type& level_idx,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto const& level             = levels[level_idx];
  auto& [null_mask, null_count] = mask_results[level_idx];
  size_type const current_level = level_idx;
  ++level_idx;

  // DICTIONARY: delegate to specialized implementation
  if (level.is_dictionary) {
    return cudf::dictionary::detail::concatenate(level.original_columns, stream, mr);
  }

  if (level.is_struct) {
    std::vector<std::unique_ptr<column>> children;
    children.reserve(level.num_children);

    for (size_type i = 0; i < level.num_children; ++i) {
      children.push_back(reconstruct_column(
        levels, data_buffers, mask_results, offsets_columns, level_idx, stream, mr));
    }

    return make_structs_column(
      level.total_rows, std::move(children), null_count, std::move(null_mask), stream, mr);
  } else if (level.is_list) {
    // LIST column: construct from offsets column and recursively reconstructed child
    std::unique_ptr<column> child_column;

    // Handle edge case: all lists were empty, so no child level exists
    if (level.total_child_size == 0) {
      // Create empty child column of correct type
      // Use empty_like for nested types, make_empty_column for simple types
      if (level.empty_child_like.has_value()) {
        child_column = empty_like(level.empty_child_like.value());
      } else {
        // Fallback for simple types (or if no non-empty column was found)
        child_column = make_empty_column(level.child_dtype);
      }
    } else {
      // Recursively reconstruct child column from next level
      child_column = reconstruct_column(
        levels, data_buffers, mask_results, offsets_columns, level_idx, stream, mr);
    }

    return make_lists_column(level.total_rows,
                             std::move(offsets_columns[current_level]),
                             std::move(child_column),
                             null_count,
                             std::move(null_mask),
                             stream,
                             mr);
  } else if (level.is_string) {
    // String column: construct from offsets column and chars buffer
    return make_strings_column(level.total_rows,
                               std::move(offsets_columns[current_level]),
                               std::move(data_buffers[current_level]),
                               null_count,
                               std::move(null_mask));
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

  // Recursively check list children
  if (cols.front().type().id() == type_id::LIST) {
    // Check offset overflow
    size_t const total_offset_count =
      std::accumulate(cols.begin(),
                      cols.end(),
                      std::size_t{},
                      [](size_t a, auto const& b) { return a + static_cast<size_t>(b.size()); }) +
      1;
    CUDF_EXPECTS(total_offset_count <= static_cast<size_t>(std::numeric_limits<size_type>::max()),
                 "Total number of concatenated offsets exceeds the column size limit",
                 std::overflow_error);

    // Recursively check children
    std::vector<column_view> child_cols;
    child_cols.reserve(cols.size());
    for (auto const& col : cols) {
      if (col.size() > 0) {
        lists_column_view lcv(col);
        child_cols.push_back(lcv.get_sliced_child(stream));
      }
    }
    if (!child_cols.empty()) { batch_bounds_and_type_check(child_cols, stream); }
  }
}

}  // anonymous namespace

std::unique_ptr<column> batch_concatenate(host_span<column_view const> columns_to_concat,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(!columns_to_concat.empty(), "Unexpected empty list of columns to concatenate.");

  // Check that all columns are supported types
  CUDF_EXPECTS(all_columns_batch_concat_supported(columns_to_concat),
               "batch_concatenate only supports fixed-width, struct, string, list, and dictionary "
               "types.",
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

  // Step 2: Batch process all data copy, mask concatenation, and string offsets
  auto result = batch_process_levels(levels, stream, mr);

  // Step 3: Reconstruct the column hierarchy
  size_type level_idx = 0;
  return reconstruct_column(levels,
                            result.data_buffers,
                            result.mask_results,
                            result.offsets_columns,
                            level_idx,
                            stream,
                            mr);
}

bool can_use_batch_concatenate(host_span<column_view const> columns)
{
  if (columns.empty()) return false;
  return all_columns_batch_concat_supported(columns);
}

/**
 * @brief Result structure for batch_process_table_levels.
 */
struct table_batch_process_result {
  std::vector<std::vector<rmm::device_buffer>> data_buffers_by_column;
  std::vector<std::vector<std::pair<rmm::device_buffer, size_type>>> mask_results_by_column;
  std::vector<std::vector<std::unique_ptr<column>>> offsets_columns_by_column;
};

/**
 * @brief Performs all batch operations for table concatenation.
 *
 * This function processes ALL columns from ALL tables with a SINGLE batched_memcpy_async call
 * for maximum efficiency.
 */
table_batch_process_result batch_process_table_levels(
  std::vector<std::vector<level_info>> const& levels_by_column,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  size_type const num_columns = static_cast<size_type>(levels_by_column.size());

  // ========== PHASE 1: Allocate output buffers for ALL columns ==========
  std::vector<std::vector<rmm::device_buffer>> data_buffers_by_column(num_columns);
  std::vector<std::vector<rmm::device_buffer>> null_masks_by_column(num_columns);
  std::vector<std::vector<std::unique_ptr<column>>> offsets_columns_by_column(num_columns);

  for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
    auto const& levels    = levels_by_column[col_idx];
    auto const num_levels = static_cast<size_type>(levels.size());
    auto& data_buffers    = data_buffers_by_column[col_idx];
    auto& null_masks      = null_masks_by_column[col_idx];
    auto& offsets_columns = offsets_columns_by_column[col_idx];

    data_buffers.reserve(num_levels);
    null_masks.reserve(num_levels);
    offsets_columns.resize(num_levels);

    for (size_type level_idx = 0; level_idx < num_levels; ++level_idx) {
      auto const& level = levels[level_idx];

      // Skip allocation for delegated levels (e.g., DICTIONARY)
      if (level.delegate_to_specialized) {
        data_buffers.emplace_back(0, stream, mr);
        null_masks.emplace_back(0, stream, mr);
        continue;
      }

      // Data buffer
      if (level.is_struct) {
        data_buffers.emplace_back(0, stream, mr);
      } else if (level.is_string) {
        data_buffers.emplace_back(level.total_child_size, stream, mr);
        auto const offsets_count   = level.total_rows + 1;
        offsets_columns[level_idx] = cudf::strings::detail::create_offsets_child_column(
          static_cast<int64_t>(level.total_child_size), offsets_count, stream, mr);
      } else if (level.is_list) {
        data_buffers.emplace_back(0, stream, mr);
        auto const offsets_count   = level.total_rows + 1;
        offsets_columns[level_idx] = cudf::make_numeric_column(
          data_type{type_id::INT32}, offsets_count, mask_state::UNALLOCATED, stream, mr);
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
  }

  // ========== PHASE 2: SINGLE batched memcpy for ALL data across ALL columns ==========
  // Calculate total copy operations across ALL columns
  size_t total_copy_ops = 0;
  for (auto const& levels : levels_by_column) {
    for (auto const& level : levels) {
      if (level.delegate_to_specialized) continue;
      if (!level.is_struct && !level.is_list && !level.src_data_ptrs.empty()) {
        total_copy_ops += level.src_data_ptrs.size();
      }
    }
  }

  if (total_copy_ops > 0) {
    std::vector<void const*> all_src_ptrs;
    std::vector<void*> all_dst_ptrs;
    std::vector<std::size_t> all_sizes;
    all_src_ptrs.reserve(total_copy_ops);
    all_dst_ptrs.reserve(total_copy_ops);
    all_sizes.reserve(total_copy_ops);

    // Aggregate ALL copy operations from ALL columns
    for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
      auto const& levels = levels_by_column[col_idx];
      auto& data_buffers = data_buffers_by_column[col_idx];

      for (size_type level_idx = 0; level_idx < static_cast<size_type>(levels.size());
           ++level_idx) {
        auto const& level = levels[level_idx];
        if (level.delegate_to_specialized) continue;
        if (level.is_struct || level.is_list || level.src_data_ptrs.empty()) continue;

        char* dst_ptr = static_cast<char*>(data_buffers[level_idx].data());
        for (size_t i = 0; i < level.src_data_ptrs.size(); ++i) {
          if (level.src_data_ptrs[i] != nullptr && level.data_sizes[i] > 0) {
            all_src_ptrs.push_back(level.src_data_ptrs[i]);
            all_dst_ptrs.push_back(dst_ptr);
            all_sizes.push_back(level.data_sizes[i]);
          }
          dst_ptr += level.data_sizes[i];
        }
      }
    }

    // SINGLE batched_memcpy_async call for ALL data
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
  }

  // ========== PHASE 3: Group offsets by total_rows, launch 2D kernels ==========

  // Step 3a: Group string/list levels by total_rows
  std::map<int64_t, offsets_kernel_group> offsets_groups;
  for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
    auto const& levels = levels_by_column[col_idx];
    for (size_type level_idx = 0; level_idx < static_cast<size_type>(levels.size()); ++level_idx) {
      auto const& level = levels[level_idx];
      if (!level.is_string && !level.is_list) continue;

      auto& group      = offsets_groups[level.total_rows];
      group.total_rows = level.total_rows;
      group.col_level_indices.emplace_back(col_idx, level_idx);
    }
  }

  // Step 3b: Build flattened arrays for each group
  for (auto& [total_rows, group] : offsets_groups) {
    group.col_src_ptr_offsets.push_back(0);
    group.col_input_offsets_offsets.push_back(0);

    for (auto [col_idx, level_idx] : group.col_level_indices) {
      auto const& level  = levels_by_column[col_idx][level_idx];
      auto const num_src = level.src_offsets_ptrs.size();

      // Append to flattened arrays
      group.all_src_offsets_ptrs.insert(group.all_src_offsets_ptrs.end(),
                                        level.src_offsets_ptrs.begin(),
                                        level.src_offsets_ptrs.end());
      group.all_src_offsets_offsets.insert(group.all_src_offsets_offsets.end(),
                                           level.src_offsets_offsets.begin(),
                                           level.src_offsets_offsets.end());
      group.all_first_src_offsets.insert(group.all_first_src_offsets.end(),
                                         level.first_src_offsets.begin(),
                                         level.first_src_offsets.end());

      // Build input_offsets (prefix sum of col_sizes)
      std::vector<size_t> input_offsets(num_src + 1);
      input_offsets[0] = 0;
      for (size_t i = 0; i < num_src; ++i) {
        input_offsets[i + 1] = input_offsets[i] + level.col_sizes[i];
      }
      group.all_input_offsets.insert(
        group.all_input_offsets.end(), input_offsets.begin(), input_offsets.end());

      // chars_partition_offsets already has num_src+1 elements
      group.all_chars_partition_offsets.insert(group.all_chars_partition_offsets.end(),
                                               level.chars_partition_offsets.begin(),
                                               level.chars_partition_offsets.end());

      // Per-column metadata
      group.col_src_ptr_offsets.push_back(
        static_cast<size_type>(group.all_src_offsets_ptrs.size()));
      group.col_input_offsets_offsets.push_back(
        static_cast<size_type>(group.all_input_offsets.size()));
      group.col_num_src_columns.push_back(static_cast<size_type>(num_src));

      auto output_view = offsets_columns_by_column[col_idx][level_idx]->mutable_view();
      group.col_output_data.push_back(offsetalator_factory::make_output_iterator(output_view));
      group.col_total_child_size.push_back(level.total_child_size);
      group.col_offsets_type_size.push_back(cudf::size_of(level.offsets_dtype));
    }
  }

  // Step 3c: Launch 2D kernel for each group
  for (auto const& [total_rows, group] : offsets_groups) {
    auto const num_cols_in_group = group.col_level_indices.size();

    // Upload all arrays to device
    auto d_all_src_offsets_ptrs = make_device_uvector_async(
      group.all_src_offsets_ptrs, stream, cudf::get_current_device_resource_ref());
    auto d_all_src_offsets_offsets = make_device_uvector_async(
      group.all_src_offsets_offsets, stream, cudf::get_current_device_resource_ref());
    auto d_all_input_offsets = make_device_uvector_async(
      group.all_input_offsets, stream, cudf::get_current_device_resource_ref());
    auto d_all_chars_partition_offsets = make_device_uvector_async(
      group.all_chars_partition_offsets, stream, cudf::get_current_device_resource_ref());
    auto d_all_first_src_offsets = make_device_uvector_async(
      group.all_first_src_offsets, stream, cudf::get_current_device_resource_ref());

    auto d_col_src_ptr_offsets = make_device_uvector_async(
      group.col_src_ptr_offsets, stream, cudf::get_current_device_resource_ref());
    auto d_col_input_offsets_offsets = make_device_uvector_async(
      group.col_input_offsets_offsets, stream, cudf::get_current_device_resource_ref());
    auto d_col_num_src_columns = make_device_uvector_async(
      group.col_num_src_columns, stream, cudf::get_current_device_resource_ref());
    auto d_col_output_data = make_device_uvector_async(
      group.col_output_data, stream, cudf::get_current_device_resource_ref());
    auto d_col_total_child_size = make_device_uvector_async(
      group.col_total_child_size, stream, cudf::get_current_device_resource_ref());
    auto d_col_offsets_type_size = make_device_uvector_async(
      group.col_offsets_type_size, stream, cudf::get_current_device_resource_ref());

    // Launch 2D kernel with occupancy-based grid sizing
    constexpr size_type block_size{256};
    auto const max_blocks     = max_active_blocks_offsets_kernel_2d();
    auto const num_sms        = cudf::detail::num_multiprocessors();
    auto const occupancy_grid = max_blocks * num_sms;
    auto const num_elements   = total_rows + 1;
    // Use smaller of: occupancy-based grid or element count (avoid over-launching)
    auto const num_blocks_x =
      std::min(occupancy_grid, cudf::util::div_rounding_up_safe(num_elements, block_size));
    dim3 grid(num_blocks_x, static_cast<unsigned int>(num_cols_in_group));

    batch_concatenate_offsets_kernel_2d<block_size>
      <<<grid, block_size, 0, stream.value()>>>(d_all_src_offsets_ptrs.data(),
                                                d_all_src_offsets_offsets.data(),
                                                d_all_input_offsets.data(),
                                                d_all_chars_partition_offsets.data(),
                                                d_all_first_src_offsets.data(),
                                                d_col_src_ptr_offsets.data(),
                                                d_col_input_offsets_offsets.data(),
                                                d_col_num_src_columns.data(),
                                                d_col_output_data.data(),
                                                d_col_total_child_size.data(),
                                                d_col_offsets_type_size.data(),
                                                total_rows);
  }

  // ========== PHASE 4: Batch mask concatenation for each column ==========
  std::vector<std::vector<size_type>> null_counts_by_column(num_columns);

  for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
    auto const& levels = levels_by_column[col_idx];
    auto& null_masks   = null_masks_by_column[col_idx];
    auto& null_counts  = null_counts_by_column[col_idx];

    null_counts.resize(levels.size(), 0);

    for (size_type level_idx = 0; level_idx < static_cast<size_type>(levels.size()); ++level_idx) {
      auto const& level = levels[level_idx];
      if (level.delegate_to_specialized) continue;
      if (!level.has_nulls) continue;

      auto const num_src_columns = level.mask_ptrs.size();
      auto const num_mask_bits   = level.total_rows;

      // Build output offsets (prefix sum of column sizes)
      std::vector<size_type> output_offsets(num_src_columns + 1);
      output_offsets[0] = 0;
      for (size_t i = 0; i < num_src_columns; ++i) {
        output_offsets[i + 1] = output_offsets[i] + level.col_sizes[i];
      }

      auto d_mask_ptrs =
        make_device_uvector_async(level.mask_ptrs, stream, cudf::get_current_device_resource_ref());
      auto d_mask_offsets = make_device_uvector_async(
        level.mask_offsets, stream, cudf::get_current_device_resource_ref());
      auto d_output_offsets =
        make_device_uvector_async(output_offsets, stream, cudf::get_current_device_resource_ref());

      constexpr size_type block_size{256};
      auto const num_output_words = cudf::util::div_rounding_up_safe(num_mask_bits, 32);
      cudf::detail::grid_1d config(num_output_words, block_size);

      batch_concatenate_masks_kernel<block_size>
        <<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
          d_mask_ptrs.data(),
          d_mask_offsets.data(),
          d_output_offsets.data(),
          static_cast<size_type>(num_src_columns),
          static_cast<bitmask_type*>(null_masks[level_idx].data()),
          num_output_words);

      null_counts[level_idx] = level.total_null_count;
    }
  }

  // Build mask results
  std::vector<std::vector<std::pair<rmm::device_buffer, size_type>>> mask_results_by_column(
    num_columns);
  for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
    auto& mask_results = mask_results_by_column[col_idx];
    auto& null_masks   = null_masks_by_column[col_idx];
    auto& null_counts  = null_counts_by_column[col_idx];

    mask_results.reserve(null_masks.size());
    for (size_t i = 0; i < null_masks.size(); ++i) {
      mask_results.emplace_back(std::move(null_masks[i]), null_counts[i]);
    }
  }

  return {std::move(data_buffers_by_column),
          std::move(mask_results_by_column),
          std::move(offsets_columns_by_column)};
}

#if CONCAT_SEPARATE_COLUMN
/**
 * @brief Concatenates tables by processing each column position separately.
 *
 * This implementation calls batch_concatenate(column_view) for each column position,
 * resulting in separate batched_memcpy calls per column. Used for comparison/benchmarking
 * against the unified single batched_memcpy implementation.
 */
std::unique_ptr<table> batch_concatenate_separate_columns(
  host_span<table_view const> tables_to_concat,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if (tables_to_concat.empty()) { return std::make_unique<table>(); }

  // Validate column counts match
  table_view const first_table = tables_to_concat.front();
  CUDF_EXPECTS(std::all_of(tables_to_concat.begin(),
                           tables_to_concat.end(),
                           [&first_table](auto const& t) {
                             return t.num_columns() == first_table.num_columns();
                           }),
               "Mismatch in table columns to concatenate.");

  auto const num_columns = first_table.num_columns();
  if (num_columns == 0) { return std::make_unique<table>(); }

  // Process each column position separately
  std::vector<std::unique_ptr<column>> output_columns;
  output_columns.reserve(num_columns);

  std::vector<column_view> cols_for_position;
  cols_for_position.reserve(tables_to_concat.size());

  for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
    cols_for_position.clear();
    for (auto const& table : tables_to_concat) {
      cols_for_position.push_back(table.column(col_idx));
    }

    // Call batch_concatenate for this column position (separate batched_memcpy per column)
    output_columns.push_back(
      detail::batch_concatenate(host_span<column_view const>{cols_for_position}, stream, mr));
  }

  return std::make_unique<table>(std::move(output_columns));
}
#endif

std::unique_ptr<table> batch_concatenate(host_span<table_view const> tables_to_concat,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

#if CONCAT_SEPARATE_COLUMN
  return batch_concatenate_separate_columns(tables_to_concat, stream, mr);
#else
  if (tables_to_concat.empty()) { return std::make_unique<table>(); }

  // Validate column counts match
  table_view const first_table = tables_to_concat.front();
  CUDF_EXPECTS(std::all_of(tables_to_concat.begin(),
                           tables_to_concat.end(),
                           [&first_table](auto const& t) {
                             return t.num_columns() == first_table.num_columns();
                           }),
               "Mismatch in table columns to concatenate.");

  auto const num_columns = first_table.num_columns();
  if (num_columns == 0) { return std::make_unique<table>(); }

  // Check that all columns in all tables are supported
  for (auto const& table : tables_to_concat) {
    for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
      CUDF_EXPECTS(is_column_batch_concat_supported(table.column(col_idx)),
                   "batch_concatenate only supports fixed-width, struct, string, list, and "
                   "dictionary types.",
                   cudf::logic_error);
    }
  }

  // Step 1: Collect level_info for each column position
  std::vector<std::vector<level_info>> levels_by_column(num_columns);
  std::vector<column_view> cols_for_position;
  cols_for_position.reserve(tables_to_concat.size());

  for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
    cols_for_position.clear();
    for (auto const& table : tables_to_concat) {
      cols_for_position.push_back(table.column(col_idx));
    }

    // Verify bounds and type compatibility
    batch_bounds_and_type_check(cols_for_position, stream);

    // Handle all-empty case for this column
    bool all_empty = std::all_of(cols_for_position.begin(),
                                 cols_for_position.end(),
                                 [](auto const& c) { return c.is_empty(); });

    if (!all_empty) {
      collect_level_info_recursive(cols_for_position, levels_by_column[col_idx], stream);
    }
  }

  // Step 2: Batch process all levels with SINGLE batched_memcpy
  auto result = batch_process_table_levels(levels_by_column, stream, mr);

  // Step 3: Reconstruct each column
  std::vector<std::unique_ptr<column>> output_columns;
  output_columns.reserve(num_columns);

  for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
    auto const& levels = levels_by_column[col_idx];

    // Handle all-empty column case
    if (levels.empty()) {
      // All columns at this position were empty - create empty_like
      output_columns.push_back(empty_like(first_table.column(col_idx)));
      continue;
    }

    size_type level_idx = 0;
    output_columns.push_back(reconstruct_column(levels,
                                                result.data_buffers_by_column[col_idx],
                                                result.mask_results_by_column[col_idx],
                                                result.offsets_columns_by_column[col_idx],
                                                level_idx,
                                                stream,
                                                mr));
  }

  return std::make_unique<table>(std::move(output_columns));
#endif
}

bool can_use_batch_concatenate(host_span<table_view const> tables)
{
  if (tables.empty()) return false;

  auto const num_columns = tables.front().num_columns();
  for (auto const& table : tables) {
    if (table.num_columns() != num_columns) return false;
    for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
      if (!is_column_batch_concat_supported(table.column(col_idx))) { return false; }
    }
  }
  return true;
}

}  // namespace detail

std::unique_ptr<column> batch_concatenate(host_span<column_view const> columns_to_concat,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::batch_concatenate(columns_to_concat, stream, mr);
}

std::unique_ptr<table> batch_concatenate(host_span<table_view const> tables_to_concat,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::batch_concatenate(tables_to_concat, stream, mr);
}

}  // namespace CUDF_EXPORT cudf
