/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "segmented_sort_keys.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace cudf {
namespace detail {

/**
 * @brief Average and total element-count bounds below which CUB `DeviceSegmentedSort` is preferred
 *
 * Chosen from benchmark results and shared by the fixed-width dispatch so its two fast paths agree
 * on which columns `DeviceSegmentedSort` claims: the packed-radix path takes exactly the complement
 * of this gate.
 */
constexpr size_type MAX_AVG_LIST_SIZE_FOR_FAST_SORT{100};
constexpr size_type MAX_LIST_SIZE_FOR_FAST_SORT{1 << 18};

/**
 * @brief Whether a single fixed-width column is small enough to prefer CUB `DeviceSegmentedSort`
 *
 * @param num_rows Element count of the column
 * @param num_offsets Segment-offsets count (the number of segments plus one), matching the
 *        historical average-size heuristic; the caller guarantees it is nonzero
 * @return true if CUB `DeviceSegmentedSort` is preferred for this shape
 */
inline bool prefer_cub_segmented_sort(size_type num_rows, size_type num_offsets)
{
  return (num_rows / num_offsets) < MAX_AVG_LIST_SIZE_FOR_FAST_SORT or
         num_rows < MAX_LIST_SIZE_FOR_FAST_SORT;
}

/**
 * @brief Fixed-width fast path a single key column takes within the explicit-(order, null_order) /
 * unstable envelope
 */
enum class fixed_width_sort_path {
  comparison,     ///< No fast path applies; fall through to the comparison sort
  tiered,         ///< Register / warp tiered kernel
  cub_segmented,  ///< CUB `DeviceSegmentedSort` over the packed rep
  packed_radix    ///< One global packed-radix sort
};

/**
 * @brief Routes a single fixed-width key column to the fast path measured best for its shape
 *
 * One segmented sort, four engines picked by type x null-presence x average list size, since each
 * engine's cost scales differently. Per path (band -> engine -> why it wins):
 * - Non-tiered numerics (narrow/unsigned ints, `bool`, `DECIMAL32`/`DECIMAL64`) -> packed radix if
 *   null-bearing or long, else CUB/comparison: the tiered key can't encode them, so keep main's
 * rule.
 * - Null-bearing tiered types -> tiered sort: validity folds into the key, no separate null pass.
 * - No-null past the fast cutoff -> packed radix: bandwidth-bound, cost ~ key bytes, amortized by
 * long lists.
 * - No-null `DECIMAL128` -> tiered when tiny, CUB in a sparse-large mid band, tiered above it:
 * CUB's 16-byte-pair merge tiles cap at 32 elements, so dense large segments would explode into a
 * radix.
 * - No-null floating point -> tiered across the short range.
 * - No-null 8-byte int/chrono -> tiered across the range: the tiered warp kernels beat CUB
 * `DeviceSegmentedSort` there; 4-byte int/chrono -> tiered: register sorting networks make
 * comparison cost width-independent for tiny segments.
 * - Outside the fixed-width fast envelope -> comparison sort.
 *
 * @param key The single key column to sort
 * @param num_rows Element count of the column
 * @param segment_offsets The segment offsets (segment count plus one); the caller guarantees
 * non-empty
 * @param stream CUDA stream used by the `DECIMAL128` mid-band shape gate, if reached
 * @return the fast path measured best for this column's shape (or `comparison` for none)
 */
fixed_width_sort_path choose_fixed_width_sort_path(column_view const& key,
                                                   size_type num_rows,
                                                   column_view const& segment_offsets,
                                                   rmm::cuda_stream_view stream);

/**
 * @brief Faster segmented sorted-order for a single fixed-width key column via one radix sort
 *
 * Replaces the comparison sort (used for null-bearing or large columns) and
 * cub::DeviceSegmentedSort (used for the small no-null case) with a single non-segmented radix sort
 * over a packed key that carries each element's segment, a null class bit, and its value
 * transformed so the unsigned key order equals the requested value order (`polarity` complements
 * the value field for descending and picks the null side). Because a fixed-width value is fully
 * encoded, there is no tie-break: the radix permutation is the final order. Nulls need no post-pass
 * -- the class bit places them on their configured side within their segment, and their order
 * among themselves is immaterial under the unstable contract. Engages for any explicit (order,
 * null_order), which the caller resolves into `polarity`; handles both nullable and non-nullable
 * columns.
 *
 * @param input The fixed-width column whose elements are sorted within each segment
 * @param segment_offsets Identifies the segments to sort within
 * @param polarity Key polarity resolved from the requested (order, null_order)
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Indices that map the column to a segmented sorted order
 */
[[nodiscard]] std::unique_ptr<column> fast_segmented_sorted_order_numeric_packed(
  column_view const& input,
  column_view const& segment_offsets,
  sort_polarity polarity,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
