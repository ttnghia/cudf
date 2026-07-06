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
 * @brief Faster segmented sorted-order for a single fixed-width key column via a tiered sort
 *
 * Classifies segments by size into three tiers whose cost tracks the segment size rather than the
 * key width -- the win over the packed-radix path when segments are tiny but values are wide: a
 * segment of
 * <= `TIERED_NETWORK_CAP` elements is sorted by one thread with a fixed Batcher network, a segment
 * of
 * <= `TIERED_WARP_CAP` by a full warp with `cub::WarpMergeSort`, and a rare radix-tier outlier by a
 * packed radix over just that segment's elements (scattered into place). Nulls are ordered on the
 * polarity's side within each segment by a three-valued class flag in the sort key, and a
 * descending order complements the key's value bits; the result is the same order the comparison
 * path produces. Engages for any explicit (order, null_order), which the caller resolves into
 * `polarity`; handles both nullable and non-nullable columns.
 *
 * @param input The fixed-width column whose elements are sorted within each segment
 * @param segment_offsets Identifies the segments to sort within
 * @param polarity Key polarity resolved from the requested (order, null_order)
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Indices that map the column to a segmented sorted order
 */
[[nodiscard]] std::unique_ptr<column> fast_segmented_sorted_order_tiered(
  column_view const& input,
  column_view const& segment_offsets,
  sort_polarity polarity,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

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

/**
 * @brief Faster segmented sorted-order for a single STRING key column via iterative radix sort
 *
 * Lexicographic string comparison is replaced by radix sorts over a compact key holding the
 * element's segment and the big-endian-packed leading bytes of its string. The first key is a
 * single `uint64` laid out `[segment : S bits][class : 1 bit when nullable][prefix : P bits]`
 * (see `packed_key_layout`), so one global `cub::DeviceRadixSort` over its full 64 bits -- eight
 * byte-passes, not the twelve the 96-bit iterative key needs -- orders elements by segment, then by
 * the requested null placement (the `polarity` class bit), then by their leading `P` prefix bits
 * (complemented under a descending sort). Distinct strings in one segment
 * sharing those bits remain tied, so instead of a single-threaded comparison tie-break, only the
 * still-tied elements are compacted out and re-sorted by successive eight-byte windows: each pass
 * radixes that compacted subset by a 96-bit key whose most-significant field is a dense run rank
 * (encoding the order resolved so far) and whose low field is the next eight bytes. The run rank
 * keeps every already-separated element fixed, so a pass only reorders within a tied run, and a
 * parallel radix replaces the serial per-run sort that made a fully-shared-prefix segment a
 * multi-second straggler. After each pass the elements that became singletons are frozen at their
 * final positions and dropped, so the subset shrinks and the loop exits as soon as nothing stays
 * tied -- a prefix that resolves in one window costs one pass, not the full cap. Whatever stays
 * tied after the pass cap is finished by one comparison cleanup, so correctness never needs it.
 *
 * Correctness equals the lexicographic comparison sort under the requested `polarity` exactly:
 * - The packed prefix is the top `P` bits of the same big-endian leading-byte value the windows
 *   pack, so an unsigned compare of the `uint64` reproduces unsigned-byte order over those bits.
 *   Because `P` is not necessarily a byte multiple, a packed-key match proves only `floor(P/8)`
 *   *whole* leading bytes equal; the windows and the comparison cleanup therefore both begin at
 *   `floor(P/8)`, and the iterative run rank is seeded from the packed keys (not a wider prefix) so
 *   two strings tied on `P` bits are left in one run rather than frozen apart on their lower bits.
 * - Each window packs bytes big-endian just as the first key does, so an unsigned comparison of the
 *   packed window reproduces unsigned-byte order over those bytes -- the same ordering the
 *   comparison path uses.
 * - A string with no byte left in a window packs to zero (the minimum), so a shorter string that is
 *   a prefix of a longer one sorts first, reproducing the comparison sort's shorter-is-less length
 *   tie-break. The residual case where a shorter string and a longer one agree on every byte and
 *   the longer one's tail is all zero bytes cannot be separated by any window; it is resolved by
 *   the final comparison cleanup, whose `compare_suffix` returns the length difference.
 * - The class bit sits just below the segment field -- above every prefix bit -- so a non-null
 *   string can never set it, and nulls collect on the polarity's requested side within their
 *   segment with no sentinel prefix needed (closing the all-0xFF-vs-null collision the sentinel
 *   scheme risked). A null is thus position-final after the first pass; the tie-break never counts
 *   it tied, so it keeps its first-pass slot, exactly as the comparison path places it.
 * - `polarity` folds the requested (order, null_order) into these keys: `element_class` sets the
 *   class bit so nulls land on the requested side, and a descending sort XOR-complements only the
 *   byte fields (the packed prefix and every window), reversing byte order while leaving the
 *   segment and null placement untouched. The complement sends an exhausted window's zero to the
 *   maximum, so the shorter-is-less rule and the cleanup's length tie-break both invert to
 *   shorter-is-greater -- the length-uniform exhausted-run drop stays valid under either order (it
 *   reads real lengths and equal-key runs, both complement-invariant). ASCENDING / nulls-after
 *   keeps every key bit-identical to the shipped configuration (`element_class` and the complement
 *   are both no-ops there).
 *
 * Unlike a rank-encoding approach, no sort of the distinct strings is performed, so the per-pass
 * cost does not grow with key cardinality. All four explicit (order, null_order) combinations
 * engage this path; a zero-null column is relaxed to nulls-last so its keys match the shipped
 * ascending configuration exactly.
 *
 * @param input The STRING column whose elements are sorted within each segment
 * @param segment_offsets Identifies the segments to sort within
 * @param polarity The requested sort direction and null placement, folded into the radix keys
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Indices that map the column to a segmented sorted order
 */
[[nodiscard]] std::unique_ptr<column> fast_segmented_sorted_order_strings_prefix(
  column_view const& input,
  column_view const& segment_offsets,
  sort_polarity polarity,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Whether every segment fits the graduated path's largest warp tile
 * (`STRINGS_GRAD_WARP_CAP`)
 *
 * One device count over the segment sizes plus its host synchronization, mirroring
 * `decimal128_cub_segment_shape_ok`; the caller runs the cheap scalar gate checks first so this
 * synchronizing probe is paid only for an otherwise-qualifying STRING column.
 */
bool strings_grad_all_segments_fit(column_view const& segment_offsets,
                                   rmm::cuda_stream_view stream);

/**
 * @brief Segmented sorted-order for a STRING column via graduated in-warp sorts
 *
 * One virtual warp sorts each segment with `cub::WarpMergeSort` under a string comparator; the warp
 * width follows the segment-size band (W8 for sizes <= 16, W16 for 17-32, W32 for 33-64, two items
 * per lane, except the (0,8] slice at one), so a tiny segment never occupies a full warp. The
 * bottom bands use the 8-byte comparator key and the upper bands the 16-byte prekey. Every band
 * launches over the full segment list and self-filters to its size slice; the bands partition [1,
 * 64], so every output slot is written exactly once (an empty segment has none). The caller
 * guarantees every segment holds at most `STRINGS_GRAD_WARP_CAP` elements
 * (`strings_grad_all_segments_fit`) and offsets spanning all rows. `polarity` folds the requested
 * order and null placement into the keys, matching `fast_segmented_sorted_order_strings_prefix`,
 * whose output contract this shares: indices mapping the column to its segmented sorted order.
 */
[[nodiscard]] std::unique_ptr<column> fast_segmented_sorted_order_strings_grad(
  column_view const& input,
  column_view const& segment_offsets,
  sort_polarity polarity,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
