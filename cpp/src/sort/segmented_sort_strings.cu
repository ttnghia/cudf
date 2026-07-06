/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "segmented_sort_fast.cuh"
#include "segmented_sort_keys.cuh"
#include "segmented_sort_warp_kernel.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/utility>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace cudf {
namespace detail {
namespace {

/**
 * @brief Number of leading string bytes packed per big-endian window (the `uint64` byte capacity)
 *
 * Each iterative window packs this many bytes, and `packed_key_builder` reads at most this many
 * leading bytes before keeping the prefix bits that fit. Declared as a plain `constexpr` (its value
 * is read in host code) -- the value remains usable from device code.
 */
constexpr size_type prefix_bytes = sizeof(cuda::std::uint64_t);

/**
 * @brief Bit layout of the single-`uint64` first-pass radix key
 *
 * The initial sort uses one `uint64`, not the wider iterative `prefix_key96`, so the radix touches
 * `end_bit/8 = 8` byte-passes, not twelve. The word is laid out most- to least-significant as
 * `[segment_id : S bits][is_null : 1 bit, only when the column has nulls][prefix : P bits]`, an
 * unsigned compare of which reproduces the `prefix_key96` ordering: segment first, then non-nulls
 * (is_null 0) before nulls (is_null 1), then the packed leading bytes. `S` is the bit width of the
 * largest possible segment label; `P` is whatever is left, holding the top `P` bits of the same
 * big-endian leading-byte value the windows pack. Placing `is_null` just below `segment_id` -- a
 * bit a non-null string's prefix can never reach -- means a non-null whose leading bytes are all
 * 0xFF can no longer collide with a null, so no null sentinel is needed in this key.
 *
 * `P` encodes only `floor(P/8)` *whole* known-equal bytes once the key matches; the partial
 * trailing bits prove nothing about the next byte. So strings sharing the packed key are ordered
 * afterward by the iterative byte windows and the comparison cleanup, both started at `floor(P/8)`
 * rather than a fixed eight -- the only offset that holds when `P` is not a byte multiple.
 */
struct packed_key_layout {
  int segment_bits;  // S: bit_width of the maximum segment label (host-computed once).
  int null_bits;     // 1 when the column has nulls (the is_null bit is present), else 0.
  int prefix_bits;   // P = 64 - S - null_bits: the leading-byte bits the key can still hold.
};

/**
 * @brief Computes the `uint64` key layout for `num_segment_labels` distinct segment-label values
 *
 * `num_segment_labels` is the largest segment-label value the key must distinguish plus one. The
 * caller labels each element with its dense segment ordinal (`0, 1, ...` over the segments), whose
 * maximum is `num_segments - 1`, so the bound passed here is `num_segments` rather than the row
 * count -- the ordinals are dense in the segment count, not offset-derived. `S = bit_width` of
 * `num_segment_labels` reserves enough high bits, and `P` takes the remainder after the optional
 * null bit; the tighter `S` (segments, not rows) widens `P`, packing more leading bytes per key.
 */
inline packed_key_layout make_packed_key_layout(size_type num_segment_labels, bool has_nulls)
{
  auto const max_label    = static_cast<cuda::std::uint64_t>(num_segment_labels);
  auto const segment_bits = cuda::std::bit_width(max_label);
  auto const null_bits    = has_nulls ? 1 : 0;
  return packed_key_layout{segment_bits, null_bits, 64 - segment_bits - null_bits};
}

/**
 * @brief Builds the per-element single-`uint64` first-pass key (see `packed_key_layout`)
 *
 * Packs the segment label into the top `S` bits, the polarity's class bit (when the column is
 * nullable) into the bit just below, and the top `P` bits of the element's big-endian leading-byte
 * value into the low `P` bits. The class bit is `polarity.element_class`, so nulls collect on the
 * requested side of every value in their segment; a null zeroes the prefix bits so all nulls in a
 * segment share one key (their order is immaterial and their bytes never read). A descending sort
 * complements only the `P` prefix bits, reversing their unsigned order without reaching the class
 * or segment fields -- so the same key drives every (order, null_order) combination.
 */
struct packed_key_builder {
  size_type const* d_segment_ids;
  column_device_view const d_strings;
  bool const has_nulls;
  packed_key_layout const layout;
  sort_polarity const polarity;
  __device__ cuda::std::uint64_t operator()(size_type idx) const
  {
    auto const segment_id   = static_cast<cuda::std::uint64_t>(d_segment_ids[idx]);
    auto const segment_part = segment_id << (64 - layout.segment_bits);
    if (has_nulls && d_strings.is_null(idx)) {
      return segment_part |
             (static_cast<cuda::std::uint64_t>(polarity.element_class(true)) << layout.prefix_bits);
    }
    auto const d_str                = d_strings.element<string_view>(idx);
    auto const bytes                = d_str.size_bytes();
    auto const* ptr                 = reinterpret_cast<unsigned char const*>(d_str.data());
    auto const packed_bytes         = cuda::std::min(bytes, size_type{prefix_bytes});
    cuda::std::uint64_t full_prefix = 0;
    for (size_type i = 0; i < packed_bytes; ++i) {
      full_prefix |= static_cast<cuda::std::uint64_t>(ptr[i]) << (56 - i * 8);
    }
    // Keep only the top P bits of the eight-byte big-endian prefix; the low bits do not fit.
    // Descending complements just those P bits (`value_mask64` narrowed to a P-bit mask), reversing
    // their order while staying below the class bit.
    auto const prefix = (full_prefix >> (64 - layout.prefix_bits)) ^
                        (polarity.value_mask64() >> (64 - layout.prefix_bits));
    // When `has_nulls` is false the layout reserves no class bit, so `prefix_bits` coincides with
    // the segment field's LSB and this class OR is provably zero: callers must resolve
    // `nulls_first == false` for a null-free column (the fast-path gates do), and
    // `element_class(false) == nulls_first`.
    return segment_part |
           (static_cast<cuda::std::uint64_t>(polarity.element_class(false)) << layout.prefix_bits) |
           prefix;
  }
};

/**
 * @brief Flags the first position of each maximal run of equal `uint64` keys (1 = head, else 0)
 *
 * The `uint64` analogue of `key_head_flag`: an inclusive sum of these flags yields the dense
 * one-based run rank that seeds the iterative window path, capturing exactly the order the first
 * `uint64` sort resolved (the `P` packed bits) -- never the unencoded low bits -- so two strings
 * tied on the packed key are left in one run for the windows to resolve rather than frozen apart.
 */
struct key_head_flag_packed {
  cuda::std::uint64_t const* d_keys;
  __device__ cuda::std::uint32_t operator()(size_type i) const
  {
    if (i == 0) { return 1u; }
    return d_keys[i - 1] != d_keys[i] ? 1u : 0u;
  }
};

/**
 * @brief Flags `uint64`-key positions equal to a neighbor (a run of two or more), never a null
 *
 * The `uint64` analogue of `key_tied_flag`, used once to size and gate the iterative tie-break:
 * only these positions still need reordering. A null element is position-final after the first pass
 * -- its class bit sorts it to the polarity's side within the segment -- and its order among nulls
 * is immaterial, so it is reported untied and left in its first-pass slot rather than dragged
 * through every window only to be skipped by the cleanup. `null_flag` is the class-bit value that
 * marks a null (`polarity.element_class(true)`: 1 nulls-last, 0 nulls-first); the bit sits at
 * `prefix_bits` and is present only when the column has nulls.
 */
struct key_tied_flag_packed {
  cuda::std::uint64_t const* d_keys;
  size_type const num_elements;
  bool const has_nulls;
  int const prefix_bits;
  cuda::std::uint32_t const null_flag = 1u;
  __device__ bool operator()(size_type i) const
  {
    auto const cur = d_keys[i];
    if (has_nulls && static_cast<cuda::std::uint32_t>((cur >> prefix_bits) & 1u) == null_flag) {
      return false;
    }
    auto const eq_prev = i > 0 && d_keys[i - 1] == cur;
    auto const eq_next = i + 1 < num_elements && d_keys[i + 1] == cur;
    return eq_prev || eq_next;
  }
};

/**
 * @brief Big-endian packs a string's eight bytes starting at `window_start` into a `uint64`
 *
 * Mirrors `packed_key_builder`'s packing (byte 0 of the window in the most-significant position,
 * trailing bytes zero-filled) but at an arbitrary `window_start`, so an unsigned comparison of the
 * result reproduces unsigned-byte lexicographic order over that eight-byte window. A string with no
 * bytes at or past `window_start` packs to zero -- the minimum -- so an exhausted (shorter) string
 * orders before any string still carrying bytes in the window, reproducing the shorter-is-less
 * length tie-break for that window. A descending sort complements the returned window at the call
 * site, sending that exhausted zero to the maximum so the shorter string instead orders last -- the
 * exact reverse. Null elements are never inspected here.
 */
__device__ inline cuda::std::uint64_t pack_window(string_view const& d_str, size_type window_start)
{
  auto const bytes = d_str.size_bytes();
  if (window_start >= bytes) { return 0; }
  auto const* ptr = reinterpret_cast<unsigned char const*>(d_str.data());
  // Value copy: ODR-using the prefix_bytes constexpr by reference is ill-formed in device code.
  auto const window_bytes    = cuda::std::min(bytes - window_start, size_type{prefix_bytes});
  cuda::std::uint64_t window = 0;
  for (size_type i = 0; i < window_bytes; ++i) {
    window |= static_cast<cuda::std::uint64_t>(ptr[window_start + i]) << (56 - i * 8);
  }
  return window;
}

/**
 * @brief Per-run string-length range: the minimum and maximum byte length over a run of tied keys
 *
 * Reduced across each run of equal keys to decide whether the run is byte-identical. A run that is
 * length-uniform (`min_len == max_len`) and fully covered by the windows compared so far holds only
 * copies of one string, so its order is already final; a mixed-length run is a zero-extension
 * family (a shorter string colliding with a longer one's real or zero-filled bytes) that the
 * comparison cleanup must still order shorter-first.
 */
struct len_minmax {
  size_type min_len;
  size_type max_len;
};

/// Combines two `len_minmax` ranges: min of the minima and max of the maxima. Associative and
/// commutative, as `reduce_by_key` requires.
struct len_minmax_combine {
  __device__ len_minmax operator()(len_minmax const& a, len_minmax const& b) const
  {
    return len_minmax{cuda::std::min(a.min_len, b.min_len), cuda::std::max(a.max_len, b.max_len)};
  }
};

/// Seeds each active element's byte length as a degenerate `[len, len]` range for the per-run
/// reduction. The active set never holds a null (the tie gate excludes them), so the child index
/// always names a real string.
struct string_length_minmax {
  size_type const* d_children;
  column_device_view const d_strings;
  __device__ len_minmax operator()(size_type i) const
  {
    auto const len = d_strings.element<string_view>(d_children[i]).size_bytes();
    return len_minmax{len, len};
  }
};

/**
 * @brief First-pass tie flag refined to drop byte-identical exhausted runs
 *
 * Extends `key_tied_flag_packed`: a position stays tied only if it is in a packed-key run of two or
 * more that is not already byte-identical. A run whose lengths are uniform (`min == max`) and no
 * greater than `known_equal_bytes` -- the whole bytes the packed key proves equal -- holds one
 * repeated string, so every copy is position-final at its stable first-pass slot and is excluded
 * from the tie set entirely, sparing the loop and its O(N) buffers. A mixed-length run stays: a
 * shorter string can share a longer one's packed key through the zero-fill, and only the cleanup
 * can order it.
 *
 * The drop is polarity-independent: length uniformity reads real byte lengths, not the (possibly
 * complemented) key, and a run is a set of equal keys under either order (the complement is a
 * bijection), so a uniform-length run within the covered bytes is byte-identical duplicates whose
 * shared slot is final under any order. The rule only chooses which runs skip the cleanup; the kept
 * mixed-length runs get the polarity-aware ordering there. `null_flag` matches
 * `key_tied_flag_packed`.
 */
struct keep_tied_first {
  cuda::std::uint64_t const* d_keys;
  size_type const num_elements;
  bool const has_nulls;
  int const prefix_bits;
  cuda::std::uint32_t const* d_run_ids;
  len_minmax const* d_run_minmax;
  size_type const known_equal_bytes;
  cuda::std::uint32_t const null_flag = 1u;
  __device__ bool operator()(size_type i) const
  {
    auto const cur = d_keys[i];
    if (has_nulls && static_cast<cuda::std::uint32_t>((cur >> prefix_bits) & 1u) == null_flag) {
      return false;
    }
    auto const eq_prev = i > 0 && d_keys[i - 1] == cur;
    auto const eq_next = i + 1 < num_elements && d_keys[i + 1] == cur;
    if (!(eq_prev || eq_next)) { return false; }
    auto const mm        = d_run_minmax[d_run_ids[i] - 1];
    auto const identical = mm.min_len == mm.max_len && mm.max_len <= known_equal_bytes;
    return !identical;
  }
};

/**
 * @brief Window-loop tie flag refined to drop byte-identical exhausted runs
 *
 * Extends `key_tied_flag`: an active position stays tied only if it is in a run of two or more that
 * is not yet byte-identical. `covered` is the leading bytes every element in a run agrees on after
 * this pass (the windows compared so far). A length-uniform run (`min == max`) whose common length
 * is no greater than `covered` holds only copies of one string -- equal window stream over its
 * whole length and equal length -- so its stable radix order is final and it is frozen like a
 * singleton. A mixed-length run is a zero-extension family kept for the length-aware comparison
 * cleanup. The drop is polarity-independent for the same reason as `keep_tied_first`:
 * uniform-length covered runs are byte-identical duplicates under either order. `null_flag` is the
 * class-bit value marking a null (`polarity.element_class(true)`).
 */
struct keep_active_window {
  prefix_key96 const* d_keys;
  size_type const num_elements;
  cuda::std::uint32_t const* d_run_ids;
  len_minmax const* d_run_minmax;
  size_type const covered;
  cuda::std::uint32_t const null_flag = 1u;
  __device__ bool operator()(size_type i) const
  {
    auto const cur = d_keys[i];
    if ((cur.seg_null & 1u) == null_flag) { return false; }
    auto const eq_prev = i > 0 && keys_equal(d_keys[i - 1], cur);
    auto const eq_next = i + 1 < num_elements && keys_equal(d_keys[i + 1], cur);
    if (!(eq_prev || eq_next)) { return false; }
    auto const mm        = d_run_minmax[d_run_ids[i] - 1];
    auto const identical = mm.min_len == mm.max_len && mm.max_len <= covered;
    return !identical;
  }
};

/**
 * @brief Builds the next-pass radix key for the element currently at a sorted position
 *
 * The key reuses the `prefix_key96` layout: `seg_null` packs the run rank in its high 31 bits (the
 * dominant field, so radixing by it preserves all order resolved by prior passes) and the
 * polarity's class bit in bit 0, and the two window words hold the next eight-byte string window.
 * Two elements sharing a run rank are still tied; the window refines their order without disturbing
 * any other run. Null elements are isolated by the run rank (the class bit rides it), so the rank
 * alone keeps them grouped on the polarity's side; they carry the class bit and an unread window,
 * so the comparison cleanup recognizes and skips the run exactly as it does for the first pass. A
 * descending sort complements the window (`value_mask64`), reversing byte order and sending an
 * exhausted zero window to the maximum so a shorter string sorts after the longer ones sharing its
 * prefix; the run rank and class bit stay ascending.
 */
struct window_key_builder {
  cuda::std::uint32_t const* d_run_ids;
  size_type const* d_sorted_indices;
  column_device_view const d_strings;
  bool const has_nulls;
  size_type const window_start;
  sort_polarity const polarity;
  __device__ prefix_key96 operator()(size_type i) const
  {
    auto const run_id = d_run_ids[i];
    auto const idx    = d_sorted_indices[i];
    if (has_nulls && d_strings.is_null(idx)) {
      return prefix_key96{pack_seg_null(run_id, polarity.element_class(true)), 0u, 0u};
    }
    prefix_key96 key{pack_seg_null(run_id, polarity.element_class(false)), 0u, 0u};
    split_prefix(
      pack_window(d_strings.element<string_view>(idx), window_start) ^ polarity.value_mask64(),
      key.prefix_hi,
      key.prefix_lo);
    return key;
  }
};

/**
 * @brief Seeds the iterative window path's first-pass key from the single-`uint64` first sort
 *
 * The window path keys on `prefix_key96` and recomputes the run rank from its `cur_keys` at the top
 * of every pass, so its seed must be a `prefix_key96` whose run boundaries match exactly what the
 * first `uint64` sort resolved -- the `P` packed bits -- and nothing finer. This emits a key whose
 * `seg_null` packs the dense rank via `pack_seg_null` (reproducing the packed boundaries when
 * head-flagged) and whose window words are zero (adding no spurious boundary). The null flag read
 * from the packed key is carried for consistency, though the count-first tie set already excludes
 * nulls. The first window is built fresh by the pass itself; carrying a window here instead would
 * let two strings tied on the packed key but differing within that window be split into distinct
 * ranks, freezing an arbitrary order the windows must still be free to refine.
 */
struct tied_run_seed_builder {
  cuda::std::uint32_t const* d_run_ids;
  cuda::std::uint64_t const* d_packed_keys;
  int const prefix_bits;
  bool const has_nulls;
  __device__ prefix_key96 operator()(size_type i) const
  {
    auto const is_null =
      has_nulls ? static_cast<cuda::std::uint32_t>((d_packed_keys[i] >> prefix_bits) & 1u) : 0u;
    return prefix_key96{pack_seg_null(d_run_ids[i], is_null), 0u, 0u};
  }
};

/**
 * @brief Unsigned-byte comparison of two strings starting at a shared byte offset
 *
 * Replicates `string_view::compare` while skipping the first `offset` bytes, which the caller has
 * already proven equal. Each string's compared region runs from `offset` to its end (empty when the
 * string is no longer than `offset`); the bytes are compared unsigned. On an all-equal common
 * region the result is the full-length difference, which -- because the skipped prefixes are equal
 * -- carries the same sign as `string_view::compare`'s own length tie-break, including when one or
 * both strings are no longer than `offset` (e.g. distinguishing a string from itself plus a
 * trailing embedded null). This reproduces `string_view::compare` exactly for any pair whose first
 * `offset` bytes match, since that comparison would consume those equal bytes with no effect before
 * reaching the same suffixes.
 */
__device__ inline int compare_suffix(string_view const& a, string_view const& b, size_type offset)
{
  auto const a_len = cuda::std::max(0, a.size_bytes() - offset);
  auto const b_len = cuda::std::max(0, b.size_bytes() - offset);
  auto const* pa =
    reinterpret_cast<unsigned char const*>(a.data()) + cuda::std::min(offset, a.size_bytes());
  auto const* pb =
    reinterpret_cast<unsigned char const*>(b.data()) + cuda::std::min(offset, b.size_bytes());
  auto const common = cuda::std::min(a_len, b_len);
  for (size_type i = 0; i < common; ++i) {
    if (pa[i] != pb[i]) { return static_cast<int>(pa[i]) - static_cast<int>(pb[i]); }
  }
  return a.size_bytes() - b.size_bytes();
}

/**
 * @brief Run length at or above which a prefix-tie run is sorted by heapsort instead of insertion
 *
 * Insertion sort is optimal for the tiny runs typical of high-cardinality data, but it is O(run^2):
 * a fully-shared-prefix segment collapses to one run spanning the whole segment, which on the
 * generator's outlier rows of thousands of elements turns the tie-break into a multi-second
 * straggler. Switching to O(run*log run) heapsort once a run reaches this size caps that worst case
 * while leaving the common tiny-run path untouched.
 */
constexpr size_type TIE_HEAPSORT_THRESHOLD = 32;

/**
 * @brief Maximum number of iterative eight-byte radix windows after the initial prefix pass
 *
 * Each iterative pass resolves another eight bytes of strings still tied on every byte ordered so
 * far, so the loop terminates once every surviving run is length-uniform and exhausted (its bytes
 * fully consumed with no length difference left) or once all runs are singletons. This cap bounds
 * the worst case -- strings sharing an arbitrarily
 * long leading run (e.g. thousands agreeing on a multi-kilobyte prefix) would otherwise demand one
 * pass per eight shared bytes. Eight windows clears 64 bytes past the initial prefix; whatever
 * remains tied is finished by a single comparison cleanup, so correctness never depends on the cap,
 * only the pass count does.
 */
constexpr size_type MAX_RADIX_PASSES = 8;

/**
 * @brief Reorders prefix-tied runs by suffix comparison to match the comparison sort exactly
 *
 * The prefix orders only the leading bytes, so distinct strings in one segment sharing those bytes
 * carry an equal `prefix_key96` that the radix sort cannot separate. After the radix sort, each
 * maximal run of consecutive equal keys is owned by its first position, which sorts the run's child
 * indices by comparing the bytes from `tie_offset` onward. Genuine duplicates compare equal so
 * their order is immaterial, and an all-null run carries the null flag in `seg_null` and is left
 * untouched (its elements are null and never compared). The comparison uses unsigned-byte ordering,
 * matching the lexicographic comparison path, so the result is identical. `tie_offset` is the width
 * of the shared prefix the run agrees on, so the comparison skips those bytes and orders by the
 * remainder.
 *
 * Tiny runs (the common case on high-cardinality data) use an insertion sort; runs reaching
 * `TIE_HEAPSORT_THRESHOLD` use a heapsort instead, so a fully-shared-prefix run spanning a whole
 * segment cannot drive the single-threaded tie-break quadratic. Both sort in the requested
 * direction under the same suffix comparator and yield the same order; the path is unstable, so the
 * heapsort's reordering of equal elements is immaterial.
 *
 * Direction invariant: the cleanup must reproduce the order the window keys were driving toward for
 * the runs it refines. Ascending takes `compare_suffix` directly; descending inverts its sign,
 * which reverses both the byte comparison and the length tie-break -- matching the descending
 * window keys (whose exhausted-zero-to-maximum complement already ordered a shorter string last).
 * `null_flag` is the class-bit value marking a null run (`polarity.element_class(true)`); such runs
 * hold only nulls, are never compared, and are left in place -- and none reach here, the tie gate
 * excludes them.
 */
struct prefix_tie_breaker {
  prefix_key96 const* d_keys;
  size_type* d_indices;
  column_device_view const d_strings;
  size_type const num_elements;
  size_type const tie_offset;
  bool const descending               = false;
  cuda::std::uint32_t const null_flag = 1u;

  // True when the string referenced by index `a` orders strictly before the one referenced by `b`
  // in the requested direction -- the "less than" the run is sorted by. Descending inverts the
  // suffix comparison's sign (see the direction invariant above).
  __device__ bool index_less(size_type a, size_type b) const
  {
    auto const cmp = compare_suffix(
      d_strings.element<string_view>(a), d_strings.element<string_view>(b), tie_offset);
    return descending ? cmp > 0 : cmp < 0;
  }

  // Sift the element at local position `root` down a max-heap of size `n` over the run starting at
  // `base`, restoring the heap property below `root`.
  __device__ void sift_down(size_type* base, int64_t root, int64_t n) const
  {
    while (true) {
      auto const left = 2 * root + 1;
      if (left >= n) { break; }
      auto largest     = left;
      auto const right = left + 1;
      if (right < n && index_less(base[left], base[right])) { largest = right; }
      if (!index_less(base[root], base[largest])) { break; }
      auto const tmp = base[root];
      base[root]     = base[largest];
      base[largest]  = tmp;
      root           = largest;
    }
  }

  __device__ void operator()(size_type i) const
  {
    // Only the first position of a run does the work; runs are disjoint so writes never overlap.
    auto const key = d_keys[i];
    if (i > 0 && keys_equal(d_keys[i - 1], key)) { return; }
    // A null-classed run is entirely null elements, whose order is immaterial and whose string data
    // must not be read. The class bit is bit 0 of `seg_null`; `null_flag` is its null value.
    if ((key.seg_null & 1u) == null_flag) { return; }

    auto run_end = i + 1;
    while (run_end < num_elements && keys_equal(d_keys[run_end], key)) {
      ++run_end;
    }
    auto const run_len = run_end - i;
    if (run_len < 2) { return; }

    // Bytes before `tie_offset` are skipped since the run agrees on them.
    if (run_len < TIE_HEAPSORT_THRESHOLD) {
      // Insertion sort: optimal for the tiny runs typical of high-cardinality data, where ties are
      // rare.
      for (auto j = i + 1; j < run_end; ++j) {
        auto const idx_j = d_indices[j];
        auto const str_j = d_strings.element<string_view>(idx_j);
        auto k           = j;
        while (k > i) {
          auto const cmp =
            compare_suffix(d_strings.element<string_view>(d_indices[k - 1]), str_j, tie_offset);
          // Shift a neighbor that must follow `str_j`: ascending shifts a greater neighbor,
          // descending a smaller one.
          if (!(descending ? cmp < 0 : cmp > 0)) { break; }
          d_indices[k] = d_indices[k - 1];
          --k;
        }
        d_indices[k] = idx_j;
      }
    } else {
      // Heapsort: O(run*log run) and O(1) extra memory, capping the worst case when a shared prefix
      // makes one run span the whole segment. Build a max-heap, then repeatedly move the maximum to
      // the shrinking tail, which leaves the run ascending.
      auto* const base = d_indices + i;
      for (auto start = run_len / 2; start > 0; --start) {
        sift_down(base, start - 1, run_len);
      }
      for (auto heap_size = run_len; heap_size > 1; --heap_size) {
        auto const tmp      = base[0];
        base[0]             = base[heap_size - 1];
        base[heap_size - 1] = tmp;
        sift_down(base, 0, heap_size - 1);
      }
    }
  }
};

// ==========================================================================================
// Graduated-warp per-segment sort for a single STRING key column.
//
// When every segment fits the largest warp tile (`STRINGS_GRAD_WARP_CAP`), sorting each segment in
// a virtual warp with `cub::WarpMergeSort` beats the global multi-pass prefix-radix machine. The
// warp width graduates with the segment-size band (W8 for sizes <= 16, W16 for 17-32, W32 for
// 33-64), so a tiny segment never occupies a full warp. The key style is mixed per band: the 8-byte
// comparator key drives the bottom bands, where the 16-byte prekey's merge-exchange volume
// dominates over mostly-pad tiles, and the prekey drives W16/W32, where its packed prefix window
// amortizes. The (0,8] slice sorts at one item per lane (an 8-slot tile), halving the pad traffic
// the two-item shape pays at the smallest sizes. The requested (order, null_order) rides the same
// `sort_polarity` the prefix path uses; a column with any segment above the cap falls through to
// the prefix path.
// ==========================================================================================

/// Largest segment size the graduated-warp string path admits: the W32 x 2 register tile.
constexpr size_type STRINGS_GRAD_WARP_CAP = 64;

/**
 * @brief Ordering key for the comparator-key bands: the element's global index plus its class
 *
 * The comparator reads the string bytes through `gidx`, so the key itself stays eight bytes. `cls`
 * is a `tiered_element_class` whose ordinal comes from `polarity.element_class`, so nulls collect
 * on the requested side of a segment's valid elements and the pad slots past a segment's real
 * elements settle beyond them -- pad must rank strictly above either class for the same reason the
 * fixed-width tiers require it. `gidx` is dereferenced only for the valid class, so a null or pad
 * key never reads string data.
 */
struct strings_grad_cmp_key {
  size_type gidx;
  cuda::std::uint32_t cls;
};
static_assert(sizeof(strings_grad_cmp_key) == 8, "strings_grad_cmp_key must stay eight bytes");

/// Builds the comparator key for one element: its global index and polarity-resolved class.
struct strings_grad_cmp_key_builder {
  column_device_view d_strings;
  bool has_nulls;
  sort_polarity polarity;
  __device__ strings_grad_cmp_key operator()(size_type idx) const
  {
    return strings_grad_cmp_key{idx, polarity.element_class(has_nulls && d_strings.is_null(idx))};
  }
};

/**
 * @brief Strict-weak less for the comparator key: class first, then the full byte comparison
 *
 * Two keys in the same non-valid class (both null or both pad) compare equivalent -- the path is
 * unstable, their order is immaterial, and their bytes are never read. Two valid keys order by
 * `string_view::compare`, the unsigned-byte-then-shorter-first ordering the shipped prefix path
 * reproduces. The valid class ordinal follows the polarity (`element_class(false)` is 1
 * nulls-first, 0 otherwise), so a hardcoded constant would misfire under nulls-first; a descending
 * sort inverts the comparison's sign, reversing both byte order and the length tie-break as
 * `prefix_tie_breaker::index_less` does.
 */
struct strings_grad_cmp_less {
  column_device_view d_strings;
  sort_polarity polarity;
  __device__ bool operator()(strings_grad_cmp_key const& a, strings_grad_cmp_key const& b) const
  {
    if (a.cls != b.cls) { return a.cls < b.cls; }
    if (a.cls != polarity.element_class(false)) { return false; }
    auto const cmp =
      d_strings.element<string_view>(a.gidx).compare(d_strings.element<string_view>(b.gidx));
    return polarity.descending ? cmp > 0 : cmp < 0;
  }
};

/**
 * @brief Ordering key for the prekey bands: a packed leading-byte prefix over the index
 *
 * `prefix` is the element's first eight bytes packed big-endian with a zero-filled tail
 * (`pack_window` at offset zero), so an unsigned compare reproduces unsigned-byte order over the
 * window under the shipped shorter-is-less zero-fill convention; a descending sort complements the
 * window at build time so that order reverses. A prefix tie leaves only two cases -- both strings
 * genuinely share their first eight bytes, or a shorter string's zero-fill collided with the longer
 * one's real zero bytes (a zero-extension family such as `S` vs `S + "\0"`) -- and `compare_suffix`
 * from byte eight orders both exactly: its byte walk covers the first case and its full-length
 * difference the second, shorter-first.
 */
struct strings_grad_prekey {
  cuda::std::uint64_t prefix;
  size_type gidx;
  cuda::std::uint32_t cls;
};
static_assert(sizeof(strings_grad_prekey) == 16 and alignof(strings_grad_prekey) == 8,
              "strings_grad_prekey must stay sixteen bytes with eight-byte alignment");

/// Builds the prekey: packed eight-byte big-endian prefix (complemented when descending), global
/// index, polarity-resolved class. A null carries a zero prefix and is never inspected past the
/// class compare.
struct strings_grad_prekey_builder {
  column_device_view d_strings;
  bool has_nulls;
  sort_polarity polarity;
  __device__ strings_grad_prekey operator()(size_type idx) const
  {
    if (has_nulls && d_strings.is_null(idx)) {
      return strings_grad_prekey{0, idx, polarity.element_class(true)};
    }
    return strings_grad_prekey{
      pack_window(d_strings.element<string_view>(idx), 0) ^ polarity.value_mask64(),
      idx,
      polarity.element_class(false)};
  }
};

/**
 * @brief Strict-weak less for the prekey: class, then packed prefix, bytes only on a tie
 *
 * The packed prefix resolves most pairs in registers; it was complemented at build time under a
 * descending sort, so a plain unsigned compare of it is correct for both directions. The byte
 * comparison runs only when two valid keys tie on the prefix, starting at `prefix_bytes` since the
 * tie proves the window equal, and a descending sort inverts its sign as
 * `prefix_tie_breaker::index_less` does. The valid class follows the polarity, as in the comparator
 * key.
 */
struct strings_grad_prekey_less {
  column_device_view d_strings;
  sort_polarity polarity;
  __device__ bool operator()(strings_grad_prekey const& a, strings_grad_prekey const& b) const
  {
    if (a.cls != b.cls) { return a.cls < b.cls; }
    if (a.cls != polarity.element_class(false)) { return false; }
    if (a.prefix != b.prefix) { return a.prefix < b.prefix; }
    auto const cmp = compare_suffix(d_strings.element<string_view>(a.gidx),
                                    d_strings.element<string_view>(b.gidx),
                                    size_type{prefix_bytes});
    return polarity.descending ? cmp > 0 : cmp < 0;
  }
};

/// Launches one graduated string band: virtual warps of `W` lanes sort the segments whose size is
/// in
/// `(band_lo, band_hi]` with `cub::WarpMergeSort`, self-filtering over the full segment list
/// exactly as the fixed-width warp bands do.
template <int W, int IPT, typename KeyT, typename KeyBuilder, typename CompareOp>
void launch_strings_grad_band(KeyBuilder const& build_key,
                              CompareOp const& compare_op,
                              KeyT const pad_key,
                              size_type const* d_offsets,
                              size_type const* d_seg_list,
                              size_type num_segments,
                              size_type band_lo,
                              size_type band_hi,
                              size_type* d_out,
                              rmm::cuda_stream_view stream)
{
  auto const grid =
    cudf::detail::grid_1d(static_cast<thread_index_type>(num_segments) * W, TIERED_BLOCK_THREADS);
  tiered_warp_band_kernel<KeyT, KeyBuilder, W, IPT, TIERED_BLOCK_THREADS, CompareOp>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      d_offsets, d_seg_list, num_segments, band_lo, band_hi, build_key, d_out, compare_op, pad_key);
  CUDF_CHECK_CUDA(stream.value());
}

}  // namespace

[[nodiscard]] std::unique_ptr<column> fast_segmented_sorted_order_strings_prefix(
  column_view const& input,
  column_view const& segment_offsets,
  sort_polarity polarity,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_elements = input.size();
  auto const num_segments = segment_offsets.size() - 1;

  // The packed key reserves a class bit only when nullable (see `packed_key_builder`); guard the
  // caller-side coupling so a null-free column can never arrive with nulls_first set.
  CUDF_EXPECTS(input.has_nulls() or not polarity.nulls_first,
               "nulls_first requires a nullable column for the packed key layout");

  // Label every element with its dense segment ordinal (`0, 1, ...` across the segments) so a
  // single global radix sort orders within each segment. Unlike the offset-derived labels the
  // shared `get_segment_indices` produces (whose values range up to the row count), these ordinals
  // range only over the segment count, so the packed key's segment field needs just
  // `bit_width(num_segments)` bits -- freeing the rest for a wider prefix. The ordinals are
  // monotonic with the segments (empty segments contribute no element and simply skip a label), so
  // cross-segment order is preserved exactly. The strings path's offsets are normalized to span
  // `[0, num_elements]`, so the label output size is `num_elements`.
  rmm::device_uvector<size_type> segment_ids(num_elements, stream);
  label_segments(segment_offsets.begin<size_type>(),
                 segment_offsets.end<size_type>(),
                 segment_ids.begin(),
                 segment_ids.end(),
                 stream);

  auto const d_input   = column_device_view::create(input, stream);
  auto const has_nulls = input.has_nulls();

  // The single-uint64 first-pass key layout. The segment labels are dense ordinals bounded by the
  // segment count, so the high field is sized for the largest possible ordinal (num_segments - 1)
  // via num_segments; the leftover bits below the optional null bit hold the prefix -- wider here
  // than under a row-count bound, so the key distinguishes more leading bytes before a tie.
  auto const layout = make_packed_key_layout(num_segments, has_nulls);
  // Whole leading bytes the packed key proves equal once two keys match. Only floor(P/8) full bytes
  // are guaranteed -- the partial trailing bits of P say nothing about the next byte -- so the
  // windows and the comparison cleanup must begin here rather than at a fixed eight.
  auto const known_equal_bytes = layout.prefix_bits / 8;

  auto const counting = cuda::counting_iterator<size_type>{0};

  // First-pass outputs. They outlive the packed-key inputs, which are scoped to the sort below and
  // freed before the tie loop grows its working set.
  rmm::device_uvector<cuda::std::uint64_t> keys_out(num_elements, stream);
  auto sorted_indices = cudf::make_numeric_column(
    data_type{type_to_id<size_type>()}, num_elements, mask_state::UNALLOCATED, stream, mr);
  auto const d_indices_out = sorted_indices->mutable_view().begin<size_type>();

  // The iterative window path keys on the 96-bit `prefix_key96` {run rank, eight-byte window},
  // decomposed into its fields; `key_bits` is that key's width. `seg_null` packs the run rank in
  // bits 1..31 and a null flag in bit 0; the run rank comes from a scan over run heads and is
  // bounded by the element count, itself a `size_type`, so `(rank << 1) | flag` fits the 32-bit
  // field for every `size_type` value -- proven once here rather than guarded per element.
  static_assert(2ull * cuda::std::numeric_limits<size_type>::max() + 1ull <=
                  cuda::std::numeric_limits<cuda::std::uint32_t>::max(),
                "size_type run rank does not fit prefix_key96::seg_null after the null-flag shift");
  auto const decomposer   = prefix_decomposer{};
  auto constexpr key_bits = static_cast<int>(
    (sizeof(cuda::std::uint32_t) + sizeof(cuda::std::uint32_t) + sizeof(cuda::std::uint32_t)) * 8);

  // First pass: one global radix sort of the single-uint64 key, carrying child indices as the
  // paired values. The unsigned-key compare sorts by segment, then non-nulls before nulls, then by
  // the packed leading bytes -- the same order the iterative `prefix_key96` produces. The whole
  // word is significant (segment in the high bits, prefix filling the low bits with no zero
  // padding), so the range is the full [0, 64): eight byte-passes, not the twelve the 96-bit
  // iterative key would need. The inputs and the two-stage temporary storage are all scoped here,
  // so they are released before the tie loop.
  {
    rmm::device_uvector<cuda::std::uint64_t> keys_in(num_elements, stream);
    rmm::device_uvector<size_type> indices_in(num_elements, stream);
    thrust::transform(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      counting,
      counting + num_elements,
      keys_in.begin(),
      packed_key_builder{segment_ids.data(), *d_input, has_nulls, layout, polarity});
    thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     indices_in.begin(),
                     indices_in.end(),
                     0);
    rmm::device_buffer d_temp_storage;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                    temp_storage_bytes,
                                    keys_in.data(),
                                    keys_out.data(),
                                    indices_in.data(),
                                    d_indices_out,
                                    num_elements,
                                    0,
                                    static_cast<int>(sizeof(cuda::std::uint64_t) * 8),
                                    stream.value());
    d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
    cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                    temp_storage_bytes,
                                    keys_in.data(),
                                    keys_out.data(),
                                    indices_in.data(),
                                    d_indices_out,
                                    num_elements,
                                    0,
                                    static_cast<int>(sizeof(cuda::std::uint64_t) * 8),
                                    stream.value());
  }

  // `segment_ids` fed only the first-pass key build above; free it (stream-ordered) before the tie
  // path grows its working set, mirroring the `keys_out` release after the seed block below.
  segment_ids = rmm::device_uvector<size_type>{0, stream};

  // Count-first zero-tie gate. A single O(N) pass over the sorted first-pass keys reports how many
  // elements are still tied (a run of two or more sharing the packed key, nulls excluded). Reading
  // that count is the one host synchronization the tie-break adds -- the same D->H read the old
  // flag-then-compact ordering already paid. On high-cardinality data nothing is tied, so it
  // returns zero and allocates no tie buffer, runs no compaction, and skips the loop and cleanup:
  // the common path is the first pass plus this single count.
  auto const tied_pred = key_tied_flag_packed{
    keys_out.data(), num_elements, has_nulls, layout.prefix_bits, polarity.element_class(true)};
  auto const any_tied =
    thrust::count_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     counting,
                     counting + num_elements,
                     tied_pred) > 0;

  if (any_tied) {
    // First-key length-uniform-exhausted exclusion. `tied_flags` starts from the raw packed-key
    // ties, then drops any run that is already byte-identical: a packed-key run whose lengths are
    // uniform and no greater than `known_equal_bytes` holds one repeated string, so every copy is
    // position-final at its stable first-pass slot and never needs the loop or its O(N) buffers.
    // The per-run length range is reduced only here, on the tie path -- the zero-tie gate above
    // already short-circuited the common case, and the N-wide range temporaries are freed before
    // the loop. A mixed-length run stays: a shorter string can share a longer one's packed key
    // through the zero-fill, so only the comparison cleanup can order it.
    rmm::device_uvector<bool> tied_flags(num_elements, stream);
    {
      rmm::device_uvector<cuda::std::uint32_t> first_run_ids(num_elements, stream);
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        counting,
                        counting + num_elements,
                        first_run_ids.begin(),
                        key_head_flag_packed{keys_out.data()});
      {
        rmm::device_buffer d_temp_storage;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage.data(),
                                      temp_storage_bytes,
                                      first_run_ids.data(),
                                      first_run_ids.data(),
                                      num_elements,
                                      stream.value());
        d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
        cub::DeviceScan::InclusiveSum(d_temp_storage.data(),
                                      temp_storage_bytes,
                                      first_run_ids.data(),
                                      first_run_ids.data(),
                                      num_elements,
                                      stream.value());
      }
      rmm::device_uvector<len_minmax> first_elem_minmax(num_elements, stream);
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        counting,
                        counting + num_elements,
                        first_elem_minmax.begin(),
                        string_length_minmax{d_indices_out, *d_input});
      rmm::device_uvector<len_minmax> first_run_minmax(num_elements, stream);
      thrust::reduce_by_key(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        first_run_ids.begin(),
        first_run_ids.end(),
        first_elem_minmax.begin(),
        cuda::make_discard_iterator(),
        first_run_minmax.begin(),
        cuda::std::equal_to<cuda::std::uint32_t>{},
        len_minmax_combine{});
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        counting,
                        counting + num_elements,
                        tied_flags.begin(),
                        keep_tied_first{keys_out.data(),
                                        num_elements,
                                        has_nulls,
                                        layout.prefix_bits,
                                        first_run_ids.data(),
                                        first_run_minmax.data(),
                                        known_equal_bytes,
                                        polarity.element_class(true)});
    }

    // Compact the sorted order down to just the still-tied positions; resolved singletons, untied
    // segments, byte-identical runs, and nulls are already in their final place and excluded, so
    // the windows and cleanup re-sort only this shrinking subset rather than the whole column.
    // `comp_pos` records each tied element's output position so the refined order can be scattered
    // straight back. The compacted outputs are sized to the refined tie count; `tied_flags` stays
    // N-sized because the selection reads a flag per input element.
    auto const num_tied = static_cast<size_type>(
      thrust::count(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    tied_flags.begin(),
                    tied_flags.end(),
                    true));

    // `tied_keys_packed` holds the still-tied elements' packed keys -- it feeds only the seed run
    // rank below (capturing exactly the order the first sort resolved); the iterative loop then
    // keys on the 96-bit `prefix_key96` it builds in `tied_keys_a`.
    rmm::device_uvector<size_type> comp_pos(num_tied, stream);
    rmm::device_uvector<size_type> child_a(num_tied, stream);
    rmm::device_uvector<cuda::std::uint64_t> tied_keys_packed(num_tied, stream);
    rmm::device_uvector<prefix_key96> tied_keys_a(num_tied, stream);
    cudf::detail::device_scalar<size_type> d_num_tied(stream);

    // Each flagged compaction gathers the tied positions, their child indices, or their packed
    // keys. The count is already known, so `d_num_tied` here only satisfies the API; it is re-read
    // per pass inside the loop where the surviving count changes.
    auto const select_flagged = [&](auto d_in, auto* d_out) {
      rmm::device_buffer d_temp_storage;
      size_t temp_storage_bytes = 0;
      cub::DeviceSelect::Flagged(d_temp_storage.data(),
                                 temp_storage_bytes,
                                 d_in,
                                 tied_flags.data(),
                                 d_out,
                                 d_num_tied.data(),
                                 num_elements,
                                 stream.value());
      d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
      cub::DeviceSelect::Flagged(d_temp_storage.data(),
                                 temp_storage_bytes,
                                 d_in,
                                 tied_flags.data(),
                                 d_out,
                                 d_num_tied.data(),
                                 num_elements,
                                 stream.value());
    };
    select_flagged(counting, comp_pos.data());
    select_flagged(d_indices_out, child_a.data());
    select_flagged(keys_out.data(), tied_keys_packed.data());
    // `keys_out` is now fully captured -- its ties seeded into `tied_keys_packed` and its
    // singletons already in their final slots -- so free the N-wide first-pass keys before the
    // loop's working set peaks.
    keys_out = rmm::device_uvector<cuda::std::uint64_t>{0, stream};

    // Iterative deepening over the compacted tied set only. `cur_keys`/`cur_child` hold the current
    // sorted keys and child indices of the still-tied elements; each pass computes a dense run rank
    // from those keys, builds the next key as {run rank, next 8-byte window}, and radixes just the
    // tied subset by it. The run rank is the most-significant field, so the sort fixes every
    // element already separated by an earlier window and only reorders within a still-tied run. The
    // first `known_equal_bytes` bytes were ordered by the first pass, so windows start there and
    // advance eight bytes.
    //
    // After each pass the elements that became singletons are scattered to their final output slots
    // and dropped from the working set, so the next pass radixes only the shrinking remainder
    // rather than the full `num_tied` set. A singleton owns a unique run rank, so no later window
    // can move it: its order is final the moment it separates. The loop stops as soon as nothing
    // stays tied, turning a shared prefix that resolves in one window into one pass instead of the
    // full cap.
    //
    // `cur_pos` holds the still-tied output slots and is kept ascending: each pass the element at
    // sorted position p belongs at the p-th remaining slot, so the slots are paired with the sorted
    // children by position and are only compacted (never permuted) -- compaction drops the slots of
    // frozen singletons and leaves the survivors' slots in ascending order. Permuting the slots by
    // the sort would misassign them, so the radix reorders only the keys and children.
    rmm::device_uvector<size_type> child_b(num_tied, stream);
    rmm::device_uvector<prefix_key96> tied_keys_b(num_tied, stream);
    rmm::device_uvector<size_type> pos_b(num_tied, stream);
    rmm::device_uvector<cuda::std::uint32_t> run_ids(num_tied, stream);
    // Per-active-element length range and its per-run reduction, reused each pass to spot runs that
    // have become byte-identical (see `keep_active_window`). Sized to the initial tie count like
    // the other scratch; only the active prefix is touched per pass.
    rmm::device_uvector<len_minmax> elem_minmax(num_tied, stream);
    rmm::device_uvector<len_minmax> run_minmax(num_tied, stream);

    // Seed `tied_keys_a` so the loop's first head-flag reproduces exactly the runs the uint64 sort
    // left tied: a dense run rank over the compacted packed keys, packed back into a `prefix_key96`
    // carrying that rank and the null flag but no window. The first window is built by the pass
    // itself; embedding one here would over-split packed-key ties (see `tied_run_seed_builder`).
    {
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        counting,
                        counting + num_tied,
                        run_ids.begin(),
                        key_head_flag_packed{tied_keys_packed.data()});
      rmm::device_buffer d_temp_storage;
      size_t temp_storage_bytes = 0;
      cub::DeviceScan::InclusiveSum(d_temp_storage.data(),
                                    temp_storage_bytes,
                                    run_ids.data(),
                                    run_ids.data(),
                                    num_tied,
                                    stream.value());
      d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
      cub::DeviceScan::InclusiveSum(d_temp_storage.data(),
                                    temp_storage_bytes,
                                    run_ids.data(),
                                    run_ids.data(),
                                    num_tied,
                                    stream.value());
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        counting,
                        counting + num_tied,
                        tied_keys_a.begin(),
                        tied_run_seed_builder{
                          run_ids.data(), tied_keys_packed.data(), layout.prefix_bits, has_nulls});
    }

    auto* cur_keys      = tied_keys_a.data();
    auto* nxt_keys      = tied_keys_b.data();
    auto* cur_child     = child_a.data();
    auto* nxt_child     = child_b.data();
    auto* cur_pos       = comp_pos.data();
    auto* nxt_pos       = pos_b.data();
    auto consumed_bytes = known_equal_bytes;
    auto num_active     = num_tied;
    for (size_type pass = 0; pass < MAX_RADIX_PASSES && num_active > 0; ++pass) {
      // Dense one-based run rank over the active subset: a key change starts a new run, so an
      // inclusive sum of head flags gives a rank identical within a run and rising across runs.
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        counting,
                        counting + num_active,
                        run_ids.begin(),
                        key_head_flag{cur_keys});
      {
        rmm::device_buffer d_temp_storage;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage.data(),
                                      temp_storage_bytes,
                                      run_ids.data(),
                                      run_ids.data(),
                                      num_active,
                                      stream.value());
        d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
        cub::DeviceScan::InclusiveSum(d_temp_storage.data(),
                                      temp_storage_bytes,
                                      run_ids.data(),
                                      run_ids.data(),
                                      num_active,
                                      stream.value());
      }

      // Build the next key {run rank, next eight bytes} for each active element, then radix the
      // active subset by it. All 96 bits are significant: the run rank and null flag fill the high
      // 32-bit seg_null field and the eight-byte window the low 64 bits (prefix_hi then prefix_lo).
      // The current child indices feed back as the paired values.
      thrust::transform(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        counting,
        counting + num_active,
        nxt_keys,
        window_key_builder{
          run_ids.data(), cur_child, *d_input, has_nulls, consumed_bytes, polarity});
      {
        rmm::device_buffer d_temp_storage;
        size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                        temp_storage_bytes,
                                        nxt_keys,
                                        cur_keys,
                                        cur_child,
                                        nxt_child,
                                        num_active,
                                        decomposer,
                                        0,
                                        key_bits,
                                        stream.value());
        d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
        cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                        temp_storage_bytes,
                                        nxt_keys,
                                        cur_keys,
                                        cur_child,
                                        nxt_child,
                                        num_active,
                                        decomposer,
                                        0,
                                        key_bits,
                                        stream.value());
      }
      // The radix wrote the sorted keys into `cur_keys` and the sorted child indices into
      // `nxt_child`; swap so `cur_child` names the sorted indices and the old one becomes scratch.
      cuda::std::swap(cur_child, nxt_child);

      // Decide which active positions stay tied under the refreshed order. A position is dropped
      // when it is a singleton (its key differs from both neighbors, so its order is final) or when
      // its run has become byte-identical: length-uniform with a common length no greater than the
      // bytes the windows have now covered, so every element is a copy of one string and the stable
      // radix order is already correct. The per-run length range is reduced over a dense post-sort
      // run rank; a mixed-length run (a zero-extension family) is kept for the length-aware
      // cleanup. `covered` is `consumed_bytes` plus this pass's window -- the bytes a run now
      // agrees on.
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        counting,
                        counting + num_active,
                        elem_minmax.begin(),
                        string_length_minmax{cur_child, *d_input});
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        counting,
                        counting + num_active,
                        run_ids.begin(),
                        key_head_flag{cur_keys});
      {
        rmm::device_buffer d_temp_storage;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage.data(),
                                      temp_storage_bytes,
                                      run_ids.data(),
                                      run_ids.data(),
                                      num_active,
                                      stream.value());
        d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
        cub::DeviceScan::InclusiveSum(d_temp_storage.data(),
                                      temp_storage_bytes,
                                      run_ids.data(),
                                      run_ids.data(),
                                      num_active,
                                      stream.value());
      }
      thrust::reduce_by_key(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        run_ids.begin(),
        run_ids.begin() + num_active,
        elem_minmax.begin(),
        cuda::make_discard_iterator(),
        run_minmax.begin(),
        cuda::std::equal_to<cuda::std::uint32_t>{},
        len_minmax_combine{});
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        counting,
                        counting + num_active,
                        tied_flags.begin(),
                        keep_active_window{cur_keys,
                                           num_active,
                                           run_ids.data(),
                                           run_minmax.data(),
                                           consumed_bytes + prefix_bytes,
                                           polarity.element_class(true)});

      // Freeze the resolved singletons: scatter each untied element's child index to its final
      // output slot (the slot at its sorted position). They are dropped from the working set below
      // and never revisited.
      thrust::scatter_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         cur_child,
                         cur_child + num_active,
                         cur_pos,
                         tied_flags.begin(),
                         d_indices_out,
                         cuda::std::logical_not<bool>{});

      // Compact the keys, child indices, and output slots down to just the still-tied elements so
      // the next pass radixes only that remainder. The slots stay ascending because compaction
      // preserves order. Each compaction reports the same surviving count; reading it is the one
      // host synchronization per pass (at most MAX_RADIX_PASSES of them). A device-side predicate
      // feeding the next launch could remove the sync if profiling shows it dominates, but the pass
      // count is tiny so it is left simple here.
      auto const compact = [&](auto const* d_in, auto* d_out) {
        rmm::device_buffer d_temp_storage;
        size_t temp_storage_bytes = 0;
        cub::DeviceSelect::Flagged(d_temp_storage.data(),
                                   temp_storage_bytes,
                                   d_in,
                                   tied_flags.data(),
                                   d_out,
                                   d_num_tied.data(),
                                   num_active,
                                   stream.value());
        d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
        cub::DeviceSelect::Flagged(d_temp_storage.data(),
                                   temp_storage_bytes,
                                   d_in,
                                   tied_flags.data(),
                                   d_out,
                                   d_num_tied.data(),
                                   num_active,
                                   stream.value());
      };
      compact(cur_keys, nxt_keys);
      compact(cur_child, nxt_child);
      compact(cur_pos, nxt_pos);
      cuda::std::swap(cur_keys, nxt_keys);
      cuda::std::swap(cur_child, nxt_child);
      cuda::std::swap(cur_pos, nxt_pos);
      num_active = d_num_tied.value(stream);
      consumed_bytes += prefix_bytes;
    }

    // Resolve any runs still tied after the windows by comparison so the result matches the
    // comparison path exactly. A surviving run agrees on the first `consumed_bytes` bytes, so the
    // comparison skips them and orders by the remainder. The survivors are runs that did not
    // resolve within the pass cap and the mixed-length zero-extension families the drop
    // deliberately kept -- a string and that same string plus trailing bytes, a pure length tie no
    // window can separate but `compare_suffix`'s length difference does; byte-identical runs were
    // already frozen during the loop. A null run carries the null flag in `seg_null` and is left
    // untouched -- and none reach here, since the count-first gate excludes nulls. Singletons
    // resolved earlier were already scattered, so only the survivors remain here; when none do this
    // is a no-op.
    auto const tie_offset = consumed_bytes;
    thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     counting,
                     counting + num_active,
                     prefix_tie_breaker{cur_keys,
                                        cur_child,
                                        *d_input,
                                        num_active,
                                        tie_offset,
                                        polarity.descending,
                                        polarity.element_class(true)});

    // Scatter the surviving child indices back to their output slots. Singletons frozen during the
    // loop, and every untied position from the first pass, keep their order untouched.
    thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cur_child,
                    cur_child + num_active,
                    cur_pos,
                    d_indices_out);
  }

  return sorted_indices;
}

bool strings_grad_all_segments_fit(column_view const& segment_offsets, rmm::cuda_stream_view stream)
{
  auto const num_segments = segment_offsets.size() - 1;
  auto const d_offsets    = segment_offsets.begin<size_type>();
  auto const oversized =
    thrust::count_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<size_type>{0},
                     cuda::counting_iterator<size_type>{num_segments},
                     segment_exceeds_size{d_offsets, STRINGS_GRAD_WARP_CAP});
  return oversized == 0;
}

[[nodiscard]] std::unique_ptr<column> fast_segmented_sorted_order_strings_grad(
  column_view const& input,
  column_view const& segment_offsets,
  sort_polarity polarity,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_elements = input.size();
  auto const num_segments = segment_offsets.size() - 1;
  auto const d_input      = column_device_view::create(input, stream);
  auto const has_nulls    = input.has_nulls();
  auto const d_offsets    = segment_offsets.begin<size_type>();

  auto sorted_indices = cudf::make_numeric_column(
    data_type{type_to_id<size_type>()}, num_elements, mask_state::UNALLOCATED, stream, mr);
  auto* const d_out = sorted_indices->mutable_view().begin<size_type>();

  // The band kernel walks an explicit segment list; this path sorts every segment, so the list is
  // the identity sequence shared by every band.
  rmm::device_uvector<size_type> seg_list(num_segments, stream);
  thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   seg_list.begin(),
                   seg_list.end(),
                   0);

  // The bottom bands take the 8-byte comparator key, the upper bands the 16-byte prekey; both
  // triples carry the requested `polarity`. The pad class stays fixed at `tier_pad`, which ranks
  // strictly above either element class regardless of polarity.
  auto const cmp_build = strings_grad_cmp_key_builder{*d_input, has_nulls, polarity};
  auto const cmp_less  = strings_grad_cmp_less{*d_input, polarity};
  auto const cmp_pad   = strings_grad_cmp_key{0, static_cast<cuda::std::uint32_t>(tier_pad)};
  auto const pre_build = strings_grad_prekey_builder{*d_input, has_nulls, polarity};
  auto const pre_less  = strings_grad_prekey_less{*d_input, polarity};
  auto const pre_pad   = strings_grad_prekey{0, 0, static_cast<cuda::std::uint32_t>(tier_pad)};

  // (0,8] sorts at one item per lane in an 8-slot tile; (8,16] at two items; then W16/W32 for the
  // upper bands. Each band self-filters to its size slice over the shared segment list.
  launch_strings_grad_band<8, 1>(
    cmp_build, cmp_less, cmp_pad, d_offsets, seg_list.data(), num_segments, 0, 8, d_out, stream);
  launch_strings_grad_band<8, 2>(
    cmp_build, cmp_less, cmp_pad, d_offsets, seg_list.data(), num_segments, 8, 16, d_out, stream);
  launch_strings_grad_band<16, 2>(
    pre_build, pre_less, pre_pad, d_offsets, seg_list.data(), num_segments, 16, 32, d_out, stream);
  launch_strings_grad_band<32, 2>(pre_build,
                                  pre_less,
                                  pre_pad,
                                  d_offsets,
                                  seg_list.data(),
                                  num_segments,
                                  32,
                                  STRINGS_GRAD_WARP_CAP,
                                  d_out,
                                  stream);
  return sorted_indices;
}

}  // namespace detail
}  // namespace cudf
