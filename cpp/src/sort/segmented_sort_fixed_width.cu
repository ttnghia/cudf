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
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_partition.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/warp/warp_merge_sort.cuh>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <thrust/count.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

namespace cudf {
namespace detail {
namespace {

/**
 * @brief Maps a fixed-width value to the unsigned key whose ascending order equals value order
 *
 * Integrals: an unsigned input sorts by its bit pattern unchanged; a signed input has its sign bit
 * flipped so every negative precedes every non-negative. `bool` is already the ordered set {0, 1}.
 * Timestamps/durations order by their signed integer rep, so the rep is extracted and re-encoded
 * at its width, matching cudf's own radix sort. Floating point takes the order-preserving IEEE-754
 * flip: every NaN maps to the all-ones key (so all NaNs compare equal and sort after +Inf), and the
 * sign is mapped so -Inf < finite (incl. +/-0, denormals) < +Inf -- monotone across every finite
 * value. The transform runs at the input width so the flip lands on the correct bit; the caller
 * zero-extends to the 32-bit value field, and the equal high bits the widening adds cannot reorder.
 * Fixed-point reps are signed integers with one column-wide scale, so rep order equals value order.
 * The type branches are `if constexpr` and ordered so the non-integral cases resolve before
 * `make_unsigned_t`, which is ill-formed for floating-point and chrono types.
 */
template <typename T>
__device__ inline cuda::std::uint32_t radix_encode_u32(T value)
{
  if constexpr (cuda::std::is_same_v<T, bool>) {
    return value ? cuda::std::uint32_t{1} : cuda::std::uint32_t{0};
  } else if constexpr (cudf::is_timestamp<T>()) {
    return radix_encode_u32(value.time_since_epoch().count());
  } else if constexpr (cudf::is_duration<T>()) {
    return radix_encode_u32(value.count());
  } else if constexpr (cuda::std::is_floating_point_v<T>) {
    if (cuda::std::isnan(value)) { return ~cuda::std::uint32_t{0}; }
    auto const bits          = cuda::std::bit_cast<cuda::std::uint32_t>(value);
    auto constexpr sign_mask = cuda::std::uint32_t{1} << (sizeof(cuda::std::uint32_t) * 8 - 1);
    return (bits & sign_mask) ? ~bits : (bits | sign_mask);
  } else {
    using U      = cuda::std::make_unsigned_t<T>;
    auto encoded = static_cast<U>(value);
    if constexpr (cuda::std::is_signed_v<T>) {
      encoded ^= static_cast<U>(U{1} << (sizeof(U) * 8 - 1));
    }
    return static_cast<cuda::std::uint32_t>(encoded);
  }
}

/**
 * @brief The eight-byte analogue of `radix_encode_u32` for the `prefix_key96` value words
 *
 * Reached for eight-byte element types: `int64`/`uint64`, the `DECIMAL64` rep, `double`, and the
 * int64-rep timestamps/durations. `bool` never reaches here. Chrono reps are extracted, floating
 * point takes the IEEE-754 flip (every NaN -> all-ones), and integrals flip the sign bit of the
 * 64-bit two's-complement pattern -- the same transforms as the 32-bit encoder, at this width.
 */
template <typename T>
__device__ inline cuda::std::uint64_t radix_encode_u64(T value)
{
  if constexpr (cudf::is_timestamp<T>()) {
    return radix_encode_u64(value.time_since_epoch().count());
  } else if constexpr (cudf::is_duration<T>()) {
    return radix_encode_u64(value.count());
  } else if constexpr (cuda::std::is_floating_point_v<T>) {
    if (cuda::std::isnan(value)) { return ~cuda::std::uint64_t{0}; }
    auto const bits          = cuda::std::bit_cast<cuda::std::uint64_t>(value);
    auto constexpr sign_mask = cuda::std::uint64_t{1} << (sizeof(cuda::std::uint64_t) * 8 - 1);
    return (bits & sign_mask) ? ~bits : (bits | sign_mask);
  } else {
    using U      = cuda::std::make_unsigned_t<T>;
    auto encoded = static_cast<U>(value);
    if constexpr (cuda::std::is_signed_v<T>) {
      encoded ^= static_cast<U>(U{1} << (sizeof(U) * 8 - 1));
    }
    return static_cast<cuda::std::uint64_t>(encoded);
  }
}

/**
 * @brief The sixteen-byte analogue of `radix_encode_u64`: the order-preserving 128-bit encoding
 *
 * Only reached for the `DECIMAL128` storage rep (`__int128_t`), which is always signed, so the sign
 * bit of the 128-bit two's-complement pattern is flipped unconditionally -- the 128-bit form of the
 * same order-preserving transform the narrower encoders apply. Fixed-point reps carry one
 * column-wide scale, so ordering by the rep equals value order.
 */
template <typename T>
__device__ inline unsigned __int128 radix_encode_u128(T value)
{
  auto encoded = static_cast<unsigned __int128>(value);
  encoded ^= (static_cast<unsigned __int128>(1) << 127);
  return encoded;
}

/**
 * @brief Builds the single-`uint64` packed key for a fixed-width element of width four bytes or
 * less
 *
 * The word is laid out most- to least-significant as `[segment : S bits][class : 1 bit][value :
 * 32 bits]`, with `S = bit_width(num_segments)`; an unsigned compare therefore orders by segment,
 * then by the polarity's null placement (the class bit is 1 for a null under nulls-last, 1 for a
 * valid under nulls-first), then by the radix-encoded value -- complemented within its 32-bit field
 * when descending, which exactly reverses value order without reaching the class bit. The class
 * bit sits at bit 32, one bit above the 32-bit value field and (since `S <= 31`) strictly below
 * the segment field, so a null carries a zero value yet still sorts on its configured side of every
 * valid element in its segment regardless of that element's value, and no value can reach the class
 * bit. Any bits between the class bit and the segment field are left zero and identical for all
 * elements, so they never affect the order.
 */
template <typename T>
struct numeric_packed_key_builder {
  size_type const* d_segment_ids;
  column_device_view const d_input;
  bool const has_nulls;
  int const segment_bits;
  sort_polarity const polarity;
  __device__ cuda::std::uint64_t operator()(size_type idx) const
  {
    auto const segment_part = static_cast<cuda::std::uint64_t>(d_segment_ids[idx])
                              << (64 - segment_bits);
    if (has_nulls && d_input.is_null(idx)) {
      // Null: only its class bit one above the value field; the value stays zero and unread.
      return segment_part | (static_cast<cuda::std::uint64_t>(polarity.element_class(true))
                             << (sizeof(cuda::std::uint32_t) * 8));
    }
    return segment_part |
           (static_cast<cuda::std::uint64_t>(polarity.element_class(false))
            << (sizeof(cuda::std::uint32_t) * 8)) |
           static_cast<cuda::std::uint64_t>(radix_encode_u32<T>(d_input.element<T>(idx)) ^
                                            polarity.value_mask32());
  }
};

/**
 * @brief Builds the `prefix_key96` packed key for an eight-byte fixed-width element
 *
 * Reuses the strings path's key: `seg_null` packs the segment ordinal in bits 1..31 and the
 * polarity's class bit in bit 0 (via `pack_seg_null`), so the segment dominates and a null sorts
 * on its configured side of every valid element sharing its segment; the two window words carry
 * the radix-encoded 64-bit value split most- then least-significant -- complemented when
 * descending, which reverses value order and distributes over the split -- so the
 * `prefix_decomposer` orders tied segments by the full value. A null leaves the value words zero
 * and unread.
 */
template <typename T>
struct numeric_packed_key_builder64 {
  size_type const* d_segment_ids;
  column_device_view const d_input;
  bool const has_nulls;
  sort_polarity const polarity;
  __device__ prefix_key96 operator()(size_type idx) const
  {
    auto const segment = static_cast<cuda::std::uint32_t>(d_segment_ids[idx]);
    if (has_nulls && d_input.is_null(idx)) {
      return prefix_key96{pack_seg_null(segment, polarity.element_class(true)), 0u, 0u};
    }
    prefix_key96 key{pack_seg_null(segment, polarity.element_class(false)), 0u, 0u};
    split_prefix(radix_encode_u64<T>(d_input.element<T>(idx)) ^ polarity.value_mask64(),
                 key.prefix_hi,
                 key.prefix_lo);
    return key;
  }
};

// A DECIMAL128 segment set is sorted by the narrowest lossless radix key its value range admits,
// rather than an unconditional twenty-pass 160-bit key. One range reduction picks a min-biased
// `uint64` (range < 2^32, eight passes), a `prefix_key96` (range fits int64, twelve passes), or a
// two-phase hi64-then-lo64 sort (genuine 128-bit range) -- the machinery below, shared by the
// full-column and compact-large-segment DECIMAL128 radix sites.

/**
 * @brief Inclusive value range [min, max] over a set of `__int128` (DECIMAL128) elements
 *
 * Reduced across the elements a DECIMAL128 radix site is about to sort so the site can pick the
 * narrowest key that still encodes every value (the width gate). Nulls contribute the identity and
 * are skipped: a null is placed by its key class alone, so its value is never compared.
 */
struct dec128_value_range {
  __int128_t min_val;
  __int128_t max_val;
};

// Identity for the range reduction: a min seeded to the largest `__int128` and a max to the
// smallest, so any real value wins both and a null-only (or empty) set leaves the range degenerate.
__host__ __device__ inline dec128_value_range dec128_value_range_identity()
{
  auto constexpr i128_max = static_cast<__int128_t>((static_cast<unsigned __int128>(1) << 127) - 1);
  return dec128_value_range{i128_max, static_cast<__int128_t>(-i128_max - 1)};
}

/// Combines two `dec128_value_range`s: min of the minima, max of the maxima. Associative and
/// commutative, as `thrust::transform_reduce` requires.
struct dec128_value_range_combine {
  __device__ dec128_value_range operator()(dec128_value_range const& a,
                                           dec128_value_range const& b) const
  {
    return dec128_value_range{a.min_val < b.min_val ? a.min_val : b.min_val,
                              a.max_val > b.max_val ? a.max_val : b.max_val};
  }
};

/**
 * @brief Maps an element index to its single-value range; a null contributes the identity
 *
 * The index is a global element index -- the full-column path passes the element ordinals, the
 * compact-large-segment path the radix-tier global indices -- so the same functor reduces either
 * set.
 */
struct dec128_value_range_fn {
  column_device_view d_input;
  bool has_nulls;
  __device__ dec128_value_range operator()(size_type idx) const
  {
    if (has_nulls && d_input.is_null(idx)) { return dec128_value_range_identity(); }
    auto const value = d_input.element<__int128_t>(idx);
    return dec128_value_range{value, value};
  }
};

/**
 * @brief Builds the single-`uint64` gate key for a DECIMAL128 element whose column-wide value range
 * spans fewer than 2^32 (the min-biased path)
 *
 * Same layout as `numeric_packed_key_builder`: `[segment : S bits][class : 1][value : 32 bits]`.
 * The value is `v - min_val`, which lies in `[0, range]` with `range < 2^32` and is monotonic in
 * `v` -- complemented within the 32-bit field when descending, which reverses that order while
 * staying inside the field -- so an unsigned compare orders by segment, then by the polarity's
 * null placement, then by value, at eight byte-passes instead of the twenty a full 128-bit key
 * needs. A null carries a zero value and its class bit one bit above the value field.
 */
struct dec128_biased_u64_key_builder {
  size_type const* d_segment_ids;
  column_device_view const d_input;
  bool const has_nulls;
  int const segment_bits;
  __int128_t const min_val;
  sort_polarity const polarity;
  __device__ cuda::std::uint64_t operator()(size_type idx) const
  {
    auto const segment_part = static_cast<cuda::std::uint64_t>(d_segment_ids[idx])
                              << (64 - segment_bits);
    if (has_nulls && d_input.is_null(idx)) {
      return segment_part | (static_cast<cuda::std::uint64_t>(polarity.element_class(true))
                             << (sizeof(cuda::std::uint32_t) * 8));
    }
    auto const biased = static_cast<cuda::std::uint32_t>(
      static_cast<unsigned __int128>(d_input.element<__int128_t>(idx)) -
      static_cast<unsigned __int128>(min_val));
    return segment_part |
           (static_cast<cuda::std::uint64_t>(polarity.element_class(false))
            << (sizeof(cuda::std::uint32_t) * 8)) |
           static_cast<cuda::std::uint64_t>(biased ^ polarity.value_mask32());
  }
};

/**
 * @brief Builds the twelve-byte `prefix_key96` gate key for a DECIMAL128 element whose value fits
 * `int64` (the fits-int64 path)
 *
 * Reuses `numeric_packed_key_builder64`'s layout but sources the value from the 128-bit rep
 * narrowed to `int64` and re-encoded at 64 bits, so tied segments order by the full
 * (int64-representable) value at twelve byte-passes instead of twenty. The caller guarantees every
 * non-null value is `int64`-representable, so the narrowing is lossless.
 */
struct dec128_int64_key_builder {
  size_type const* d_segment_ids;
  column_device_view const d_input;
  bool const has_nulls;
  sort_polarity const polarity;
  __device__ prefix_key96 operator()(size_type idx) const
  {
    auto const segment = static_cast<cuda::std::uint32_t>(d_segment_ids[idx]);
    if (has_nulls && d_input.is_null(idx)) {
      return prefix_key96{pack_seg_null(segment, polarity.element_class(true)), 0u, 0u};
    }
    prefix_key96 key{pack_seg_null(segment, polarity.element_class(false)), 0u, 0u};
    split_prefix(radix_encode_u64<int64_t>(static_cast<int64_t>(d_input.element<__int128_t>(idx))) ^
                   polarity.value_mask64(),
                 key.prefix_hi,
                 key.prefix_lo);
    return key;
  }
};

/**
 * @brief Phase-one key of the two-phase DECIMAL128 sort: `prefix_key96` {seg_null, high 64 bits}
 *
 * The value words carry the high 64 bits of the sign-flipped 128-bit encoding (the bit-127 flip
 * lands on bit 63 of that word), complemented when descending -- equal high words stay equal, so
 * the phase-two tie set is polarity-independent while the resolved order reverses. A first radix
 * over (segment, class, hi64) thus orders every element whose hi64 differs and leaves only exact
 * hi64 ties for the low-64 phase. A null sets its class bit and leaves the words zero, sorting on
 * the polarity's side.
 */
struct dec128_hi64_key_builder {
  size_type const* d_segment_ids;
  column_device_view const d_input;
  bool const has_nulls;
  sort_polarity const polarity;
  __device__ prefix_key96 operator()(size_type idx) const
  {
    auto const segment = static_cast<cuda::std::uint32_t>(d_segment_ids[idx]);
    if (has_nulls && d_input.is_null(idx)) {
      return prefix_key96{pack_seg_null(segment, polarity.element_class(true)), 0u, 0u};
    }
    prefix_key96 key{pack_seg_null(segment, polarity.element_class(false)), 0u, 0u};
    auto const encoded = radix_encode_u128<__int128_t>(d_input.element<__int128_t>(idx));
    split_prefix(static_cast<cuda::std::uint64_t>(encoded >> 64) ^ polarity.value_mask64(),
                 key.prefix_hi,
                 key.prefix_lo);
    return key;
  }
};

/**
 * @brief Phase-two key of the two-phase DECIMAL128 sort: `prefix_key96` {run rank, low 64 bits}
 *
 * Applied only to the elements still tied after phase one (equal segment and hi64), which are all
 * non-null. `d_run_ids` is a dense one-based rank over those tied elements, so the run rank
 * dominates and keeps distinct (segment, hi64) runs apart while the low 64 bits -- unaffected by
 * the bit-127 flip, hence equal to the raw low word, and complemented when descending so equal
 * high words resolve in reversed value order -- order within a run. `d_child` maps each tied slot
 * to its element's global index.
 */
struct dec128_lo64_key_builder {
  cuda::std::uint32_t const* d_run_ids;
  size_type const* d_child;
  column_device_view const d_input;
  sort_polarity const polarity;
  __device__ prefix_key96 operator()(size_type i) const
  {
    prefix_key96 key{pack_seg_null(d_run_ids[i], 0u), 0u, 0u};
    auto const value = static_cast<unsigned __int128>(d_input.element<__int128_t>(d_child[i]));
    split_prefix(static_cast<cuda::std::uint64_t>(value) ^ polarity.value_mask64(),
                 key.prefix_hi,
                 key.prefix_lo);
    return key;
  }
};

/**
 * @brief Two-phase DECIMAL128 sort: order by the high 64 value bits, then resolve exact hi64 ties
 * by the low 64 bits
 *
 * `global_indices` lists the elements' global indices and is the paired sort value, so `d_order[j]`
 * receives the global index of the j-th smallest element. Phase one radix-sorts a twelve-byte key
 * {segment, null, hi64}; on data whose high words differ this alone is the order. Only elements
 * sharing (segment, hi64) with a neighbor -- nulls excluded, being position-final -- take phase
 * two, which re-sorts just those by {dense run rank, lo64} and scatters the refined order back into
 * their (ascending) slots. Correct because the sign-flipped 128-bit encoding compares
 * lexicographically hi64 then lo64, so the two passes together reproduce full value order at
 * twelve-byte keys. Each phase trims the radix end bit to the significant width of its leading
 * field -- the segment field in phase one, the run-rank field in phase two -- whose higher bits are
 * constant zero across every key.
 */
inline void dec128_two_phase_sort(column_device_view const& d_input,
                                  size_type const* d_segment_ids,
                                  bool has_nulls,
                                  sort_polarity polarity,
                                  size_type const* global_indices,
                                  size_type num_elements,
                                  int segment_bits,
                                  size_type* d_order,
                                  rmm::cuda_stream_view stream)
{
  // The run rank rides seg_null's high 31 bits and phase two's null flag (always zero) its bit 0,
  // the same (label << 1) | flag packing the shipped keys use, so the shift must fit a uint32.
  static_assert(2ull * cuda::std::numeric_limits<size_type>::max() + 1ull <=
                  cuda::std::numeric_limits<cuda::std::uint32_t>::max(),
                "size_type run rank does not fit prefix_key96::seg_null after the shift");
  auto const alloc      = cudf::get_current_device_resource_ref();
  auto const counting   = cuda::counting_iterator<size_type>{0};
  auto const decomposer = prefix_decomposer{};
  // Phase one's leading field is seg_null = (segment << 1) | flag, significant to segment_bits + 1
  // bits (capped at the 32-bit field); the constant-zero high bits are dropped from the radix.
  auto const phase1_end = cuda::std::min(static_cast<int>(sizeof(prefix_key96) * 8),
                                         64 + cuda::std::min(32, segment_bits + 1));

  // Phase one: sort by (segment, null, hi64). `d_order` gets the sorted global indices; `k1` the
  // sorted keys, read below to find exact hi64 ties.
  rmm::device_uvector<prefix_key96> k1(num_elements, stream);
  {
    rmm::device_uvector<prefix_key96> keys_in(num_elements, stream);
    thrust::transform(rmm::exec_policy_nosync(stream, alloc),
                      global_indices,
                      global_indices + num_elements,
                      keys_in.begin(),
                      dec128_hi64_key_builder{d_segment_ids, d_input, has_nulls, polarity});
    rmm::device_buffer d_temp_storage;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                    temp_storage_bytes,
                                    keys_in.data(),
                                    k1.data(),
                                    global_indices,
                                    d_order,
                                    num_elements,
                                    decomposer,
                                    0,
                                    phase1_end,
                                    stream.value());
    d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
    cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                    temp_storage_bytes,
                                    keys_in.data(),
                                    k1.data(),
                                    global_indices,
                                    d_order,
                                    num_elements,
                                    decomposer,
                                    0,
                                    phase1_end,
                                    stream.value());
  }

  // Zero-tie gate: only elements sharing (segment, hi64) with a neighbor need the low-64 phase;
  // nulls are position-final and excluded. On distinct-hi64 data nothing is tied, so phase one is
  // the order.
  auto const any_tied =
    thrust::count_if(rmm::exec_policy_nosync(stream, alloc),
                     counting,
                     counting + num_elements,
                     key_tied_flag{k1.data(), num_elements, polarity.element_class(true)}) > 0;
  if (not any_tied) { return; }

  // Compact the still-tied slots' output positions, global indices, and phase-one keys.
  rmm::device_uvector<bool> tied_flags(num_elements, stream);
  thrust::transform(rmm::exec_policy_nosync(stream, alloc),
                    counting,
                    counting + num_elements,
                    tied_flags.begin(),
                    key_tied_flag{k1.data(), num_elements, polarity.element_class(true)});
  auto const num_tied = static_cast<size_type>(thrust::count(
    rmm::exec_policy_nosync(stream, alloc), tied_flags.begin(), tied_flags.end(), true));

  rmm::device_uvector<size_type> comp_pos(num_tied, stream);
  rmm::device_uvector<size_type> child(num_tied, stream);
  rmm::device_uvector<prefix_key96> tied_keys(num_tied, stream);
  cudf::detail::device_scalar<size_type> d_num_tied(stream);
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
  select_flagged(d_order, child.data());
  select_flagged(k1.data(), tied_keys.data());

  // Dense one-based run rank over the tied subset: one rank per (segment, hi64) run.
  rmm::device_uvector<cuda::std::uint32_t> run_ids(num_tied, stream);
  thrust::transform(rmm::exec_policy_nosync(stream, alloc),
                    counting,
                    counting + num_tied,
                    run_ids.begin(),
                    key_head_flag{tied_keys.data()});
  {
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
  }

  // Phase two: sort the tied subset by {run rank, lo64}. The run rank is bounded by the tied count,
  // so its packed field (rank << 1) needs at most bit_width(num_tied) + 1 bits; the end bit trims
  // to that, matching phase one's trim on its own leading field.
  auto const run_field_bits = cuda::std::min(
    32, static_cast<int>(cuda::std::bit_width(static_cast<cuda::std::uint64_t>(num_tied))) + 1);
  auto const phase2_end =
    cuda::std::min(static_cast<int>(sizeof(prefix_key96) * 8), 64 + run_field_bits);
  rmm::device_uvector<prefix_key96> p2_in(num_tied, stream);
  rmm::device_uvector<prefix_key96> p2_out(num_tied, stream);
  rmm::device_uvector<size_type> child_sorted(num_tied, stream);
  thrust::transform(rmm::exec_policy_nosync(stream, alloc),
                    counting,
                    counting + num_tied,
                    p2_in.begin(),
                    dec128_lo64_key_builder{run_ids.data(), child.data(), d_input, polarity});
  {
    rmm::device_buffer d_temp_storage;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                    temp_storage_bytes,
                                    p2_in.data(),
                                    p2_out.data(),
                                    child.data(),
                                    child_sorted.data(),
                                    num_tied,
                                    decomposer,
                                    0,
                                    phase2_end,
                                    stream.value());
    d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
    cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                    temp_storage_bytes,
                                    p2_in.data(),
                                    p2_out.data(),
                                    child.data(),
                                    child_sorted.data(),
                                    num_tied,
                                    decomposer,
                                    0,
                                    phase2_end,
                                    stream.value());
  }
  thrust::scatter(rmm::exec_policy_nosync(stream, alloc),
                  child_sorted.begin(),
                  child_sorted.end(),
                  comp_pos.begin(),
                  d_order);
}

/**
 * @brief Sorts a DECIMAL128 element set within its segments, picking the narrowest lossless radix
 * key from the value range
 *
 * `global_indices` lists the elements' global indices -- the full-column path passes a dense
 * sequence, the compact-large-segment path the radix-tier indices -- and doubles as the paired sort
 * value, so `d_order[j]` receives the global index of the j-th smallest element. A single range
 * reduction (nulls skipped, one D->H sync) picks the key: a range under 2^32 takes a min-biased
 * `uint64` (eight passes), a range fitting `int64` a `prefix_key96` (twelve passes, end bit
 * trimmed), and a genuine 128-bit range the two-phase hi64-then-lo64 sort -- replacing the
 * twenty-pass 160-bit key the width never needs. The reduction is bandwidth-bound and small against
 * the radix it narrows.
 */
inline void dec128_segmented_radix_sort(column_device_view const& d_input,
                                        size_type const* d_segment_ids,
                                        bool has_nulls,
                                        sort_polarity polarity,
                                        int segment_bits,
                                        size_type const* global_indices,
                                        size_type num_elements,
                                        size_type* d_order,
                                        rmm::cuda_stream_view stream)
{
  // The segment ordinal rides seg_null's high 31 bits and the null flag its bit 0, the same
  // (label << 1) | flag packing the shipped keys use, so the shift must fit a uint32.
  static_assert(2ull * cuda::std::numeric_limits<size_type>::max() + 1ull <=
                  cuda::std::numeric_limits<cuda::std::uint32_t>::max(),
                "size_type segment label does not fit the packed key's seg_null after the shift");
  auto const alloc = cudf::get_current_device_resource_ref();

  // Range reduction over the sign-preserving raw values, nulls skipped; returns to the host.
  auto const range = thrust::transform_reduce(rmm::exec_policy_nosync(stream, alloc),
                                              global_indices,
                                              global_indices + num_elements,
                                              dec128_value_range_fn{d_input, has_nulls},
                                              dec128_value_range_identity(),
                                              dec128_value_range_combine{});

  enum class key_width { u64_biased, key96, twophase };
  auto width         = key_width::twophase;
  __int128_t min_val = 0;
  if (range.max_val < range.min_val) {
    // No non-null value in the set: the null-only order is width-independent, so take the cheapest.
    width = key_width::u64_biased;
  } else {
    auto const span =
      static_cast<unsigned __int128>(range.max_val) - static_cast<unsigned __int128>(range.min_val);
    auto constexpr i64_min = static_cast<__int128_t>(cuda::std::numeric_limits<int64_t>::min());
    auto constexpr i64_max = static_cast<__int128_t>(cuda::std::numeric_limits<int64_t>::max());
    if ((span >> 32) == 0) {
      width   = key_width::u64_biased;
      min_val = range.min_val;
    } else if (range.min_val >= i64_min and range.max_val <= i64_max) {
      width = key_width::key96;
    } else {
      width = key_width::twophase;
    }
  }

  if (width == key_width::u64_biased) {
    // Min-biased 32-bit value in a single uint64 key: eight byte-passes. The segment rides the high
    // bits, so its end bit cannot be tightened, unlike the wider decomposer key.
    rmm::device_uvector<cuda::std::uint64_t> keys_in(num_elements, stream);
    rmm::device_uvector<cuda::std::uint64_t> keys_out(num_elements, stream);
    thrust::transform(rmm::exec_policy_nosync(stream, alloc),
                      global_indices,
                      global_indices + num_elements,
                      keys_in.begin(),
                      dec128_biased_u64_key_builder{
                        d_segment_ids, d_input, has_nulls, segment_bits, min_val, polarity});
    rmm::device_buffer d_temp_storage;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                    temp_storage_bytes,
                                    keys_in.data(),
                                    keys_out.data(),
                                    global_indices,
                                    d_order,
                                    num_elements,
                                    0,
                                    static_cast<int>(sizeof(cuda::std::uint64_t) * 8),
                                    stream.value());
    d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
    cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                    temp_storage_bytes,
                                    keys_in.data(),
                                    keys_out.data(),
                                    global_indices,
                                    d_order,
                                    num_elements,
                                    0,
                                    static_cast<int>(sizeof(cuda::std::uint64_t) * 8),
                                    stream.value());
  } else if (width == key_width::key96) {
    // Value fits int64: prefix_key96, twelve byte-passes, end bit trimmed to seg_null's width.
    auto const key96_end = cuda::std::min(static_cast<int>(sizeof(prefix_key96) * 8),
                                          64 + cuda::std::min(32, segment_bits + 1));
    rmm::device_uvector<prefix_key96> keys_in(num_elements, stream);
    rmm::device_uvector<prefix_key96> keys_out(num_elements, stream);
    thrust::transform(rmm::exec_policy_nosync(stream, alloc),
                      global_indices,
                      global_indices + num_elements,
                      keys_in.begin(),
                      dec128_int64_key_builder{d_segment_ids, d_input, has_nulls, polarity});
    auto const decomposer = prefix_decomposer{};
    rmm::device_buffer d_temp_storage;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                    temp_storage_bytes,
                                    keys_in.data(),
                                    keys_out.data(),
                                    global_indices,
                                    d_order,
                                    num_elements,
                                    decomposer,
                                    0,
                                    key96_end,
                                    stream.value());
    d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
    cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                    temp_storage_bytes,
                                    keys_in.data(),
                                    keys_out.data(),
                                    global_indices,
                                    d_order,
                                    num_elements,
                                    decomposer,
                                    0,
                                    key96_end,
                                    stream.value());
  } else {
    // Genuine 128-bit range: two-phase hi64-then-lo64, cheaper than the full 160-bit key.
    dec128_two_phase_sort(d_input,
                          d_segment_ids,
                          has_nulls,
                          polarity,
                          global_indices,
                          num_elements,
                          segment_bits,
                          d_order,
                          stream);
  }
}

/**
 * @brief Sorts a single fixed-width key column within its segments by one global radix sort
 *
 * The whole value is encoded into the key, so there is no tie-break: the radix order is the final
 * order. Elements four bytes wide or less use one `uint64` key (eight byte-passes); eight-byte
 * elements use the twelve-byte `prefix_key96` (twelve byte-passes); the `DECIMAL128` rep takes the
 * narrowest lossless key its value range admits (`dec128_segmented_radix_sort`). Each carries the
 * per-element index as the paired value, so the sorted indices are the segmented sorted order. The
 * enabled overload
 * matches every fixed-width type the packed key encodes losslessly -- integrals (incl. `bool`),
 * floating point, timestamps/durations, and the `DECIMAL32/64/128` reps `dispatch_storage_type`
 * maps to `int32`/`int64`/`__int128` -- so only the non-fixed-width types (string, list, struct,
 * dictionary) take the failing overload, which the caller's gate never reaches at run time.
 */
struct numeric_packed_sort_fn {
  // Compile-time counterpart of the runtime gate `is_numeric_packed_radix_supported`: it must
  // accept exactly the same types, so widen the two together. Every fixed-width type the packed key
  // encodes losslessly -- integrals (incl. bool), floating point, timestamps/durations, and the
  // DECIMAL32/64/128 storage reps (int32/int64/__int128). The non-fixed-width types (string, list,
  // struct, dictionary) take the failing overload.
  template <typename T>
  static constexpr bool is_supported()
  {
    return cudf::is_integral<T>() or cudf::is_floating_point<T>() or cudf::is_chrono<T>() or
           cuda::std::is_same_v<__int128, T>;
  }

  template <typename T, CUDF_ENABLE_IF(is_supported<T>())>
  void operator()(column_device_view const& d_input,
                  size_type const* d_segment_ids,
                  bool has_nulls,
                  sort_polarity polarity,
                  int segment_bits,
                  size_type num_elements,
                  size_type* d_indices_out,
                  rmm::cuda_stream_view stream) const
  {
    auto const counting = cuda::counting_iterator<size_type>{0};
    rmm::device_uvector<size_type> indices_in(num_elements, stream);
    thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     indices_in.begin(),
                     indices_in.end(),
                     0);

    if constexpr (sizeof(T) <= 4) {
      // The value occupies a fixed 32 bits and the null flag one bit; the segment field takes the
      // top S = bit_width(num_segments) bits. num_segments is a size_type, so S never exceeds
      // bit_width(size_type max) = 31 and S + 1 + 32 <= 64 -- all three fields always fit the key
      // with the null flag strictly above the value and strictly below the segment field.
      static_assert(cuda::std::bit_width(static_cast<cuda::std::uint64_t>(
                      cuda::std::numeric_limits<size_type>::max())) +
                        1 + 32 <=
                      64,
                    "packed numeric key fields exceed the 64-bit key for some segment count");
      rmm::device_uvector<cuda::std::uint64_t> keys_in(num_elements, stream);
      rmm::device_uvector<cuda::std::uint64_t> keys_out(num_elements, stream);
      thrust::transform(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        counting,
        counting + num_elements,
        keys_in.begin(),
        numeric_packed_key_builder<T>{d_segment_ids, d_input, has_nulls, segment_bits, polarity});
      // One global radix over the full 64 bits (segment in the high bits, value in the low): eight
      // byte-passes. The two-stage temporary-storage call is wrapped so the storage is released
      // before the sort returns.
      {
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
    } else if constexpr (sizeof(T) == 8) {
      // Eight-byte value: reuse prefix_key96 {seg_null, value_hi, value_lo}. The segment ordinal
      // rides seg_null's high 31 bits and the null flag its bit 0, so (label << 1) | flag must fit
      // a uint32 for every size_type segment count.
      static_assert(2ull * cuda::std::numeric_limits<size_type>::max() + 1ull <=
                      cuda::std::numeric_limits<cuda::std::uint32_t>::max(),
                    "size_type segment label does not fit prefix_key96::seg_null after the shift");
      rmm::device_uvector<prefix_key96> keys_in(num_elements, stream);
      rmm::device_uvector<prefix_key96> keys_out(num_elements, stream);
      thrust::transform(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        counting,
        counting + num_elements,
        keys_in.begin(),
        numeric_packed_key_builder64<T>{d_segment_ids, d_input, has_nulls, polarity});
      auto const decomposer = prefix_decomposer{};
      // seg_null = (segment ordinal << 1) | null flag is significant only to segment_bits + 1 bits;
      // the constant-zero bits above it are dropped from the radix end bit, skipping their passes
      // exactly as dec128_segmented_radix_sort's key96 path does for the identical key layout.
      auto const key_bits = cuda::std::min(static_cast<int>(sizeof(prefix_key96) * 8),
                                           64 + cuda::std::min(32, segment_bits + 1));
      {
        rmm::device_buffer d_temp_storage;
        size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                        temp_storage_bytes,
                                        keys_in.data(),
                                        keys_out.data(),
                                        indices_in.data(),
                                        d_indices_out,
                                        num_elements,
                                        decomposer,
                                        0,
                                        key_bits,
                                        stream.value());
        d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
        cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                        temp_storage_bytes,
                                        keys_in.data(),
                                        keys_out.data(),
                                        indices_in.data(),
                                        d_indices_out,
                                        num_elements,
                                        decomposer,
                                        0,
                                        key_bits,
                                        stream.value());
      }
    } else {
      // Sixteen-byte value (the DECIMAL128 rep): pick the narrowest lossless radix key from the
      // value range instead of an unconditional twenty-pass 160-bit key. `indices_in` is the dense
      // global-index sequence, the key-build source and the paired sort value.
      dec128_segmented_radix_sort(d_input,
                                  d_segment_ids,
                                  has_nulls,
                                  polarity,
                                  segment_bits,
                                  indices_in.data(),
                                  num_elements,
                                  d_indices_out,
                                  stream);
    }
  }

  template <typename T, CUDF_ENABLE_IF(not is_supported<T>())>
  void operator()(column_device_view const&,
                  size_type const*,
                  bool,
                  sort_polarity,
                  int,
                  size_type,
                  size_type*,
                  rmm::cuda_stream_view) const
  {
    CUDF_FAIL("Column type cannot be used with the numeric packed-radix segmented sort");
  }
};

/**
 * @brief Run-time predicate for the numeric packed-radix fast path on a key column's type
 *
 * True for every fixed-width type the path encodes losslessly: integrals (including `bool`), all
 * fixed-point types (`DECIMAL32/64/128`), floating point, and timestamps/durations.
 * Only the non-fixed-width types (string, list, struct, dictionary) are excluded.
 *
 * @param type The key column's data type
 * @return true if the type is eligible for the packed-radix fast path
 */
inline bool is_numeric_packed_radix_supported(data_type type)
{
  return cudf::is_integral(type) or cudf::is_fixed_point(type) or cudf::is_floating_point(type) or
         cudf::is_chrono(type);
}

// ==========================================================================================
// Tiered per-segment sort for a single fixed-width key column.
//
// The packed-radix path pays a fixed per-element cost (a global radix over a key as wide as the
// value) that dominates when segments are tiny -- the LIST<DECIMAL128> / small-int regime where the
// value is wide but each segment holds only a handful of elements. This path instead classifies
// segments by size into three tiers, each sorting at a cost that tracks the segment size rather
// than the key width: a segment of <= `TIERED_NETWORK_CAP` elements is sorted by one thread with a
// fixed sorting network held entirely in registers; a segment of <= `TIERED_WARP_CAP` by a full
// warp with `cub::WarpMergeSort`; and a rare large outlier by a packed radix restricted to just
// that segment's elements. Nulls sort on the polarity's side within a segment via a three-valued
// class flag folded into the key, and a descending order complements the key's value bits. Engages
// for any explicit (order, null_order) / unstable on the fixed-width reps
// `is_tiered_sort_supported` admits (the caller resolves the polarity); handles both nullable and
// non-nullable columns.
// ==========================================================================================

/**
 * @brief Ordering key for the sixteen-byte (DECIMAL128) tiered path: a class flag over a 128-bit
 * value
 *
 * `flag` dominates the order, placing the polarity's element classes below the pad as
 * `tiered_element_class` requires; within the valid class `hi` then `lo` -- the sign-flipped
 * 128-bit value split most- then least-significant, complemented when descending -- give
 * unsigned-compare == the requested value order. Three fields pack to 24 bytes (a uint32 flag
 * beside two uint64 words), the width the warp tier's register / shared budget is sized against; a
 * null or pad leaves the value words zero and unread.
 */
struct tiered_key128 {
  cuda::std::uint64_t hi;
  cuda::std::uint64_t lo;
  cuda::std::uint32_t flag;
  __device__ bool operator<(tiered_key128 const& o) const
  {
    if (flag != o.flag) { return flag < o.flag; }
    if (hi != o.hi) { return hi < o.hi; }
    return lo < o.lo;
  }
};
static_assert(sizeof(tiered_key128) == 24 and alignof(tiered_key128) == 8,
              "tiered_key128 must stay 24 bytes with 8-byte alignment");

/**
 * @brief Maps a tiered storage type to its packed ordering key by value width
 *
 * The class flag rides the high bits so a native (unsigned / lexicographic) compare orders valid <
 * null < pad and then by value. A value of four bytes or fewer packs into a `uint64` (flag in bits
 * 32-33), an eight-byte value into an `unsigned __int128` (flag in bits 64-65), and the
 * sixteen-byte DECIMAL128 rep into the 24-byte `tiered_key128`. Selection is by `sizeof(T)`, so
 * every supported rep
 * -- int32 / int64, float / double, the int32 / int64-rep timestamps and durations, and the
 * DECIMAL128
 * `__int128` rep -- maps to the key of its width; the value bits are produced by the
 * `radix_encode_*` of the matching width, which carries the float (NaN / sign) and chrono
 * (rep-extract) transforms.
 */
template <typename T>
using tiered_key_t = cuda::std::conditional_t<
  sizeof(T) <= 4,
  cuda::std::uint64_t,
  cuda::std::conditional_t<sizeof(T) == 8, unsigned __int128, tiered_key128>>;

/**
 * @brief The pad key for type `T`: the class flag set to `tier_pad`, value words zero
 *
 * Ordered strictly after every valid element and null under the native / `operator<` compare, so
 * the network and `WarpMergeSort` keep the pad slots beyond a segment's valid-item boundary. Host-
 * and device-usable because the launch code builds it on the host and passes it to the kernel by
 * value.
 */
template <typename T>
__host__ __device__ inline tiered_key_t<T> tiered_pad_key()
{
  using KeyT = tiered_key_t<T>;
  if constexpr (cuda::std::is_same_v<KeyT, tiered_key128>) {
    return tiered_key128{0, 0, tier_pad};
  } else if constexpr (cuda::std::is_same_v<KeyT, unsigned __int128>) {
    return static_cast<unsigned __int128>(tier_pad) << 64;
  } else {
    return static_cast<cuda::std::uint64_t>(tier_pad) << 32;
  }
}

/**
 * @brief Strict-weak less-than over any tiered key: native `<` for the packed integer keys,
 * `tiered_key128::operator<` for the DECIMAL128 key. Comparator-generic, as the network and
 * `cub::WarpMergeSort` want.
 */
struct tiered_key_less {
  template <typename KeyT>
  __device__ bool operator()(KeyT const& a, KeyT const& b) const
  {
    return a < b;
  }
};

/**
 * @brief Builds the packed tiered key for one element: class flag then radix-encoded value
 *
 * The polarity's `element_class` assigns the null and valid classes (0/1) so a null -- its value
 * left zero and unread -- sorts on the requested side of every valid element in its segment
 * regardless of that element's value; a valid element carries its value through the same
 * order-preserving `radix_encode_*` the packed-radix path uses, complemented within the value bits
 * when descending (for the 128-bit key, complementing each word complements the whole value, and
 * the hi-then-lo compare equals the 128-bit unsigned compare), so an unsigned compare of the key
 * reproduces the requested value order. The complement never reaches the class flag: it is
 * confined to the value field below (or, for `tiered_key128`, to the separate value words).
 */
template <typename T>
struct tiered_key_builder {
  column_device_view d_input;
  bool has_nulls;
  sort_polarity polarity;
  __device__ tiered_key_t<T> operator()(size_type idx) const
  {
    using KeyT     = tiered_key_t<T>;
    bool const nul = has_nulls && d_input.is_null(idx);
    auto const cls = polarity.element_class(nul);
    if constexpr (cuda::std::is_same_v<KeyT, tiered_key128>) {
      if (nul) { return tiered_key128{0, 0, cls}; }
      auto const encoded = radix_encode_u128<T>(d_input.element<T>(idx));
      auto const mask    = polarity.value_mask64();
      return tiered_key128{static_cast<cuda::std::uint64_t>(encoded >> 64) ^ mask,
                           static_cast<cuda::std::uint64_t>(encoded) ^ mask,
                           cls};
    } else if constexpr (cuda::std::is_same_v<KeyT, unsigned __int128>) {
      if (nul) { return static_cast<unsigned __int128>(cls) << 64; }
      return (static_cast<unsigned __int128>(cls) << 64) |
             static_cast<unsigned __int128>(radix_encode_u64<T>(d_input.element<T>(idx)) ^
                                            polarity.value_mask64());
    } else {
      if (nul) { return static_cast<cuda::std::uint64_t>(cls) << 32; }
      return (static_cast<cuda::std::uint64_t>(cls) << 32) |
             static_cast<cuda::std::uint64_t>(radix_encode_u32<T>(d_input.element<T>(idx)) ^
                                              polarity.value_mask32());
    }
  }
};

// Network tier: one thread sorts a whole segment of <= `TIERED_NETWORK_CAP` elements with a fixed
// Batcher odd-even mergesort network, entirely in registers -- no shared memory, no warp
// cooperation. The 19-comparator network for eight keys sorts every one of the 2^8 binary inputs,
// so by the zero-one principle it sorts any totally-ordered keys, including the three-valued (flag,
// value) key. The widest key is 24 bytes, so a thread holds at most eight keys and eight indices in
// registers, which fits without spilling.
constexpr size_type TIERED_NETWORK_CAP = 8;
// The network kernel unrolls a fixed 19-comparator Batcher network over exactly eight register
// slots: raising the cap would leave slots the network never orders (silent mis-sort), lowering it
// would index past the register arrays. Regenerate the network before changing the cap.
static_assert(TIERED_NETWORK_CAP == 8,
              "tiered_network_sort_kernel hardcodes the eight-key Batcher network");
// Warp tier: a full 32-lane warp sorts a segment of <= `TIERED_WARP_CAP` elements with
// `cub::WarpMergeSort` at two items per lane. One 32*2 = 64 tile covers the class for every
// supported key width -- the register and shared budget stays comfortable even for the widest key
// -- so the cap is uniform rather than shrinking with the key width as a wider tile would force. A
// 128-thread block hosts four warp-segments.
constexpr int TIERED_WARP_LANES     = 32;
constexpr int TIERED_WARP_ITEMS     = 2;
constexpr size_type TIERED_WARP_CAP = TIERED_WARP_LANES * TIERED_WARP_ITEMS;  // 64

/**
 * @brief Selects segments whose size is <= `cap` (the network tier) for `cub::DevicePartition::If`
 */
struct segment_in_network_tier {
  size_type const* d_offsets;
  size_type cap;
  __device__ bool operator()(size_type seg) const
  {
    return (d_offsets[seg + 1] - d_offsets[seg]) <= cap;
  }
};

/**
 * @brief Selects segments whose size is in `(network_cap, warp_cap]` (the warp tier)
 *
 * The explicit lower bound keeps the predicate correct whether `DevicePartition::If` evaluates the
 * second selector on all items or only those the first selector rejected; segments above `warp_cap`
 * match neither selector and land in the unselected (radix-tier) partition.
 */
struct segment_in_warp_tier {
  size_type const* d_offsets;
  size_type network_cap;
  size_type warp_cap;
  __device__ bool operator()(size_type seg) const
  {
    auto const sz = d_offsets[seg + 1] - d_offsets[seg];
    return sz > network_cap && sz <= warp_cap;
  }
};

/**
 * @brief Selects elements whose segment is in the radix tier (size > `warp_cap`) for
 * `cub::DeviceSelect::If`
 *
 * Applied to a counting iterator over element indices, so it compacts the global indices of exactly
 * the radix-tier segments' elements -- and, because it scans in ascending index order, they come
 * out grouped by segment then by within-segment position, the arrangement
 * `radix_sort_large_segments` relies on to scatter each sorted element to its output slot.
 */
struct element_in_radix_tier {
  size_type const* d_segment_ids;
  size_type const* d_offsets;
  size_type warp_cap;
  __device__ bool operator()(size_type i) const
  {
    auto const seg = d_segment_ids[i];
    return (d_offsets[seg + 1] - d_offsets[seg]) > warp_cap;
  }
};

/**
 * @brief One compare-exchange of a sorting network: orders `keys[a] <= keys[b]` under `cmp`,
 * carrying each key's paired value with it. `cmp(keys[b], keys[a])` is true exactly when the pair
 * is inverted.
 */
template <typename KeyT, typename CompareOp>
__device__ inline void network_compare_exchange(
  KeyT* keys, size_type* vals, int a, int b, CompareOp cmp)
{
  if (cmp(keys[b], keys[a])) {
    KeyT const tk      = keys[a];
    keys[a]            = keys[b];
    keys[b]            = tk;
    size_type const tv = vals[a];
    vals[a]            = vals[b];
    vals[b]            = tv;
  }
}

/**
 * @brief Sorts one segment of <= `TIERED_NETWORK_CAP` elements per thread with a fixed
 * 19-comparator Batcher network
 *
 * Thread `t` owns segment `d_seg_list[t]`, loads its <= 8 elements' keys and global indices into
 * registers, fills the unused slots past the segment size with the pad key (which sorts after every
 * valid element and null), applies the network, then writes the sorted global indices back to
 * `[seg_start, seg_start + seg_size)`. The network sorts every one of the 2^8 binary inputs, so by
 * the zero-one principle it sorts the three-valued (flag, value) key for any values; the pads
 * settle into the top `8 - seg_size` slots and are never written back. Purely register-resident --
 * no shared memory and no warp cooperation -- so a segment's cost is independent of its
 * neighbours'.
 *
 * A function template so the two translation units that include this header (segmented_sort.cu and
 * stable_segmented_sort.cu) instantiate it only where used -- the STABLE unit never does, avoiding
 * an unused `static` kernel under -Werror.
 */
template <typename T, int BLOCK_THREADS, typename CompareOp>
CUDF_KERNEL __launch_bounds__(BLOCK_THREADS) void tiered_network_sort_kernel(
  column_device_view const d_input,
  size_type const* d_offsets,
  size_type const* d_seg_list,
  size_type num_class_segments,
  bool has_nulls,
  sort_polarity polarity,
  size_type* d_out,
  CompareOp compare_op,
  tiered_key_t<T> pad_key)
{
  using KeyT     = tiered_key_t<T>;
  auto const tid = cudf::detail::grid_1d::global_thread_id<BLOCK_THREADS>();
  if (tid >= static_cast<thread_index_type>(num_class_segments)) { return; }

  auto const seg       = d_seg_list[tid];
  auto const seg_start = d_offsets[seg];
  auto const seg_size  = d_offsets[seg + 1] - seg_start;

  tiered_key_builder<T> const build_key{d_input, has_nulls, polarity};
  KeyT keys[TIERED_NETWORK_CAP];
  size_type vals[TIERED_NETWORK_CAP];
#pragma unroll
  for (int i = 0; i < TIERED_NETWORK_CAP; ++i) {
    if (i < seg_size) {
      auto const gidx = seg_start + i;
      keys[i]         = build_key(gidx);
      vals[i]         = gidx;
    } else {
      keys[i] = pad_key;
      vals[i] = size_type{0};
    }
  }

  // Batcher odd-even mergesort for 8 keys (19 compare-exchanges). Each orders its lower index <=
  // its higher, so the pads settle above every real element and [0, seg_size) ends up sorted in the
  // key order the polarity encodes: the requested value order with nulls on their configured side.
  network_compare_exchange(keys, vals, 0, 1, compare_op);
  network_compare_exchange(keys, vals, 2, 3, compare_op);
  network_compare_exchange(keys, vals, 4, 5, compare_op);
  network_compare_exchange(keys, vals, 6, 7, compare_op);
  network_compare_exchange(keys, vals, 0, 2, compare_op);
  network_compare_exchange(keys, vals, 1, 3, compare_op);
  network_compare_exchange(keys, vals, 4, 6, compare_op);
  network_compare_exchange(keys, vals, 5, 7, compare_op);
  network_compare_exchange(keys, vals, 1, 2, compare_op);
  network_compare_exchange(keys, vals, 5, 6, compare_op);
  network_compare_exchange(keys, vals, 0, 4, compare_op);
  network_compare_exchange(keys, vals, 3, 7, compare_op);
  network_compare_exchange(keys, vals, 1, 5, compare_op);
  network_compare_exchange(keys, vals, 2, 6, compare_op);
  network_compare_exchange(keys, vals, 1, 4, compare_op);
  network_compare_exchange(keys, vals, 3, 6, compare_op);
  network_compare_exchange(keys, vals, 2, 4, compare_op);
  network_compare_exchange(keys, vals, 3, 5, compare_op);
  network_compare_exchange(keys, vals, 3, 4, compare_op);

#pragma unroll
  for (int i = 0; i < TIERED_NETWORK_CAP; ++i) {
    if (i < seg_size) { d_out[seg_start + i] = vals[i]; }
  }
}

/**
 * @brief Sorts one segment per virtual warp with `cub::WarpMergeSort` under a null-aware comparator
 *
 * Each virtual warp of `W` lanes owns one segment index from `d_seg_list` (a single size class).
 * Lane `l` holds items `[l*IPT, l*IPT+IPT)` of the segment in blocked order -- the arrangement
 * `BlockMergeSortStrategy` expects -- with slots past the segment's real size filled with the pad
 * key. The `valid_items = seg_size` boundary and that pad key make `WarpMergeSort` keep the real
 * elements -- in the polarity's key order -- in `[0, seg_size)` and the pads beyond it, so writing
 * back only `[0, seg_size)` yields the segment's sorted order. The sorted values -- global element
 * indices -- go straight into the output gather map at the segment's slots. `W*IPT` must be >= the
 * class's maximum segment size, guaranteed by the caps the caller partitions on.
 *
 * A function template for the same reason as `tiered_network_sort_kernel`: the STABLE translation
 * unit must not instantiate an unused `static` kernel under -Werror.
 */
template <typename T, int W, int IPT, int BLOCK_THREADS, typename CompareOp>
CUDF_KERNEL __launch_bounds__(BLOCK_THREADS) void tiered_warp_sort_kernel(
  column_device_view const d_input,
  size_type const* d_offsets,
  size_type const* d_seg_list,
  size_type num_class_segments,
  bool has_nulls,
  sort_polarity polarity,
  size_type* d_out,
  CompareOp compare_op,
  tiered_key_t<T> pad_key)
{
  using KeyT                     = tiered_key_t<T>;
  using WarpMergeSortT           = cub::WarpMergeSort<KeyT, IPT, W, size_type>;
  constexpr int VWARPS_PER_BLOCK = BLOCK_THREADS / W;
  __shared__ typename WarpMergeSortT::TempStorage temp_storage[VWARPS_PER_BLOCK];

  auto const global_tid = cudf::detail::grid_1d::global_thread_id<BLOCK_THREADS>();
  auto const vwarp_id   = static_cast<size_type>(global_tid / W);  // the class segment to sort
  auto const lane       = static_cast<int>(threadIdx.x % W);  // logical lane in the virtual warp
  auto const vwarp_slot = static_cast<int>(threadIdx.x / W);  // virtual warp's shared-mem slot
  // `vwarp_id` is identical across a virtual warp's W lanes, so the whole virtual warp returns
  // together and never desynchronizes the `__syncwarp(member_mask)` inside WarpMergeSort::Sort.
  if (vwarp_id >= num_class_segments) { return; }

  auto const seg       = d_seg_list[vwarp_id];
  auto const seg_start = d_offsets[seg];
  auto const seg_size  = d_offsets[seg + 1] - seg_start;

  tiered_key_builder<T> const build_key{d_input, has_nulls, polarity};
  KeyT keys[IPT];
  size_type vals[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    auto const local = lane * IPT + i;
    if (local < seg_size) {
      auto const gidx = seg_start + local;
      keys[i]         = build_key(gidx);
      vals[i]         = gidx;
    } else {
      keys[i] = pad_key;
      vals[i] = size_type{0};
    }
  }

  WarpMergeSortT(temp_storage[vwarp_slot])
    .Sort(keys, vals, compare_op, static_cast<int>(seg_size), pad_key);

#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    auto const local = lane * IPT + i;
    if (local < seg_size) { d_out[seg_start + local] = vals[i]; }
  }
}

// The warp tier routes a no-null INT32 / INT64 segment to a kernel measured faster than the packed
// `WarpMergeSort`: the value alone (no null field) fits a raw key half the packed key's width,
// cutting the sort's data traffic. INT32 (and the low sub-band of INT64) uses a register
// shuffle-bitonic; the high sub-band of INT64 a `WarpMergeSort` over that raw key. A null-bearing
// column keeps the packed three-valued key (a raw key cannot express null-ordering), so these apply
// only when `has_nulls` is false, guaranteed by the caller before a raw-key band launches.

/**
 * @brief Raw ordering key for the no-null warp tier: the order-preserving value alone, no null
 * field
 *
 * Four-byte-or-narrower values pack into a `uint32`, eight-byte into a `uint64` -- half the packed
 * `tiered_key_t` width at each, since a no-null segment needs no class flag. The `radix_encode_*`
 * of the matching width carries the same value transform the packed key uses, complemented when
 * descending, so an unsigned compare reproduces the requested value order.
 */
template <typename T>
using tiered_raw_key_t =
  cuda::std::conditional_t<sizeof(T) <= 4, cuda::std::uint32_t, cuda::std::uint64_t>;

/**
 * @brief Builds the raw no-null key for one element: its order-preserving encoded value,
 * complemented when descending
 */
template <typename T>
struct tiered_raw_key_builder {
  column_device_view d_input;
  sort_polarity polarity;
  __device__ tiered_raw_key_t<T> operator()(size_type idx) const
  {
    if constexpr (sizeof(T) <= 4) {
      return radix_encode_u32<T>(d_input.element<T>(idx)) ^ polarity.value_mask32();
    } else {
      return radix_encode_u64<T>(d_input.element<T>(idx)) ^ polarity.value_mask64();
    }
  }
};

/**
 * @brief The pad key for the raw no-null path: the maximum unsigned value, ordered after every real
 * element. A real element can encode to this same maximum (an `INT*_MAX` value ascending, or the
 * minimum under the descending complement), so the warp band uses the stability-guaranteed
 * `StableSort`: reals hold lower tile indices than pads, and a stable sort keeps an equal-valued
 * real ahead of the pads -- inside the valid range -- so no real element is dropped for a pad.
 */
template <typename T>
__host__ __device__ inline tiered_raw_key_t<T> tiered_raw_pad_key()
{
  return cuda::std::numeric_limits<tiered_raw_key_t<T>>::max();
}

/// Plain ascending less-than for the raw key -- no class flag, since the raw path is no-null only.
struct tiered_raw_less {
  template <typename KeyT>
  __device__ bool operator()(KeyT const& a, KeyT const& b) const
  {
    return a < b;
  }
};

/**
 * @brief Ascending compare of a raw-key/index pair with a pad tie-break for the register bitonic
 *
 * Compares `(key, is_pad)` where `is_pad == (val < 0)`: a pad carries `val = -1`, a real element
 * its non-negative global index. Ordering a real before a pad on equal keys keeps every real within
 * `[0, seg_size)` after the sort even when a real key equals the maximum pad sentinel (the
 * `INT*_MAX` element ascending, or the minimum element under the descending complement -- either
 * encodes to all-ones), so the write-back never drops it or emits a pad slot.
 */
template <typename KeyT>
__device__ inline bool bitonic_pad_less(KeyT ka, size_type va, KeyT kb, size_type vb)
{
  if (ka != kb) { return ka < kb; }
  return (va >= 0) && (vb < 0);
}

/**
 * @brief Clean-room register shuffle-bitonic sort of `W * IPT` elements across a logical warp
 *
 * The standard public-domain Batcher bitonic network (compile-time bounds unroll it to
 * straight-line code). Blocked layout: element index `e = lane * IPT + item`. Intra-lane
 * compare-exchanges
 * (`j < IPT`) run in registers; inter-lane ones (`j >= IPT`) exchange with lane `lane ^ (j / IPT)`
 * via `__shfl_xor_sync` confined to the logical warp by `mask` and width `W`. Each bitonic stage's
 * direction is `(e & k) == 0` (ascending) and the low index of a pair is the one with the `j` bit
 * clear, so each thread keeps the min or max consistently with its partner. The `(key, is_pad)`
 * compare orders pads (max key, negative index) after every real element. Validated exhaustively on
 * the host (zero-one principle) for the instantiated shapes.
 */
template <int W, int IPT, typename KeyT>
__device__ inline void bitonic_warp_sort(KeyT (&keys)[IPT],
                                         size_type (&vals)[IPT],
                                         int lane,
                                         unsigned mask)
{
  constexpr int n = W * IPT;
  for (int k = 2; k <= n; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      if (j >= IPT) {
        int const jl = j / IPT;
#pragma unroll
        for (int m = 0; m < IPT; ++m) {
          int const e          = lane * IPT + m;
          KeyT const pk        = __shfl_xor_sync(mask, keys[m], jl, W);
          size_type const pv   = __shfl_xor_sync(mask, vals[m], jl, W);
          bool const ascending = ((e & k) == 0);
          bool const keep_min  = (((e & j) == 0) == ascending);
          bool const take      = keep_min ? bitonic_pad_less(pk, pv, keys[m], vals[m])
                                          : bitonic_pad_less(keys[m], vals[m], pk, pv);
          if (take) {
            keys[m] = pk;
            vals[m] = pv;
          }
        }
      } else {
#pragma unroll
        for (int m = 0; m < IPT; ++m) {
          int const m2 = m ^ j;
          if (m2 > m) {
            int const e          = lane * IPT + m;
            bool const ascending = ((e & k) == 0);
            bool const inverted  = ascending
                                     ? bitonic_pad_less(keys[m2], vals[m2], keys[m], vals[m])
                                     : bitonic_pad_less(keys[m], vals[m], keys[m2], vals[m2]);
            if (inverted) {
              KeyT const tk      = keys[m];
              keys[m]            = keys[m2];
              keys[m2]           = tk;
              size_type const tv = vals[m];
              vals[m]            = vals[m2];
              vals[m2]           = tv;
            }
          }
        }
      }
    }
  }
}

/**
 * @brief Register shuffle-bitonic warp kernel over a segment-size band (raw keys, no-null case)
 *
 * One logical warp of `W` lanes sorts one segment of `<= W * IPT` elements whose size is in
 * `(band_lo, band_hi]`; slots past the segment are pads (max key, index -1) that the `(key,
 * is_pad)` tie-break keeps beyond `[0, seg_size)`. Template for the same -Werror reason as the
 * other tiered kernels.
 */
template <typename T, int W, int IPT, int BLOCK_THREADS>
CUDF_KERNEL __launch_bounds__(BLOCK_THREADS) void tiered_bitonic_band_kernel(
  column_device_view const d_input,
  size_type const* d_offsets,
  size_type const* d_seg_list,
  size_type num_class_segments,
  size_type band_lo,
  size_type band_hi,
  sort_polarity polarity,
  size_type* d_out)
{
  using KeyT            = tiered_raw_key_t<T>;
  auto const global_tid = cudf::detail::grid_1d::global_thread_id<BLOCK_THREADS>();
  auto const vwarp_id   = static_cast<size_type>(global_tid / W);
  auto const lane       = static_cast<int>(threadIdx.x % W);
  if (vwarp_id >= num_class_segments) { return; }

  auto const seg       = d_seg_list[vwarp_id];
  auto const seg_start = d_offsets[seg];
  auto const seg_size  = d_offsets[seg + 1] - seg_start;
  if (seg_size <= band_lo || seg_size > band_hi) { return; }
  // The register tile holds W*IPT elements; a band_hi above that would silently drop elements.
  cudf_assert(seg_size <= W * IPT && "band segment exceeds the register tile (band_hi > W*IPT)");

  tiered_raw_key_builder<T> const build_key{d_input, polarity};
  KeyT keys[IPT];
  size_type vals[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    auto const local = lane * IPT + i;
    if (local < seg_size) {
      auto const gidx = seg_start + local;
      keys[i]         = build_key(gidx);
      vals[i]         = gidx;
    } else {
      keys[i] = tiered_raw_pad_key<T>();
      vals[i] = size_type{-1};
    }
  }
  // Member mask of this virtual warp's W physical lanes (W <= 16 here, so the shift is
  // well-defined).
  unsigned const mask = ((1u << W) - 1u) << ((static_cast<unsigned>(threadIdx.x) % 32u / W) * W);
  bitonic_warp_sort<W, IPT>(keys, vals, lane, mask);
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    auto const local = lane * IPT + i;
    if (local < seg_size) { d_out[seg_start + local] = vals[i]; }
  }
}

/// Launches one packed-key `WarpMergeSort` band over the shared warp-segment list (all tiered
/// types)
template <typename T, int W, int IPT>
void launch_packed_warp_band(column_device_view const& d_input,
                             size_type const* d_offsets,
                             size_type const* d_warp_segs,
                             size_type num_warp,
                             bool has_nulls,
                             sort_polarity polarity,
                             size_type band_lo,
                             size_type band_hi,
                             size_type* d_out,
                             rmm::cuda_stream_view stream)
{
  if (num_warp == 0) { return; }
  using KeyT = tiered_key_t<T>;
  auto const grid =
    cudf::detail::grid_1d(static_cast<thread_index_type>(num_warp) * W, TIERED_BLOCK_THREADS);
  tiered_warp_band_kernel<KeyT,
                          tiered_key_builder<T>,
                          W,
                          IPT,
                          TIERED_BLOCK_THREADS,
                          tiered_key_less>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      d_offsets,
      d_warp_segs,
      num_warp,
      band_lo,
      band_hi,
      tiered_key_builder<T>{d_input, has_nulls, polarity},
      d_out,
      tiered_key_less{},
      tiered_pad_key<T>());
  CUDF_CHECK_CUDA(stream.value());
}

/// Launches one raw-key `WarpMergeSort` band over the shared warp-segment list (no-null only)
template <typename T, int W, int IPT>
void launch_raw_warp_band(column_device_view const& d_input,
                          size_type const* d_offsets,
                          size_type const* d_warp_segs,
                          size_type num_warp,
                          sort_polarity polarity,
                          size_type band_lo,
                          size_type band_hi,
                          size_type* d_out,
                          rmm::cuda_stream_view stream)
{
  if (num_warp == 0) { return; }
  using KeyT = tiered_raw_key_t<T>;
  auto const grid =
    cudf::detail::grid_1d(static_cast<thread_index_type>(num_warp) * W, TIERED_BLOCK_THREADS);
  tiered_warp_band_kernel<KeyT,
                          tiered_raw_key_builder<T>,
                          W,
                          IPT,
                          TIERED_BLOCK_THREADS,
                          tiered_raw_less>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      d_offsets,
      d_warp_segs,
      num_warp,
      band_lo,
      band_hi,
      tiered_raw_key_builder<T>{d_input, polarity},
      d_out,
      tiered_raw_less{},
      tiered_raw_pad_key<T>());
  CUDF_CHECK_CUDA(stream.value());
}

/// Launches one register-bitonic band over the shared warp-segment list (raw keys, no-null only)
template <typename T, int W, int IPT>
void launch_bitonic_band(column_device_view const& d_input,
                         size_type const* d_offsets,
                         size_type const* d_warp_segs,
                         size_type num_warp,
                         sort_polarity polarity,
                         size_type band_lo,
                         size_type band_hi,
                         size_type* d_out,
                         rmm::cuda_stream_view stream)
{
  if (num_warp == 0) { return; }
  auto const grid =
    cudf::detail::grid_1d(static_cast<thread_index_type>(num_warp) * W, TIERED_BLOCK_THREADS);
  tiered_bitonic_band_kernel<T, W, IPT, TIERED_BLOCK_THREADS>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      d_input, d_offsets, d_warp_segs, num_warp, band_lo, band_hi, polarity, d_out);
  CUDF_CHECK_CUDA(stream.value());
}

/**
 * @brief Run-time predicate for the tiered fast path on a key column's type
 *
 * True for the reps the path instantiates: int32 / int64 and the DECIMAL128 `__int128` rep, plus
 * FLOAT32 / FLOAT64 and every TIMESTAMP / DURATION (whose int32 / int64 rep flows through the same
 * width-keyed path via `radix_encode_*`). DECIMAL32/64, the narrower integrals, and the
 * non-fixed-width types are excluded and fall through to the paths below.
 *
 * @param type The key column's data type
 * @return true if the type is eligible for the tiered fast path
 */
inline bool is_tiered_sort_supported(data_type type)
{
  return type.id() == type_id::INT32 or type.id() == type_id::INT64 or
         type.id() == type_id::DECIMAL128 or cudf::is_floating_point(type) or cudf::is_chrono(type);
}

/**
 * @brief Sorts only the radix-tier segments' elements via the packed-radix key, scattering the
 * result into the output gather map
 *
 * The radix tier -- segments above the warp-tile cap -- is rare (typically the single oversized
 * segment the list generator forces), so sorting the whole column would waste a full-width global
 * radix on the many elements the network and warp tiers already handle. This instead keys just the
 * compacted radix-tier element indices with the same packed key the numeric packed-radix path uses
 * (segment ordinal, null flag, order-preserving value), radix-sorts that subset, and scatters each
 * sorted global index to the slot it fills. `d_large_gidx` holds those elements' global indices in
 * ascending order -- which are exactly the radix-tier output slots in order -- and the radix orders
 * the subset by (segment, value), so `d_out[d_large_gidx[j]] = sorted_gidx[j]` writes each such
 * segment's k-th smallest element to its k-th slot. `d_large_gidx` is the radix's value input and
 * is left unmodified so it doubles as the scatter map. Called only when a radix-tier segment
 * exists, over `num_large_elems`, not the whole column.
 */
template <typename T>
void radix_sort_large_segments(column_device_view const& d_input,
                               size_type const* d_segment_ids,
                               bool has_nulls,
                               sort_polarity polarity,
                               int segment_bits,
                               size_type const* d_large_gidx,
                               size_type num_large_elems,
                               size_type* d_out,
                               rmm::cuda_stream_view stream)
{
  auto const alloc = cudf::get_current_device_resource_ref();
  rmm::device_uvector<size_type> sorted_gidx(num_large_elems, stream);

  // Key each compacted radix-tier index directly (a global index is the builder's input), pair it
  // with that index, and radix-sort. The value input `d_large_gidx` is read-only, so it survives
  // for the scatter below. Each branch mirrors the width-specific key and byte-pass count of the
  // packed-radix path, restricted to the subset.
  if constexpr (sizeof(T) <= 4) {
    rmm::device_uvector<cuda::std::uint64_t> keys_in(num_large_elems, stream);
    rmm::device_uvector<cuda::std::uint64_t> keys_out(num_large_elems, stream);
    thrust::transform(
      rmm::exec_policy_nosync(stream, alloc),
      d_large_gidx,
      d_large_gidx + num_large_elems,
      keys_in.begin(),
      numeric_packed_key_builder<T>{d_segment_ids, d_input, has_nulls, segment_bits, polarity});
    rmm::device_buffer d_temp_storage;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                    temp_storage_bytes,
                                    keys_in.data(),
                                    keys_out.data(),
                                    d_large_gidx,
                                    sorted_gidx.data(),
                                    num_large_elems,
                                    0,
                                    static_cast<int>(sizeof(cuda::std::uint64_t) * 8),
                                    stream.value());
    d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
    cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                    temp_storage_bytes,
                                    keys_in.data(),
                                    keys_out.data(),
                                    d_large_gidx,
                                    sorted_gidx.data(),
                                    num_large_elems,
                                    0,
                                    static_cast<int>(sizeof(cuda::std::uint64_t) * 8),
                                    stream.value());
  } else if constexpr (sizeof(T) == 8) {
    rmm::device_uvector<prefix_key96> keys_in(num_large_elems, stream);
    rmm::device_uvector<prefix_key96> keys_out(num_large_elems, stream);
    thrust::transform(rmm::exec_policy_nosync(stream, alloc),
                      d_large_gidx,
                      d_large_gidx + num_large_elems,
                      keys_in.begin(),
                      numeric_packed_key_builder64<T>{d_segment_ids, d_input, has_nulls, polarity});
    auto const decomposer = prefix_decomposer{};
    // Same trim as the full-column 8-byte branch: seg_null's bits above segment_bits + 1 are
    // constant zero, so the radix skips their passes (the dec128 key96 discipline).
    auto const key_bits = cuda::std::min(static_cast<int>(sizeof(prefix_key96) * 8),
                                         64 + cuda::std::min(32, segment_bits + 1));
    rmm::device_buffer d_temp_storage;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                    temp_storage_bytes,
                                    keys_in.data(),
                                    keys_out.data(),
                                    d_large_gidx,
                                    sorted_gidx.data(),
                                    num_large_elems,
                                    decomposer,
                                    0,
                                    key_bits,
                                    stream.value());
    d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
    cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                    temp_storage_bytes,
                                    keys_in.data(),
                                    keys_out.data(),
                                    d_large_gidx,
                                    sorted_gidx.data(),
                                    num_large_elems,
                                    decomposer,
                                    0,
                                    key_bits,
                                    stream.value());
  } else {
    // Sixteen-byte value (the DECIMAL128 rep): pick the narrowest lossless radix key from the range
    // over the compacted radix-tier indices, instead of an unconditional twenty-pass 160-bit key.
    // `d_large_gidx` is the key-build source and paired value; `sorted_gidx` receives the subset's
    // sorted order, which the scatter below places into the output.
    dec128_segmented_radix_sort(d_input,
                                d_segment_ids,
                                has_nulls,
                                polarity,
                                segment_bits,
                                d_large_gidx,
                                num_large_elems,
                                sorted_gidx.data(),
                                stream);
  }

  // The j-th compacted (ascending) radix-tier index is the j-th radix-tier output slot;
  // sorted_gidx[j] is the element that belongs there, so scatter writes d_out[d_large_gidx[j]] =
  // sorted_gidx[j].
  thrust::scatter(rmm::exec_policy_nosync(stream, alloc),
                  sorted_gidx.begin(),
                  sorted_gidx.end(),
                  d_large_gidx,
                  d_out);
}

/**
 * @brief Type-dispatched worker for `fast_segmented_sorted_order_tiered`
 *
 * Classifies the segments once with a three-way `cub::DevicePartition::If` (network / warp / radix
 * tier), reads the two selected counts back to the host, then launches the Batcher-network kernel
 * for the network class and the full-warp `WarpMergeSort` kernel for the warp class. Any radix-tier
 * outliers are sorted by a packed radix over just their own elements and scattered into place
 * (`radix_sort_large_segments`), which adds a second host sync only when such a segment exists. The
 * enabled overload matches the reps `is_tiered_sort_supported` admits.
 */
struct tiered_sort_fn {
  template <typename T>
  static constexpr bool is_supported()
  {
    return cuda::std::is_same_v<T, int32_t> or cuda::std::is_same_v<T, int64_t> or
           cuda::std::is_same_v<T, __int128_t> or cudf::is_floating_point<T>() or
           cudf::is_chrono<T>();
  }

  template <typename T, CUDF_ENABLE_IF(is_supported<T>())>
  void operator()(column_view const& segment_offsets,
                  column_device_view const& d_input,
                  size_type num_elements,
                  size_type num_segments,
                  bool has_nulls,
                  sort_polarity polarity,
                  size_type* d_out,
                  rmm::cuda_stream_view stream) const
  {
    auto const d_offsets = segment_offsets.begin<size_type>();

    // Three-way classify the segment indices by size: network (<= TIERED_NETWORK_CAP), warp (<=
    // TIERED_WARP_CAP), unselected (radix-tier outliers). `large_segs` is the partition's required
    // third sink; the radix tier is driven off element size (below), not this list, so it is
    // written but never read.
    rmm::device_uvector<size_type> network_segs(num_segments, stream);
    rmm::device_uvector<size_type> warp_segs(num_segments, stream);
    rmm::device_uvector<size_type> large_segs(num_segments, stream);
    rmm::device_uvector<size_type> d_counts(2, stream);
    auto const seg_iter       = cuda::counting_iterator<size_type>{0};
    auto const select_network = segment_in_network_tier{d_offsets, TIERED_NETWORK_CAP};
    auto const select_warp = segment_in_warp_tier{d_offsets, TIERED_NETWORK_CAP, TIERED_WARP_CAP};
    {
      rmm::device_buffer d_temp_storage;
      size_t temp_storage_bytes = 0;
      cub::DevicePartition::If(d_temp_storage.data(),
                               temp_storage_bytes,
                               seg_iter,
                               network_segs.data(),
                               warp_segs.data(),
                               large_segs.data(),
                               d_counts.data(),
                               num_segments,
                               select_network,
                               select_warp,
                               stream.value());
      d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
      cub::DevicePartition::If(d_temp_storage.data(),
                               temp_storage_bytes,
                               seg_iter,
                               network_segs.data(),
                               warp_segs.data(),
                               large_segs.data(),
                               d_counts.data(),
                               num_segments,
                               select_network,
                               select_warp,
                               stream.value());
    }
    auto const h_counts =
      cudf::detail::make_host_vector(device_span<size_type const>{d_counts.data(), 2}, stream);
    auto const num_network = h_counts[0];
    auto const num_warp    = h_counts[1];
    auto const num_large   = num_segments - num_network - num_warp;

    // Radix-tier outliers: sort only their elements with the packed-radix key and scatter them into
    // their (disjoint) output slots; the network and warp tiers fill the rest, so every slot is
    // written exactly once. When no segment is in the radix tier the whole block is skipped -- no
    // labeling, compaction, radix, or extra host sync -- so the tiny-segment common case pays
    // nothing for a tier it does not use.
    if (num_large > 0) {
      rmm::device_uvector<size_type> segment_ids(num_elements, stream);
      label_segments(segment_offsets.begin<size_type>(),
                     segment_offsets.end<size_type>(),
                     segment_ids.begin(),
                     segment_ids.end(),
                     stream);
      auto const segment_bits =
        static_cast<int>(cuda::std::bit_width(static_cast<cuda::std::uint64_t>(num_segments)));

      // Compact the global indices of the radix-tier segments' elements (ascending, hence grouped
      // by segment then position), then read their count back -- the one extra sync the radix tier
      // adds.
      rmm::device_uvector<size_type> large_gidx(num_elements, stream);
      rmm::device_uvector<size_type> d_num_large_elems(1, stream);
      {
        rmm::device_buffer d_temp_storage;
        size_t temp_storage_bytes = 0;
        auto const is_large = element_in_radix_tier{segment_ids.data(), d_offsets, TIERED_WARP_CAP};
        cub::DeviceSelect::If(d_temp_storage.data(),
                              temp_storage_bytes,
                              seg_iter,
                              large_gidx.data(),
                              d_num_large_elems.data(),
                              num_elements,
                              is_large,
                              stream.value());
        d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
        cub::DeviceSelect::If(d_temp_storage.data(),
                              temp_storage_bytes,
                              seg_iter,
                              large_gidx.data(),
                              d_num_large_elems.data(),
                              num_elements,
                              is_large,
                              stream.value());
      }
      auto const h_num_large_elems = cudf::detail::make_host_vector(
        device_span<size_type const>{d_num_large_elems.data(), 1}, stream);
      radix_sort_large_segments<T>(d_input,
                                   segment_ids.data(),
                                   has_nulls,
                                   polarity,
                                   segment_bits,
                                   large_gidx.data(),
                                   h_num_large_elems[0],
                                   d_out,
                                   stream);
    }

    auto const pad_key = tiered_pad_key<T>();
    if (num_network > 0) {
      auto const grid =
        cudf::detail::grid_1d(static_cast<thread_index_type>(num_network), TIERED_BLOCK_THREADS);
      tiered_network_sort_kernel<T, TIERED_BLOCK_THREADS>
        <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(d_input,
                                                                             d_offsets,
                                                                             network_segs.data(),
                                                                             num_network,
                                                                             has_nulls,
                                                                             polarity,
                                                                             d_out,
                                                                             tiered_key_less{},
                                                                             pad_key);
      CUDF_CHECK_CUDA(stream.value());
    }
    // Warp tier: segment-size sub-bands routed to the kernel measured best for this (type,
    // null-presence). Each band launches over the full warp-segment list and self-filters to its
    // size slice, so the sub-bands need no re-partition. No-null INT32 uses a register bitonic
    // across the whole band; no-null INT64 a bitonic to 32 then a raw-key WarpMergeSort for 33-64
    // (the raw 8-byte value halves the key traffic vs the packed key at that width); a null-bearing
    // INT32 / INT64 a full-warp packed-key WarpMergeSort at one item per lane to 32 then two to 64.
    // Every other tiered type (float, chrono, DECIMAL128) keeps the shipped whole-band packed
    // WarpMergeSort
    // -- the raw-key / bitonic kernels are unmeasured there, so those types stay on the proven
    // path.
    if (num_warp > 0) {
      auto const* wl = warp_segs.data();
      if constexpr (cuda::std::is_same_v<T, int32_t> or cuda::std::is_same_v<T, int64_t>) {
        if (has_nulls) {
          launch_packed_warp_band<T, TIERED_WARP_LANES, 1>(d_input,
                                                           d_offsets,
                                                           wl,
                                                           num_warp,
                                                           has_nulls,
                                                           polarity,
                                                           TIERED_NETWORK_CAP,
                                                           32,
                                                           d_out,
                                                           stream);
          launch_packed_warp_band<T, TIERED_WARP_LANES, TIERED_WARP_ITEMS>(d_input,
                                                                           d_offsets,
                                                                           wl,
                                                                           num_warp,
                                                                           has_nulls,
                                                                           polarity,
                                                                           32,
                                                                           TIERED_WARP_CAP,
                                                                           d_out,
                                                                           stream);
        } else {
          launch_bitonic_band<T, 4, 4>(
            d_input, d_offsets, wl, num_warp, polarity, TIERED_NETWORK_CAP, 16, d_out, stream);
          launch_bitonic_band<T, 8, 4>(
            d_input, d_offsets, wl, num_warp, polarity, 16, 32, d_out, stream);
          if constexpr (cuda::std::is_same_v<T, int32_t>) {
            launch_bitonic_band<T, 16, 4>(
              d_input, d_offsets, wl, num_warp, polarity, 32, TIERED_WARP_CAP, d_out, stream);
          } else {
            launch_raw_warp_band<T, TIERED_WARP_LANES, TIERED_WARP_ITEMS>(
              d_input, d_offsets, wl, num_warp, polarity, 32, TIERED_WARP_CAP, d_out, stream);
          }
        }
      } else {
        auto const grid = cudf::detail::grid_1d(
          static_cast<thread_index_type>(num_warp) * TIERED_WARP_LANES, TIERED_BLOCK_THREADS);
        tiered_warp_sort_kernel<T, TIERED_WARP_LANES, TIERED_WARP_ITEMS, TIERED_BLOCK_THREADS>
          <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(d_input,
                                                                               d_offsets,
                                                                               wl,
                                                                               num_warp,
                                                                               has_nulls,
                                                                               polarity,
                                                                               d_out,
                                                                               tiered_key_less{},
                                                                               pad_key);
        CUDF_CHECK_CUDA(stream.value());
      }
    }
  }

  template <typename T, CUDF_ENABLE_IF(not is_supported<T>())>
  void operator()(column_view const&,
                  column_device_view const&,
                  size_type,
                  size_type,
                  bool,
                  sort_polarity,
                  size_type*,
                  rmm::cuda_stream_view) const
  {
    CUDF_FAIL("Column type cannot be used with the tiered segmented sort");
  }
};

// Measured crossover below which the tiered kernel beats CUB and the packed radix for every type.
constexpr size_type TIERED_MAX_TINY_AVG_LIST_SIZE{4};

// Measured top of the DECIMAL128 no-null CUB band; above it the tiered kernel wins.
constexpr size_type DECIMAL128_CUB_MAX_AVG_LIST_SIZE{16};

// Measured cap of CUB's 16-byte-pair merge tile; longer segments fall to a costly per-block radix.
constexpr size_type DECIMAL128_CUB_MAX_SEGMENT_SIZE{32};

/**
 * @brief Whether a `DECIMAL128` mid-band column's segment shape favors CUB `DeviceSegmentedSort`
 *
 * CUB degrades on long segments, so the mid band takes it only when long segments are sparse: the
 * count of segments longer than `DECIMAL128_CUB_MAX_SEGMENT_SIZE`, scaled by that length, must stay
 * within the segment count. Runs a single device count -- and so adds one host synchronization --
 * only on the narrow `DECIMAL128` mid-band branch that calls it. The product is formed in 64-bit to
 * avoid overflow when many segments are long.
 *
 * @param segment_offsets The segment offsets (segment count plus one)
 * @param stream CUDA stream used for the device count
 * @return true if the segment shape favors CUB `DeviceSegmentedSort`
 */
inline bool decimal128_cub_segment_shape_ok(column_view const& segment_offsets,
                                            rmm::cuda_stream_view stream)
{
  auto const num_segments = segment_offsets.size() - 1;
  auto const d_offsets    = segment_offsets.begin<size_type>();
  auto const oversized =
    thrust::count_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<size_type>{0},
                     cuda::counting_iterator<size_type>{num_segments},
                     segment_exceeds_size{d_offsets, DECIMAL128_CUB_MAX_SEGMENT_SIZE});
  return static_cast<int64_t>(oversized) * DECIMAL128_CUB_MAX_SEGMENT_SIZE <= num_segments;
}

}  // namespace

[[nodiscard]] std::unique_ptr<column> fast_segmented_sorted_order_numeric_packed(
  column_view const& input,
  column_view const& segment_offsets,
  sort_polarity polarity,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_elements = input.size();
  auto const num_segments = segment_offsets.size() - 1;

  // Dense segment ordinals (0, 1, ... across the segments), exactly as the strings path labels
  // them: the ordinal maximum is num_segments - 1, so the key's segment field needs only
  // bit_width(num_segments) bits, and an empty segment simply contributes no element and skips a
  // label, preserving cross-segment order.
  rmm::device_uvector<size_type> segment_ids(num_elements, stream);
  label_segments(segment_offsets.begin<size_type>(),
                 segment_offsets.end<size_type>(),
                 segment_ids.begin(),
                 segment_ids.end(),
                 stream);

  auto const d_input   = column_device_view::create(input, stream);
  auto const has_nulls = input.has_nulls();
  auto const segment_bits =
    static_cast<int>(cuda::std::bit_width(static_cast<cuda::std::uint64_t>(num_segments)));

  auto sorted_indices = cudf::make_numeric_column(
    data_type{type_to_id<size_type>()}, num_elements, mask_state::UNALLOCATED, stream, mr);

  // Dispatch on the storage type so DECIMAL32/DECIMAL64 sort by their int32/int64 rep; the functor
  // picks the uint64 or prefix_key96 key by width and writes the sorted indices into the output.
  cudf::type_dispatcher<dispatch_storage_type>(input.type(),
                                               numeric_packed_sort_fn{},
                                               *d_input,
                                               segment_ids.data(),
                                               has_nulls,
                                               polarity,
                                               segment_bits,
                                               num_elements,
                                               sorted_indices->mutable_view().begin<size_type>(),
                                               stream);
  return sorted_indices;
}

[[nodiscard]] std::unique_ptr<column> fast_segmented_sorted_order_tiered(
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

  auto sorted_indices = cudf::make_numeric_column(
    data_type{type_to_id<size_type>()}, num_elements, mask_state::UNALLOCATED, stream, mr);

  // Dispatch on the storage type so DECIMAL128 sorts by its __int128 rep and DECIMAL32/64,
  // timestamps, and durations by their integer rep; the functor picks the uint64, unsigned
  // __int128, or tiered_key128 key by width and writes the sorted indices into d_out.
  cudf::type_dispatcher<dispatch_storage_type>(input.type(),
                                               tiered_sort_fn{},
                                               segment_offsets,
                                               *d_input,
                                               num_elements,
                                               num_segments,
                                               has_nulls,
                                               polarity,
                                               sorted_indices->mutable_view().begin<size_type>(),
                                               stream);
  return sorted_indices;
}

fixed_width_sort_path choose_fixed_width_sort_path(column_view const& key,
                                                   size_type num_rows,
                                                   column_view const& segment_offsets,
                                                   rmm::cuda_stream_view stream)
{
  auto const type = key.type();
  // String / nested keys have no fixed-width fast path.
  if (not is_numeric_packed_radix_supported(type)) { return fixed_width_sort_path::comparison; }

  // avg_list_size uses num_rows / num_offsets, the heuristic prefer_cub_segmented_sort also uses.
  auto const num_offsets   = segment_offsets.size();
  auto const avg_list_size = num_rows / num_offsets;

  // Types the tiered key can't encode: keep main's packed-radix-or-CUB decision.
  if (not is_tiered_sort_supported(type)) {
    return (key.has_nulls() or not prefer_cub_segmented_sort(num_rows, num_offsets))
             ? fixed_width_sort_path::packed_radix
             : fixed_width_sort_path::comparison;
  }
  // Tiered sort folds validity into the key, so nulls need no separate pass, at any list size.
  if (key.has_nulls()) { return fixed_width_sort_path::tiered; }
  // Long lists amortize the global packed-key radix's bandwidth-bound pass.
  if (avg_list_size >= MAX_AVG_LIST_SIZE_FOR_FAST_SORT) {
    return fixed_width_sort_path::packed_radix;
  }

  // DECIMAL128 no-null: tiny -> tiered, sparse-large mid band -> lifted CUB, longer -> tiered.
  if (type.id() == type_id::DECIMAL128) {
    if (avg_list_size <= TIERED_MAX_TINY_AVG_LIST_SIZE) { return fixed_width_sort_path::tiered; }
    if (avg_list_size <= DECIMAL128_CUB_MAX_AVG_LIST_SIZE and
        decimal128_cub_segment_shape_ok(segment_offsets, stream)) {
      return fixed_width_sort_path::cub_segmented;
    }
    return fixed_width_sort_path::tiered;
  }
  // Floating point no-null short: tiered across the range.
  if (cudf::is_floating_point(type)) { return fixed_width_sort_path::tiered; }
  // Eight-byte no-null: tiered across the range. The tiered warp kernels (register bitonic to 32,
  // raw-key `WarpMergeSort` for 33-64) beat CUB `DeviceSegmentedSort` for INT64 mid-band lists, so
  // INT64 no longer prefers CUB there. Chrono (and any non-integral 8-byte rep) must stay on tiered
  // regardless: `column_fast_sort_fn` (the CUB engine) sorts only integral reps --
  // `dispatch_storage_type` does not reduce a timestamp/duration to its integer rep -- so routing
  // them to `cub_segmented` would hit that engine's disabled overload (`CUDF_FAIL`), a regression
  // from main's comparison path.
  if (cudf::size_of(type) == 8) { return fixed_width_sort_path::tiered; }
  // Four-byte int/chrono no-null short: tiered.
  return fixed_width_sort_path::tiered;
}

}  // namespace detail
}  // namespace cudf
