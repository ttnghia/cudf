/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "segmented_sort_fast.cuh"
#include "segmented_sort_keys.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <thrust/sequence.h>
#include <thrust/transform.h>

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
  // DECIMAL32/64 storage reps (int32/int64). The non-fixed-width types (string, list, struct,
  // dictionary) take the failing overload; the 16-byte DECIMAL128 rep is fenced off at run time.
  template <typename T>
  static constexpr bool is_supported()
  {
    return cudf::is_integral<T>() or cudf::is_floating_point<T>() or cudf::is_chrono<T>();
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
      // The sixteen-byte DECIMAL128 (`__int128`) rep is out of this stage's scope: the run-time
      // gate `is_numeric_packed_radix_supported` excludes it, so this branch is an unreached
      // fail-safe until the DECIMAL128 engine lands.
      CUDF_FAIL("Column type cannot be used with the numeric packed-radix segmented sort");
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
 * True for every fixed-width type the path encodes losslessly: integrals (including `bool`), the
 * `DECIMAL32/64` fixed-point reps, floating point, and timestamps/durations. `DECIMAL128` (16-byte
 * rep) and the non-fixed-width types (string, list, struct, dictionary) are excluded.
 *
 * @param type The key column's data type
 * @return true if the type is eligible for the packed-radix fast path
 */
inline bool is_numeric_packed_radix_supported(data_type type)
{
  return cudf::is_integral(type) or
         (cudf::is_fixed_point(type) and type.id() != type_id::DECIMAL128) or
         cudf::is_floating_point(type) or cudf::is_chrono(type);
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

fixed_width_sort_path choose_fixed_width_sort_path(column_view const& key,
                                                   size_type num_rows,
                                                   column_view const& segment_offsets,
                                                   [[maybe_unused]] rmm::cuda_stream_view stream)
{
  auto const type = key.type();
  // String / nested keys have no fixed-width fast path.
  if (not is_numeric_packed_radix_supported(type)) { return fixed_width_sort_path::comparison; }

  // avg_list_size uses num_rows / num_offsets, the heuristic prefer_cub_segmented_sort also uses.
  auto const num_offsets   = segment_offsets.size();
  auto const avg_list_size = num_rows / num_offsets;

  // Validity folds into the packed key, so a null-bearing column skips the separate null partition
  // at any list size.
  if (key.has_nulls()) { return fixed_width_sort_path::packed_radix; }
  // Long lists amortize the global packed-key radix's bandwidth-bound pass.
  if (avg_list_size >= MAX_AVG_LIST_SIZE_FOR_FAST_SORT) {
    return fixed_width_sort_path::packed_radix;
  }
  // The short no-null remainder keeps the CUB-or-comparison decision of the gate below.
  return fixed_width_sort_path::comparison;
}

}  // namespace detail
}  // namespace cudf
