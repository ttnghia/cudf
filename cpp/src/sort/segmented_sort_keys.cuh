/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

#include <cuda/std/cstdint>
#include <cuda/std/tuple>

namespace cudf {
namespace detail {

// Class of an element within the tiered ordering key. The ranks are load-bearing: classes 0 and 1
// go to valid-then-null under the nulls-last polarity and null-then-valid under nulls-first (via
// `sort_polarity::element_class`), and both sort before the pad the network / `WarpMergeSort`
// assigns to the slots past a segment's real elements. `tier_pad` MUST rank strictly above both
// element classes: `cub::BlockMergeSortStrategy::Sort` fills those slots with the pad key and
// requires it ordered after every valid item, so a flag value tying an element class with the pad
// could let the merge displace a real element past the valid-item boundary and drop it.
enum tiered_element_class : cuda::std::uint32_t { tier_valid = 0, tier_null = 1, tier_pad = 2 };

/// Block size hosting the register/warp tiered virtual warps and the graduated-string warp bands.
constexpr int TIERED_BLOCK_THREADS = 128;

/**
 * @brief Runtime key polarity realizing one explicit (order, null_order) on the segmented-sort
 * fast paths
 *
 * Every fast-path engine orders elements by an unsigned comparison of a packed key, so the
 * requested configuration is folded into the key bits themselves -- one XOR-class operation per
 * element -- rather than into per-engine comparators, which would multiply kernel instantiations.
 * `descending` complements the encoded value field: the complement is a strictly order-reversing
 * bijection on the field's unsigned range, and confining the mask to the value field's exact width
 * leaves the class/segment bits above it untouched, so within-segment value order reverses while
 * everything else is preserved. `nulls_first` picks the null class bit so nulls sort on the
 * requested side of a segment's valid elements. cudf's `null_order` is comparator-level -- a
 * DESCENDING sort swaps the comparison operands, inverting null placement too -- so the caller
 * resolves `nulls_first = (null_order == BEFORE) XOR descending` and relaxes a zero-null column to
 * the shipped nulls-last polarity. The only current caller constructs the default polarity, gated
 * to an explicit ascending / nulls-after request; the non-default states are exercised once that
 * gate widens to the full matrix. The default `{false, false}` state reproduces the shipped
 * ascending / nulls-last keys bit for bit.
 */
struct sort_polarity {
  bool descending  = false;
  bool nulls_first = false;

  /// Class bit for a valid (0/1) or null (1/0) element: unsigned key order then places nulls on
  /// the requested side of every valid element; the tiered pad class (2) stays strictly above both.
  __host__ __device__ cuda::std::uint32_t element_class(bool is_null) const
  {
    return static_cast<cuda::std::uint32_t>(is_null != nulls_first);
  }
  /// XOR mask reversing a 32-bit encoded value's unsigned order when descending.
  __host__ __device__ cuda::std::uint32_t value_mask32() const
  {
    return descending ? ~cuda::std::uint32_t{0} : cuda::std::uint32_t{0};
  }
  /// XOR mask reversing a 64-bit encoded value's unsigned order when descending.
  __host__ __device__ cuda::std::uint64_t value_mask64() const
  {
    return descending ? ~cuda::std::uint64_t{0} : cuda::std::uint64_t{0};
  }
};

/**
 * @brief Fixed-width radix key ordering runs still tied after the single-`uint64` first pass
 *
 * Twelve bytes with no padding, so one global radix decomposes it to exactly 96 bits (12 passes)
 * while still carrying a full eight-byte window; key storage size is the cost driver for this sort,
 * so dropping the four padding bytes a 16-byte key would carry is a direct data-movement win. The
 * first sort uses the single-`uint64` `packed_key_layout` instead; this key drives only the
 * iterative passes that reorder runs still tied after it. The fields are ordered most- to
 * least-significant. `seg_null` packs a dense run rank in its high 31 bits and the null flag in bit
 * 0 (`(rank << 1) | is_null`): the rank dominates so a radix by it preserves every order an earlier
 * pass resolved and a pass only reorders within a tied run, and within a rank a null (bit 0 = 1)
 * sorts after every non-null (bit 0 = 0). `prefix_hi` and `prefix_lo` are the most- and
 * least-significant four bytes of the element's next eight window bytes packed big-endian (byte 0
 * most significant, low bytes zero-filled); `prefix_hi` is the more significant field, so comparing
 * `prefix_hi` then `prefix_lo` reproduces unsigned-byte lexicographic order over the window. A null
 * element carries unread zero window words, never its string bytes.
 */
struct prefix_key96 {
  cuda::std::uint32_t seg_null;
  cuda::std::uint32_t prefix_hi;
  cuda::std::uint32_t prefix_lo;
};

// Twelve tightly-packed bytes is the whole point: it decomposes to 96 radix bits (12 byte-passes),
// and padding to sixteen would restore the four data-movement bytes this key exists to shed.
static_assert(sizeof(prefix_key96) == 12 and alignof(prefix_key96) == 4,
              "prefix_key96 must stay 12 bytes with 4-byte alignment");

/**
 * @brief Decomposes a `prefix_key96` into its fields for `cub::DeviceRadixSort`
 *
 * The leftmost tuple element is the most significant, so one ascending radix pass sorts by the
 * packed run-rank-and-null field, then by the high window word, then by the low window word -- i.e.
 * by run rank, then null flag, then the eight-byte window in unsigned-byte lexicographic order.
 */
struct prefix_decomposer {
  __device__ cuda::std::tuple<cuda::std::uint32_t&, cuda::std::uint32_t&, cuda::std::uint32_t&>
  operator()(prefix_key96& key) const
  {
    return {key.seg_null, key.prefix_hi, key.prefix_lo};
  }
};

/**
 * @brief True when two keys are bit-for-bit equal, i.e. the same run of an unresolved tie
 */
__device__ inline bool keys_equal(prefix_key96 const& a, prefix_key96 const& b)
{
  return a.seg_null == b.seg_null && a.prefix_hi == b.prefix_hi && a.prefix_lo == b.prefix_lo;
}

/**
 * @brief Packs a dense run rank and a null flag into one 32-bit field
 *
 * The rank occupies bits 1..31 and the null flag bit 0, so the rank dominates the ordering and a
 * null sorts after a non-null sharing the rank (null_order::AFTER). The run rank comes from an
 * inclusive scan over run-head flags, so it is bounded by the element count (<= 2^31 - 1, a
 * `size_type`); `(rank << 1) | flag` is then at most 2^32 - 1 and never overflows the field. A
 * `static_assert` in the caller proves this for every `size_type` value.
 */
__device__ inline cuda::std::uint32_t pack_seg_null(cuda::std::uint32_t label,
                                                    cuda::std::uint32_t flag)
{
  return (label << 1) | flag;
}

/**
 * @brief Splits a big-endian-packed eight-byte window into a `prefix_key96`'s two window words
 *
 * `prefix_hi` takes the most-significant four bytes and `prefix_lo` the least-significant four, so
 * comparing `prefix_hi` then `prefix_lo` reproduces an unsigned comparison of the full eight bytes.
 */
__device__ inline void split_prefix(cuda::std::uint64_t packed,
                                    cuda::std::uint32_t& prefix_hi,
                                    cuda::std::uint32_t& prefix_lo)
{
  prefix_hi = static_cast<cuda::std::uint32_t>(packed >> 32);
  prefix_lo = static_cast<cuda::std::uint32_t>(packed & 0xFFFF'FFFFu);
}

}  // namespace detail
}  // namespace cudf
