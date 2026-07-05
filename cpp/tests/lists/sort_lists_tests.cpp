/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/lists/sorting.hpp>

#include <algorithm>
#include <limits>
#include <vector>

using namespace cudf::test::iterators;

template <typename T>
using LCW = cudf::test::lists_column_wrapper<T, int32_t>;

auto generate_sorted_lists(cudf::lists_column_view const& input,
                           cudf::order column_order,
                           cudf::null_order null_precedence)
{
  return std::pair{cudf::lists::sort_lists(input, column_order, null_precedence),
                   cudf::lists::stable_sort_lists(input, column_order, null_precedence)};
}

template <typename T>
struct SortLists : public cudf::test::BaseFixture {};

using TypesForTest = cudf::test::Concat<cudf::test::NumericTypes, cudf::test::FixedPointTypes>;
TYPED_TEST_SUITE(SortLists, TypesForTest);

TYPED_TEST(SortLists, NoNull)
{
  using T = TypeParam;

  // List<T>
  LCW<T> list{{3, 2, 1, 4}, {5}, {10, 8, 9}, {6, 7}};

  // Ascending
  // LCW<int>  order{{2, 1, 0, 3}, {0}, {1, 2, 0},  {0, 1}};
  LCW<T> expected{{1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}};
  {
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::ASCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
  {
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::ASCENDING, cudf::null_order::BEFORE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  // Descending
  // LCW<int>  order{{3, 0, 1, 2}, {0}, {0, 1, 2},  {1, 0}};
  LCW<T> expected2{{4, 3, 2, 1}, {5}, {10, 9, 8}, {7, 6}};
  {
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::DESCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected2);
  }
  {
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::DESCENDING, cudf::null_order::BEFORE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected2);
  }
}

TYPED_TEST(SortLists, Null)
{
  using T = TypeParam;
  if (std::is_same_v<T, bool>) return;
  std::vector<bool> valids_o{true, true, false, true};
  std::vector<bool> valids_a{true, true, true, false};
  std::vector<bool> valids_b{false, true, true, true};

  // List<T>
  LCW<T> list{{{3, 2, 4, 1}, valids_o.begin()}, {5}, {10, 8, 9}, {6, 7}};
  // LCW<int>  order{{2, 1, 3, 0}, {0}, {1, 2, 0},  {0, 1}};

  {
    LCW<T> expected{{{1, 2, 3, 4}, valids_a.begin()}, {5}, {8, 9, 10}, {6, 7}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::ASCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    LCW<T> expected{{{4, 1, 2, 3}, valids_b.begin()}, {5}, {8, 9, 10}, {6, 7}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::ASCENDING, cudf::null_order::BEFORE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  // Descending
  // LCW<int>  order{{3, 0, 1, 2}, {0}, {0, 1, 2},  {1, 0}};
  {
    LCW<T> expected{{{4, 3, 2, 1}, valids_b.begin()}, {5}, {10, 9, 8}, {7, 6}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::DESCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    LCW<T> expected{{{3, 2, 1, 4}, valids_a.begin()}, {5}, {10, 9, 8}, {7, 6}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::DESCENDING, cudf::null_order::BEFORE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
}

using SortListsInt = SortLists<int>;

TEST_F(SortListsInt, Empty)
{
  using T = int;

  {
    LCW<T> l{};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
  {
    LCW<T> l{LCW<T>{}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
  {
    LCW<T> l{LCW<T>{}, LCW<T>{}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
}

TEST_F(SortListsInt, Single)
{
  using T = int;

  {
    LCW<T> l{1};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
  {
    LCW<T> l{{1, 2, 3}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
}

TEST_F(SortListsInt, NullRows)
{
  using T = int;
  std::vector<int> valids{0, 1, 0};
  LCW<T> l{{{1, 2, 3}, {4, 5, 6}, {7}}, valids.begin()};  // offset 0, 0, 3, 3

  auto const [sorted_lists, stable_sorted_lists] =
    generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
}

namespace {
// Sorts `input` under the given explicit (order, null_order) -- ascending / nulls-after by default
// -- through the fast path (`sort_lists`) and the comparison path (`stable_sort_lists`), asserting
// both equal `expected`. The fast path must reproduce the comparison path's order exactly; only
// speed differs. The stable assertion doubles as a run-time check of every hand- or host-computed
// expectation against the comparison semantics.
void expect_both_sort_paths_match(cudf::lists_column_view const& input,
                                  cudf::column_view const& expected,
                                  cudf::order column_order         = cudf::order::ASCENDING,
                                  cudf::null_order null_precedence = cudf::null_order::AFTER)
{
  auto const sorted = cudf::lists::sort_lists(input, column_order, null_precedence);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->view(), expected);
  auto const stable_sorted = cudf::lists::stable_sort_lists(input, column_order, null_precedence);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted->view(), expected);
}

// EQUIVALENT variant of `expect_both_sort_paths_match` for floating point. Under the unstable
// contract -0.0/+0.0 and distinct NaN bit patterns are interchangeable, so the exact per-slot bytes
// within such a group may differ between the fast and comparison paths; EQUIVALENT accepts that
// while still pinning the order of every distinct finite value, the infinities, and the NaN block's
// position.
void expect_both_sort_paths_equivalent(cudf::lists_column_view const& input,
                                       cudf::column_view const& expected,
                                       cudf::order column_order         = cudf::order::ASCENDING,
                                       cudf::null_order null_precedence = cudf::null_order::AFTER)
{
  auto const sorted = cudf::lists::sort_lists(input, column_order, null_precedence);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted->view(), expected);
  auto const stable_sorted = cudf::lists::stable_sort_lists(input, column_order, null_precedence);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(stable_sorted->view(), expected);
}

// Wraps a leaf column as a single-row LIST (every element in one list row) -- the shape the
// value-ordering tests for the wide chrono and decimal128 keys need, so the sort orders within that
// one segment. Element-level nulls live in the leaf's validity.
std::unique_ptr<cudf::column> as_single_row_list(std::unique_ptr<cudf::column> leaf)
{
  auto const n = static_cast<cudf::size_type>(leaf->size());
  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, n};
  return cudf::make_lists_column(1, offsets.release(), std::move(leaf), 0, {});
}
}  // namespace

// Sign-flip correctness at the signed-integer boundaries: INT_MIN must sort first and INT_MAX last
// among the non-nulls, with every negative ahead of every non-negative, for both rep widths. The
// null-bearing rows route to the tiered kernel (which orders nulls last under null_order::AFTER);
// the int32 rep takes its uint64 tiered key and the int64 rep the unsigned __int128 key. Verified
// against the comparison path.
TEST_F(SortListsInt, NumericPackedNegativesAndBounds)
{
  {  // int32: the <= 4-byte uint64 tiered key.
    auto constexpr lo = std::numeric_limits<int32_t>::min();
    auto constexpr hi = std::numeric_limits<int32_t>::max();
    std::vector<bool> const in_valids{true, true, false, true, true, true, false};
    LCW<int32_t> input{{{hi, -1, 0, lo, 0, 1, 0}, in_valids.begin()}};
    std::vector<bool> const ex_valids{true, true, true, true, true, false, false};
    LCW<int32_t> expected{{{lo, -1, 0, 1, hi, 0, 0}, ex_valids.begin()}};
    expect_both_sort_paths_match(cudf::lists_column_view{input}, expected);
  }
  {  // int64: the 8-byte unsigned __int128 tiered key. The file-wide `LCW` alias fixes its source
     // element type to `int32_t`, which cannot represent the 64-bit bounds, so this block uses a
     // wrapper sourcing directly from `int64_t`.
    using LCW64       = cudf::test::lists_column_wrapper<int64_t>;
    auto constexpr lo = std::numeric_limits<int64_t>::min();
    auto constexpr hi = std::numeric_limits<int64_t>::max();
    std::vector<bool> const in_valids{true, true, false, true, true, true, false};
    LCW64 input{{{hi, -1, 0, lo, 0, 1, 0}, in_valids.begin()}};
    std::vector<bool> const ex_valids{true, true, true, true, true, false, false};
    LCW64 expected{{{lo, -1, 0, 1, hi, 0, 0}, ex_valids.begin()}};
    expect_both_sort_paths_match(cudf::lists_column_view{input}, expected);
  }
}

// Empty, single-element, and duplicate rows through the tiered kernel. A null element routes the
// column there (any null-bearing column takes the tiered path, which orders nulls last), so these
// small shapes exercise the network tier: an empty row contributes no slots, a single-element row
// is trivially ordered, an all-duplicate row keeps its value, and the last row's non-nulls sort
// ascending with the null placed last under null_order::AFTER. The default-order Empty/Single tests
// cover the same shapes on the comparison path.
TEST_F(SortListsInt, NumericPackedEmptyAndSingleRows)
{
  std::vector<bool> const in_valids{true, false, true};
  LCW<int32_t> input{LCW<int32_t>{},
                     LCW<int32_t>{5},
                     LCW<int32_t>{-3, -3},
                     LCW<int32_t>{{2, 7, -1}, in_valids.begin()}};
  std::vector<bool> const ex_valids{true, true, false};
  LCW<int32_t> expected{LCW<int32_t>{},
                        LCW<int32_t>{5},
                        LCW<int32_t>{-3, -3},
                        LCW<int32_t>{{-1, 2, 7}, ex_valids.begin()}};
  expect_both_sort_paths_match(cudf::lists_column_view{input}, expected);
}

// Pre-epoch (negative rep) and post-epoch timestamps plus a null, for both rep widths:
// TIMESTAMP_DAYS (int32 rep -> uint64 tiered key) and TIMESTAMP_MILLISECONDS (int64 rep -> unsigned
// __int128 key). The null-bearing rows route to the tiered kernel; the signed flip orders negatives
// before non-negatives, so pre-epoch sorts first, and the null collects last under AFTER. Chrono
// values arrive as wrappers keyed by their extracted rep, so this also guards the is_timestamp
// rep-extraction branch. Verified against the comparison path.
TEST_F(SortListsInt, NumericPackedTimestamps)
{
  {  // TIMESTAMP_DAYS: int32 rep, the <= 4-byte uint64 tiered key.
    std::vector<int32_t> const in{100, -50, 0, 0, 25};
    std::vector<int32_t> const ex{-50, 0, 25, 100, 0};
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<cudf::timestamp_D>(in.begin(), in.end(), null_at(2))
        .release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<cudf::timestamp_D>(ex.begin(), ex.end(), null_at(4))
        .release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
  {  // TIMESTAMP_MILLISECONDS: int64 rep exceeding int32, the 8-byte unsigned __int128 tiered key.
    auto constexpr big = int64_t{5} * 1'000 * 1'000 * 1'000;  // 5e9 > INT32_MAX, post-epoch
    std::vector<int64_t> const in{big, -big, 0, 1, -1};
    std::vector<int64_t> const ex{-big, -1, 1, big, 0};
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms>(in.begin(), in.end(), null_at(2))
        .release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms>(ex.begin(), ex.end(), null_at(4))
        .release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
}

// Null-bearing BOOL8: `bool` is packed-radix-supported but not tiered, so any null routes it to the
// packed radix at any list size. Exercises the bool branch of `radix_encode_u32` plus the packed
// key's null bit, which no other test reaches (the TYPED Null test skips bool and the
// segmented-sort Bool test is no-null on the legacy CUB path).
TEST_F(SortListsInt, NumericPackedBoolWithNulls)
{
  std::vector<bool> const in_valids{true, true, false, true, true, false, true};
  auto input =
    as_single_row_list(cudf::test::fixed_width_column_wrapper<bool>(
                         {true, false, true, true, false, false, false}, in_valids.begin())
                         .release());
  std::vector<bool> const ex_valids{true, true, true, true, true, false, false};
  auto expected =
    as_single_row_list(cudf::test::fixed_width_column_wrapper<bool>(
                         {false, false, false, true, true, false, false}, ex_valids.begin())
                         .release());
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// The exact average-list-size cells of the packed-radix gate (avg >= 100): a 200-element single row
// (two offsets -> average exactly 100) is the first packed-radix shape, and a 198-element row
// (average 99) the last tiered shape. Both must reproduce the comparison order, pinning the gate's
// >= boundary, which the other tests straddle only from a distance (20 vs 110).
TEST_F(SortListsInt, NumericPackedRadixAvgGateBoundary)
{
  auto const check_single_row = [](cudf::size_type n) {
    std::vector<int32_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = n / 2 - i;
    }
    std::vector<int32_t> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int32_t>(in.begin(), in.end()).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int32_t>(ex.begin(), ex.end()).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  };
  check_single_row(200);  // average exactly 100 -> the first packed-radix cell
  check_single_row(198);  // average 99 -> the last tiered cell
}

// No-null long lists (average at/above the packed-radix cutoff) route to the one-shot packed-radix
// sort for every supported type -- the only path that claims the long-list band. A single row of
// ~two hundred distinct values spanning the sign boundary, for the three key widths: int32, int64
// (beyond the 32-bit range), and decimal128 (into the high value words). Each is checked against
// the comparison sort's ascending order.
TEST_F(SortListsInt, NumericPackedRadixLongLists)
{
  cudf::size_type const n = 220;  // single row, average 110 -> packed radix
  {                               // int32
    std::vector<int32_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = n / 2 - i;
    }
    std::vector<int32_t> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int32_t>(in.begin(), in.end()).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int32_t>(ex.begin(), ex.end()).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
  {  // int64 beyond the 32-bit range
    auto constexpr big = int64_t{5} * 1'000 * 1'000 * 1'000;
    std::vector<int64_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = (static_cast<int64_t>(n / 2) - i) * big;
    }
    std::vector<int64_t> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int64_t>(in.begin(), in.end()).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int64_t>(ex.begin(), ex.end()).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
  {  // decimal128 into the high value words
    auto constexpr scale = numeric::scale_type{0};
    auto const k         = [] {
      __int128_t v = 1;
      for (int i = 0; i < 30; ++i) {
        v *= 10;
      }
      return v;
    }();
    std::vector<__int128_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = static_cast<__int128_t>(n / 2 - i) * k;
    }
    std::vector<__int128_t> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(in.begin(), in.end(), scale).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(ex.begin(), ex.end(), scale).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
}

// No-null non-tiered types reach packed_radix only when num_rows >= 1<<18 AND avg >= 100 at once
// (prefer_cub_segmented_sort's OR becomes an AND when negated) -- a route no other test takes:
// TYPED_TEST(SortLists, Null) covers these types' has-nulls key layout at tiny scale only. One
// no-null row of 1<<18 elements satisfies both halves, exercising the no-null packed key (no
// null-class bit, full value budget) for the narrow/unsigned and small-decimal encodings.
TEST_F(SortListsInt, NumericPackedRadixNoNullNarrowTypes)
{
  cudf::size_type const n = 1 << 18;
  auto const check        = [&](auto tag) {
    using T = decltype(tag);
    std::vector<T> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = static_cast<T>((static_cast<int64_t>(n) - i) % 60'000);
    }
    std::vector<T> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input =
      as_single_row_list(cudf::test::fixed_width_column_wrapper<T>(in.begin(), in.end()).release());
    auto expected =
      as_single_row_list(cudf::test::fixed_width_column_wrapper<T>(ex.begin(), ex.end()).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  };
  check(uint16_t{});
  check(int16_t{});
  check(uint64_t{});

  // DECIMAL32/64 (non-tiered) take the same no-null packed radix over their int32/int64 rep.
  auto const check_decimal = [&](auto rep_tag) {
    using Rep            = decltype(rep_tag);
    auto constexpr scale = numeric::scale_type{0};
    std::vector<Rep> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = static_cast<Rep>((static_cast<int64_t>(n) - i) % 60'000);
    }
    std::vector<Rep> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<Rep>(in.begin(), in.end(), scale).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<Rep>(ex.begin(), ex.end(), scale).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  };
  check_decimal(int32_t{});  // DECIMAL32.
  check_decimal(int64_t{});  // DECIMAL64.
}

TEST_F(SortListsInt, NestedListElement)
{
  using T = int;
  // Column of LIST<LIST<int>>: each row's inner lists are reordered as whole elements. The third
  // row's inner lists tie on their first element, so ordering falls through to the second.
  LCW<T> input{LCW<T>{{3, 1}, {2, 0}}, LCW<T>{{5, 5}, {4, 9}}, LCW<T>{{1, 3}, {1, 2}}};
  {
    // Ascending.
    LCW<T> expected{LCW<T>{{2, 0}, {3, 1}}, LCW<T>{{4, 9}, {5, 5}}, LCW<T>{{1, 2}, {1, 3}}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
  {
    // Descending reverses each row's ascending order.
    LCW<T> expected{LCW<T>{{3, 1}, {2, 0}}, LCW<T>{{5, 5}, {4, 9}}, LCW<T>{{1, 3}, {1, 2}}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{input}, cudf::order::DESCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
}

// E = LIST<STRUCT<int, int>>: a list-of-struct element type sorts; struct ranks are computed
// internally before the lexicographic comparison.
TEST_F(SortListsInt, ListOfStructElement)
{
  // One row with two elements [{3, 30}] and [{1, 10}]; ascending reorders to [{1, 10}], [{3, 30}].
  cudf::test::fixed_width_column_wrapper<int> in_f0{3, 1};
  cudf::test::fixed_width_column_wrapper<int> in_f1{30, 10};
  cudf::test::structs_column_wrapper in_structs{{in_f0, in_f1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> in_inner_off{0, 1, 2};
  auto in_inner = cudf::make_lists_column(2, in_inner_off.release(), in_structs.release(), 0, {});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> in_outer_off{0, 2};
  auto in_outer = cudf::make_lists_column(1, in_outer_off.release(), std::move(in_inner), 0, {});

  cudf::test::fixed_width_column_wrapper<int> ex_f0{1, 3};
  cudf::test::fixed_width_column_wrapper<int> ex_f1{10, 30};
  cudf::test::structs_column_wrapper ex_structs{{ex_f0, ex_f1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> ex_inner_off{0, 1, 2};
  auto ex_inner = cudf::make_lists_column(2, ex_inner_off.release(), ex_structs.release(), 0, {});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> ex_outer_off{0, 2};
  auto ex_outer = cudf::make_lists_column(1, ex_outer_off.release(), std::move(ex_inner), 0, {});

  auto const [sorted_lists, stable_sorted_lists] =
    generate_sorted_lists(cudf::lists_column_view{in_outer->view()}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), ex_outer->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), ex_outer->view());
}

// E = STRUCT<int, LIST<int>>: a struct-with-list-field element type sorts.
TEST_F(SortListsInt, StructOfListElement)
{
  // One row with two struct elements {2, [9, 0]} and {1, [8, 7]}; ascending reorders them to
  // {1, [8, 7]}, {2, [9, 0]}.
  cudf::test::fixed_width_column_wrapper<int> in_f0{2, 1};
  cudf::test::lists_column_wrapper<int, int32_t> in_f1{{9, 0}, {8, 7}};
  cudf::test::structs_column_wrapper in_structs{{in_f0, in_f1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> in_off{0, 2};
  auto in_list = cudf::make_lists_column(1, in_off.release(), in_structs.release(), 0, {});

  cudf::test::fixed_width_column_wrapper<int> ex_f0{1, 2};
  cudf::test::lists_column_wrapper<int, int32_t> ex_f1{{8, 7}, {9, 0}};
  cudf::test::structs_column_wrapper ex_structs{{ex_f0, ex_f1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> ex_off{0, 2};
  auto ex_list = cudf::make_lists_column(1, ex_off.release(), ex_structs.release(), 0, {});

  auto const [sorted_lists, stable_sorted_lists] =
    generate_sorted_lists(cudf::lists_column_view{in_list->view()}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), ex_list->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), ex_list->view());
}

TEST_F(SortListsInt, Sliced)
{
  using T = int;
  LCW<T> l{{3, 2, 1, 4}, {7, 5, 6}, {8, 9}, {10}};

  {
    auto const sliced_list = cudf::slice(l, {0, 4})[0];
    auto const expected    = LCW<T>{{1, 2, 3, 4}, {5, 6, 7}, {8, 9}, {10}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{sliced_list}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    auto const sliced_list = cudf::slice(l, {1, 4})[0];
    auto const expected    = LCW<T>{{5, 6, 7}, {8, 9}, {10}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{sliced_list}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    auto const sliced_list = cudf::slice(l, {1, 2})[0];
    auto const expected    = LCW<T>{{5, 6, 7}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{sliced_list}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    auto const sliced_list = cudf::slice(l, {0, 2})[0];
    auto const expected    = LCW<T>{{1, 2, 3, 4}, {5, 6, 7}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{sliced_list}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
}

using SortListsDouble = SortLists<double>;
TEST_F(SortListsDouble, InfinityAndNaN)
{
  auto constexpr NaN = std::numeric_limits<double>::quiet_NaN();
  auto constexpr Inf = std::numeric_limits<double>::infinity();

  using LCW = cudf::test::lists_column_wrapper<double>;
  {
    LCW input{-0.0, -NaN, -NaN, NaN, Inf, -Inf, 7, 5, 6, NaN, Inf, -Inf, -NaN, -NaN, -0.0};
    auto [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{input}, {}, {});
    LCW expected{-Inf, -Inf, -0, -0, 5, 6, 7, Inf, Inf, -NaN, -NaN, NaN, NaN, -NaN, -NaN};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(stable_sorted_lists->view(), expected);
  }
  // This row has over 200 no-null elements, so under the new fast paths the unstable `sort_lists`
  // routes it through the packed-radix engine (its average list size is at/above the fast-sort
  // cutoff); the comparison path (`stable_sort_lists`) sorts it directly. The EQUIVALENT assertions
  // below tolerate the engine choice -- both reproduce the comparison sort's ordering.
  {
    // clang-format off
    LCW input{0.0, -0.0, -NaN, -NaN, NaN, Inf, -Inf,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
              NaN, Inf, -Inf, -NaN, -NaN, -0.0, 0.0};
    LCW expected{-Inf, -Inf, 0.0, -0.0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0, 0,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
               3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
               4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
               5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
               6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
               7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
               8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
               9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
              Inf, Inf, -NaN, -NaN, NaN, NaN, -NaN, -NaN};
    // clang-format on
    auto [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{input}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(stable_sorted_lists->view(), expected);
  }
}

// Floating-point NaN/Inf/signed-zero/denormal ordering through the tiered kernel (explicit
// ASCENDING
// + nulls-AFTER; the null-bearing rows route there, unlike InfinityAndNaN which uses the default
// order and the comparison path). cudf orders -Inf < finite (incl. +/-0 and denormals) < +Inf < NaN
// (every sign/payload, mutually equal, last); nulls follow NaN. Every NaN is canonicalized to the
// all-ones key, so an un-canonicalized -NaN cannot sort near -Inf. EQUIVALENT since +/-0 and NaN
// groups collapse. The thirteen-element FLOAT64 row (unsigned __int128 key) and the ten-element
// FLOAT32 row (uint64 key) both land in the warp tier, covering both key widths; the network tier
// (size <= TIERED_NETWORK_CAP == 8) is exercised by the tiny-segment tests.
TEST_F(SortListsDouble, NumericPackedFloatNaNInfinity)
{
  {  // FLOAT64: the 8-byte unsigned __int128 tiered key; thirteen elements land in the warp tier.
    auto constexpr NaN    = std::numeric_limits<double>::quiet_NaN();
    auto constexpr Inf    = std::numeric_limits<double>::infinity();
    auto constexpr denorm = std::numeric_limits<double>::denorm_min();
    auto constexpr Max    = std::numeric_limits<double>::max();  // DBL_MAX boundary vs. Inf
    using LCWd            = cudf::test::lists_column_wrapper<double>;
    LCWd input{
      {{NaN, -Inf, -0.0, 3.5, -NaN, Inf, 0.0, denorm, -2.0, 0.0, Inf, Max, -Max}, null_at(9)}};
    LCWd expected{
      {{-Inf, -Max, -2.0, -0.0, 0.0, denorm, 3.5, Max, Inf, Inf, NaN, -NaN, 0.0}, null_at(12)}};
    expect_both_sort_paths_equivalent(cudf::lists_column_view{input}, expected);
  }
  {  // FLOAT32: the <= 4-byte uint64 tiered key; ten elements land in the warp tier.
    auto constexpr NaN    = std::numeric_limits<float>::quiet_NaN();
    auto constexpr Inf    = std::numeric_limits<float>::infinity();
    auto constexpr denorm = std::numeric_limits<float>::denorm_min();
    auto constexpr Max    = std::numeric_limits<float>::max();  // FLT_MAX boundary vs. Inf
    using LCWf            = cudf::test::lists_column_wrapper<float>;
    LCWf input{{{-NaN, Inf, -0.0f, 2.5f, -Inf, denorm, 0.0f, NaN, Max, -Max}, null_at(6)}};
    LCWf expected{{{-Inf, -Max, -0.0f, denorm, 2.5f, Max, Inf, NaN, -NaN, 0.0f}, null_at(9)}};
    expect_both_sort_paths_equivalent(cudf::lists_column_view{input}, expected);
  }
}
