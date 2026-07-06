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
#include <array>
#include <limits>
#include <random>
#include <string>
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

using SortListsString = SortLists<cudf::string_view>;

// E = STRING: the element type Spark `array_sort(array<string>)` lowers to. Uses the default order
// (ascending, nulls-after). Covers nulls-last element ordering within a row and a preserved null
// list row.
TEST_F(SortListsString, Strings)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  // Row 0 holds a null element (at index 2) among unsorted strings; row 2 is a whole null list row.
  std::vector<bool> const valids{true, true, false, true};
  StrLCW input{{StrLCW{{"pear", "apple", "fig", "kiwi"}, null_at(2)},
                StrLCW{"banana"},
                StrLCW{"unused"},
                StrLCW{"melon", "cherry"}},
               valids.begin()};

  // Default order ascending, nulls-after: row 0's non-null strings sort to {apple, kiwi, pear}
  // with the null moved to the end. The null slot's placeholder value is not compared. Row 2 stays
  // a null list row, unchanged by the sort.
  StrLCW expected{{StrLCW{{"apple", "kiwi", "pear", "fig"}, null_at(3)},
                   StrLCW{"banana"},
                   StrLCW{"unused"},
                   StrLCW{"cherry", "melon"}},
                  valids.begin()};

  auto const [sorted_lists, stable_sorted_lists] =
    generate_sorted_lists(cudf::lists_column_view{input}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// The Prefix* tests target the STRING ascending/nulls-after fast path, which orders elements by a
// packed key of their leading bytes and then resolves prefix ties by successive byte windows and a
// final byte comparison. Each asserts both sort_lists (the fast path) and stable_sort_lists (the
// comparison path) match a host-built expected column, so any divergence in the fast path from the
// comparison semantics is caught directly.

// Distinct strings whose first eight bytes are identical force the prefix to tie, exercising the
// full-string tie-break. "abcdefgh" is a proper prefix of the other two, so it sorts first.
TEST_F(SortListsString, PrefixEqualFirstEightBytes)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{{"abcdefgh1", "abcdefgh0", "abcdefgh"}};
  StrLCW expected{{"abcdefgh", "abcdefgh0", "abcdefgh1"}};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// A short string that is a prefix of a longer one must sort first ("aa" < "aaa"). With both under
// eight bytes, the longer string's prefix has a non-zero byte where the shorter is zero-padded, so
// the prefix alone already orders them.
TEST_F(SortListsString, PrefixSubstringShorterFirst)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{{"aaa", "aa", "a"}};
  StrLCW expected{{"a", "aa", "aaa"}};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// Empty strings carry an all-zero prefix and must sort before any non-empty string. Multiple empty
// strings tie on both prefix and full string, so their relative order is unconstrained.
TEST_F(SortListsString, PrefixEmptyStringsFirst)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{{"banana", "", "apple", "", "fig"}};
  StrLCW expected{{"", "", "apple", "banana", "fig"}};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// Several null elements in one row must collect at the end under null_order::AFTER while the
// non-null strings sort ascending ahead of them. The null slots' placeholder values are not
// compared.
TEST_F(SortListsString, PrefixMultipleNullsLast)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{StrLCW{{"pear", "apple", "fig", "kiwi", "lime"}, nulls_at({1, 3})}};
  StrLCW expected{StrLCW{{"fig", "lime", "pear", "apple", "kiwi"}, nulls_at({3, 4})}};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// A whole null list row sits next to populated rows; the sort leaves the null row untouched and
// orders the others, exercising the fast path alongside a null row in the same column.
TEST_F(SortListsString, PrefixAllNullListRow)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  std::vector<bool> const valids{true, false, true};
  StrLCW input{{StrLCW{"cherry", "apple", "banana"}, StrLCW{"unused"}, StrLCW{"date", "egg"}},
               valids.begin()};
  StrLCW expected{{StrLCW{"apple", "banana", "cherry"}, StrLCW{"unused"}, StrLCW{"date", "egg"}},
                  valids.begin()};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// Every element of the row shares one eight-byte prefix, so the entire row is a single prefix tie
// run resolved wholly by full-string comparison. This is the worst case for the tie-break: one run
// spanning the whole segment.
TEST_F(SortListsString, PrefixWholeRowShared)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{{"prefix__zebra", "prefix__apple", "prefix__mango", "prefix__"}};
  StrLCW expected{{"prefix__", "prefix__apple", "prefix__mango", "prefix__zebra"}};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// UTF-8 multibyte strings must order by raw byte value, which is what the fast path's byte-packed
// prefix and full-string compare both use. The two-byte (é, U+00E9 -> 0xC3 0xA9), three-byte
// (€, U+20AC -> 0xE2 0x82 0xAC), and four-byte (U+1F600 -> 0xF0 0x9F 0x98 0x80) lead bytes order
// after the ASCII strings and by lead-byte class among themselves (0xC3 < 0xE2 < 0xF0), so all four
// UTF-8 sequence lengths are pinned through the packed key.
TEST_F(SortListsString, PrefixUtf8Multibyte)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{
    {"\xE2\x82\xAC\x75\x72\x6F", "zoo", "\xF0\x9F\x98\x80", "\xC3\xA9\x70\xC3\xA9\x65", "apple"}};
  StrLCW expected{
    {"apple", "zoo", "\xC3\xA9\x70\xC3\xA9\x65", "\xE2\x82\xAC\x75\x72\x6F", "\xF0\x9F\x98\x80"}};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// Multiple rows in one column exercise the segment_id key field: each row sorts independently and
// the per-row prefix ties (rows 0 and 3 share an eight-byte prefix within the row) resolve without
// crossing row boundaries. An empty row (row 1) checks that dense segment-ordinal labeling skips a
// zero-length segment without misaligning the populated rows around it.
TEST_F(SortListsString, PrefixMultipleSegments)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{StrLCW{"commonpx_b", "commonpx_a", "commonpx_c"},
               StrLCW{},
               StrLCW{"melon", "apple", "cherry"},
               StrLCW{"sharedpx2", "sharedpx0", "sharedpx1"}};
  StrLCW expected{StrLCW{"commonpx_a", "commonpx_b", "commonpx_c"},
                  StrLCW{},
                  StrLCW{"apple", "cherry", "melon"},
                  StrLCW{"sharedpx0", "sharedpx1", "sharedpx2"}};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
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

// Host oracle for one list row under an explicit (order, null_order): sorts the non-null values
// (descending reverses) and rebuilds the row with its nulls on the side the combo produces. cudf's
// null_order is comparator-level and a DESCENDING sort swaps the comparison operands, inverting
// null placement too, so nulls land first exactly when (BEFORE) != (DESCENDING) -- the placement
// the pre-existing comparison-path combo tests pin, and which the paired stable_sort_lists
// assertion re-verifies at run time. Null slots carry a zeroed payload; their bytes are never
// compared.
template <typename T>
std::pair<std::vector<T>, std::vector<bool>> host_sorted_row(std::vector<T> const& vals,
                                                             std::vector<bool> const& valids,
                                                             cudf::order column_order,
                                                             cudf::null_order null_precedence)
{
  std::vector<T> nn;
  for (std::size_t i = 0; i < vals.size(); ++i) {
    if (valids[i]) { nn.push_back(vals[i]); }
  }
  std::sort(nn.begin(), nn.end());
  if (column_order == cudf::order::DESCENDING) { std::reverse(nn.begin(), nn.end()); }
  auto const num_nulls = vals.size() - nn.size();
  auto const nulls_first =
    (null_precedence == cudf::null_order::BEFORE) != (column_order == cudf::order::DESCENDING);
  std::vector<T> out;
  std::vector<bool> out_valids;
  auto const push_nulls = [&] {
    for (std::size_t k = 0; k < num_nulls; ++k) {
      out.push_back(T{});
      out_valids.push_back(false);
    }
  };
  if (nulls_first) { push_nulls(); }
  for (auto const& v : nn) {
    out.push_back(v);
    out_valids.push_back(true);
  }
  if (not nulls_first) { push_nulls(); }
  return {std::move(out), std::move(out_valids)};
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

// Short distinct strings, each shorter than the packed key, ordered entirely by the key with no
// tie-break. Adds multi-byte discrimination ("aaa" < "ab" turns on the second byte) beyond the
// simpler substring case.
TEST_F(SortListsString, PrefixShortStrings)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{{"aaa", "aa", "a", "ab", "b"}};
  StrLCW expected{{"a", "aa", "aaa", "ab", "b"}};
  expect_both_sort_paths_match(cudf::lists_column_view{input}, expected);
}

// Shared-prefix data with a null and a duplicate: the realistic regime the shared_prefix_len
// benchmark axis targets, where distinct strings agree on a leading run of bytes so the packed key
// is uninformative and the tie-break does the real ordering. Row 0's strings share an eight-byte
// "AAAAAAAA" prefix, forcing the comparison past the leading bytes, and include a null (sorts last
// under null_order::AFTER) and a duplicate. Row 1's strings share a five-byte "food_" prefix.
TEST_F(SortListsString, PrefixSharedPrefixWithNullAndDuplicate)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{
    StrLCW{{"AAAAAAAAzebra", "AAAAAAAAapple", "AAAAAAAAmango", "unused", "AAAAAAAAapple"},
           null_at(3)},
    StrLCW{"food_pear", "food_kiwi", "food_pear"}};
  StrLCW expected{
    StrLCW{{"AAAAAAAAapple", "AAAAAAAAapple", "AAAAAAAAmango", "AAAAAAAAzebra", "unused"},
           null_at(4)},
    StrLCW{"food_kiwi", "food_pear", "food_pear"}};
  expect_both_sort_paths_match(cudf::lists_column_view{input}, expected);
}

// Guards the null/0xFF distinction. A naive 0xFF..FF null sentinel would collide with a non-null
// string whose leading bytes are all 0xFF, making the two indistinguishable to the radix and
// leaving the non-null 0xFF strings unordered and misplaced relative to the real null under
// null_order::AFTER. `packed_key_builder`'s dedicated is_null bit avoids that by construction: it
// sits above every prefix bit a non-null value can set, so the null separates cleanly in the first
// pass. The two 0xFF strings instead tie with each other -- they run 72 and 73 bytes of 0xFF, so
// every byte window the iterative passes inspect is also all-0xFF, and they reach the comparison
// cleanup as one run. The correct order sorts the non-nulls ascending by unsigned byte value --
// both 0xFF strings after the ASCII strings, the shorter 0xFF string before the longer (a prefix
// sorts first) -- with the genuine null last. Built against a host reference so the fast path's
// divergence is caught directly.
TEST_F(SortListsString, PrefixNullSentinelCollisionFF)
{
  using StrCW = cudf::test::strings_column_wrapper;

  std::string const ff72(72, '\xff');
  std::string const ff73(73, '\xff');

  // One row: two ASCII strings, two all-0xFF strings, and a genuine null (placeholder value
  // unused).
  std::vector<std::string> const in_strings{"apple", ff73, "mango", ff72, ""};
  std::vector<bool> const in_valids{true, true, true, true, false};

  // Host reference: non-nulls ascending by raw bytes (0xFF strings sort after ASCII; ff72 < ff73 as
  // a prefix), then the null last under null_order::AFTER.
  std::vector<std::string> const ex_strings{"apple", "mango", ff72, ff73, ""};
  std::vector<bool> const ex_valids{true, true, true, true, false};

  auto const make_single_row_list = [](std::vector<std::string> const& strs,
                                       std::vector<bool> const& valids) {
    StrCW leaf{strs.begin(), strs.end(), valids.begin()};
    auto const n = static_cast<cudf::size_type>(strs.size());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, n};
    return cudf::make_lists_column(1, offsets.release(), leaf.release(), 0, {});
  };

  auto const input    = make_single_row_list(in_strings, in_valids);
  auto const expected = make_single_row_list(ex_strings, ex_valids);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// A single row whose non-null elements all share an 80-byte prefix -- past the 71 bytes any
// iterative radix window reaches (see PrefixTrueHeapsortRun) -- forms one prefix tie run longer
// than TIE_HEAPSORT_THRESHOLD, so the tie-break takes its heapsort branch, not insertion sort.
// Exceeding STRINGS_GRAD_WARP_CAP (64) keeps the column on the prefix path. The run holds seventy
// distinct two-digit-tail strings plus two exact duplicates (72 non-null) in a deterministic
// shuffle, with two nulls that collect last under null_order::AFTER, so heapsort is exercised
// alongside null placement and checked against std::sort and the comparison path.
TEST_F(SortListsString, PrefixSharedPrefixHeapsortRun)
{
  std::string const prefix(80, 'A');  // 80 > 7 + 8*8 = 71, so ties survive every iterative window.
  std::vector<std::string> non_null;
  for (int i = 0; i < 70; ++i) {
    non_null.push_back(prefix + std::to_string(i / 10) + std::to_string(i % 10));
  }
  // Two exact duplicates land among genuine distinct strings; their relative order is immaterial.
  non_null.push_back(prefix + "05");
  non_null.push_back(prefix + "17");

  // The fast path is unstable, so the input must arrive unsorted to prove the heapsort orders it.
  auto shuffled = non_null;
  std::shuffle(shuffled.begin(), shuffled.end(), std::mt19937{0xC0'FFEE});

  // Interleave two nulls so the row mixes the long shared-prefix run with null elements.
  auto const null_pos_a = shuffled.size() / 3;
  auto const null_pos_b = (shuffled.size() * 2) / 3 + 1;
  std::vector<std::string> in_strings;
  std::vector<bool> in_valids;
  for (std::size_t i = 0; i <= shuffled.size(); ++i) {
    if (i == null_pos_a || i == null_pos_b) {
      in_strings.emplace_back("");  // Placeholder for a null element; its value is never compared.
      in_valids.push_back(false);
    }
    if (i < shuffled.size()) {
      in_strings.push_back(shuffled[i]);
      in_valids.push_back(true);
    }
  }

  // Host reference: non-null strings ascending by raw bytes, the two nulls last (nulls AFTER).
  auto sorted = non_null;
  std::sort(sorted.begin(), sorted.end());
  std::vector<std::string> ex_strings = sorted;
  std::vector<bool> ex_valids(sorted.size(), true);
  ex_strings.emplace_back("");
  ex_valids.push_back(false);
  ex_strings.emplace_back("");
  ex_valids.push_back(false);

  using StrCW                     = cudf::test::strings_column_wrapper;
  auto const make_single_row_list = [](std::vector<std::string> const& strs,
                                       std::vector<bool> const& valids) {
    StrCW leaf{strs.begin(), strs.end(), valids.begin()};
    auto const n = static_cast<cudf::size_type>(strs.size());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, n};
    return cudf::make_lists_column(1, offsets.release(), leaf.release(), 0, {});
  };

  auto const input    = make_single_row_list(in_strings, in_valids);
  auto const expected = make_single_row_list(ex_strings, ex_valids);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// Worst case for the iterative-radix tie-break: leading runs far longer than the eight-byte key,
// so the order is decided only after several windows past the initial prefix. Row 0 mixes a
// twenty-byte shared prefix (more than twice the eight-byte key and five times the four-byte key)
// with distinct tails of varying length (mixed run lengths), two exact duplicates, an empty string,
// a whole-element null, and -- crucially -- a length tie that no byte window can resolve: a string
// and that same string plus a trailing embedded null byte, separable only by the comparison
// cleanup's shorter-is-less rule. Row 1 is an outlier of thousands of strings sharing a twenty-four
// byte prefix with distinct four-digit tails, exercising the pass cap and the parallel re-sort that
// replaces the serial straggler. The host reference sorts each row's non-null strings by
// std::string ordering (raw-byte lexicographic with the same length tie-break) and appends the
// nulls last, and the assertion checks the fast path against the comparison path.
TEST_F(SortListsString, PrefixIterativeDeepLongSharedPrefix)
{
  using StrCW = cudf::test::strings_column_wrapper;

  std::string const prefix0(20, 'A');  // Twenty shared bytes: > 2x the 8-byte key, 5x the 4-byte.
  std::vector<std::string> row0_non_null{
    prefix0 + "zebra",
    prefix0 + "ant",  // Shorter tail than its neighbors -> mixed run lengths within the shared run.
    prefix0 + "mango",
    prefix0 + "ant",                 // Exact duplicate of an earlier element.
    prefix0,                         // The shared prefix with no tail (the prefix-of element).
    prefix0 + std::string(1, '\0'),  // Same bytes as `prefix0` plus a trailing embedded null byte;
                                     // a pure length tie that no byte window can separate.
    prefix0 + "mango",               // Second exact duplicate.
    ""};                             // Empty string: all-zero key, must sort first in the row.

  // Row 1: a large outlier sharing a long prefix, with distinct tails forcing many tied passes.
  std::string const prefix1(24, 'Q');  // Twenty-four shared bytes -> three windows past an 8B key.
  std::vector<std::string> row1;
  row1.reserve(5'000);
  for (int i = 0; i < 5'000; ++i) {
    auto tail = std::to_string(i);
    tail.insert(tail.begin(), 4 - tail.size(), '0');  // Zero-pad to four digits for a fixed width.
    row1.push_back(prefix1 + tail);
  }
  // Arrive unsorted so the sort must do real work; a fixed shuffle keeps the test deterministic.
  std::shuffle(row1.begin(), row1.end(), std::mt19937{0x5EED});

  // Assemble the leaf strings and validity: row 0 (with one null appended), then row 1 (no nulls).
  std::vector<std::string> in_strings;
  std::vector<bool> in_valids;
  for (auto const& s : row0_non_null) {
    in_strings.push_back(s);
    in_valids.push_back(true);
  }
  in_strings.emplace_back("");  // Placeholder for a null element; its value is never compared.
  in_valids.push_back(false);
  auto const row0_len = static_cast<cudf::size_type>(in_strings.size());
  for (auto const& s : row1) {
    in_strings.push_back(s);
    in_valids.push_back(true);
  }
  auto const total_len = static_cast<cudf::size_type>(in_strings.size());

  // Host reference: each row's non-null strings ascending by std::string (raw-byte order with the
  // same length tie-break), then the row 0 null last.
  auto row0_sorted = row0_non_null;
  std::sort(row0_sorted.begin(), row0_sorted.end());
  auto row1_sorted = row1;
  std::sort(row1_sorted.begin(), row1_sorted.end());

  std::vector<std::string> ex_strings;
  std::vector<bool> ex_valids;
  for (auto const& s : row0_sorted) {
    ex_strings.push_back(s);
    ex_valids.push_back(true);
  }
  ex_strings.emplace_back("");
  ex_valids.push_back(false);
  for (auto const& s : row1_sorted) {
    ex_strings.push_back(s);
    ex_valids.push_back(true);
  }

  auto const make_two_row_list = [&](std::vector<std::string> const& strs,
                                     std::vector<bool> const& valids) {
    StrCW leaf{strs.begin(), strs.end(), valids.begin()};
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, row0_len, total_len};
    return cudf::make_lists_column(2, offsets.release(), leaf.release(), 0, {});
  };

  auto const input    = make_two_row_list(in_strings, in_valids);
  auto const expected = make_two_row_list(ex_strings, ex_valids);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// Exercises the length-uniform exhausted-run drop directly: a run of many byte-identical strings
// longer than the packed key enters the window loop as one tie run, and once the windows cover
// their length the run is length-uniform and exhausted, so it is frozen in one step instead of
// being dragged through the pass cap and the comparison cleanup. A distinct-tailed string (a
// singleton) and a shorter proper prefix round out the row. The frozen order must still match the
// comparison path.
TEST_F(SortListsString, PrefixExhaustedIdenticalRunDropped)
{
  using StrCW = cudf::test::strings_column_wrapper;
  std::vector<std::string> in;
  for (int i = 0; i < 40; ++i) {
    in.emplace_back("abcdefghijkl");  // Forty copies of one 12-byte string: a byte-identical run.
  }
  in.emplace_back("abcdefghijkZ");  // Shares the twelve-byte prefix but the last byte: a singleton.
  in.emplace_back("abcdefgh");      // Shorter proper prefix: its own run, sorts first.

  auto expected = in;
  std::sort(expected.begin(), expected.end());

  auto const input        = as_single_row_list(StrCW{in.begin(), in.end()}.release());
  auto const expected_col = as_single_row_list(StrCW{expected.begin(), expected.end()}.release());
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected_col->view());
}

// Regression guard for the drop's length-uniformity condition. A string and its zero-extension (the
// same bytes plus a trailing embedded null) collide in every byte window and in the packed key --
// only their length differs -- so the stable radix keeps whichever arrived first. Here the LONGER
// one is placed before the shorter, so the stable order is the WRONG lexicographic order (a proper
// prefix sorts first). An exhaustion-only drop would freeze that wrong order; the length-uniform
// guard keeps this mixed-length run for the comparison cleanup, which orders shorter-first. The
// shared bytes exceed the eight-byte window and the packed key, so the run exhausts well inside the
// loop where a naive drop would fire.
TEST_F(SortListsString, PrefixZeroExtensionLongerFirstNotDropped)
{
  using StrCW = cudf::test::strings_column_wrapper;
  std::string const base(20, 'A');   // Twenty shared bytes.
  std::string const shorter = base;  // Length twenty.
  std::string const longer =
    base + std::string(1, '\0');  // Length twenty-one: `base` + a NUL byte.

  // Longer first, then shorter: stable radix keeps this order, so only a length-aware step can fix
  // it. Two strings diverging at the twenty-first byte keep the run non-trivial and resolve as
  // singletons.
  std::vector<std::string> in{longer, shorter, base + "B", base + "C"};
  auto expected = in;
  std::sort(expected.begin(), expected.end());  // shorter < longer < base+"B" < base+"C".

  auto const input        = as_single_row_list(StrCW{in.begin(), in.end()}.release());
  auto const expected_col = as_single_row_list(StrCW{expected.begin(), expected.end()}.release());
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected_col->view());
}

// Targets the singleton-compaction path: a column mixing many rows whose elements are all distinct
// in their first bytes (resolved by the first pass, so they must be left untouched by the iterative
// windows) with one deep-shared-prefix row that drives several windows. The compaction must extract
// only the deep row's still-tied elements, re-sort just those, and scatter them back without
// disturbing the already-resolved rows. Built as many short distinct-prefix rows surrounding one
// giant row of strings sharing a long prefix; the host reference sorts each row independently so a
// stray reorder of any resolved row -- or a misplaced scatter -- is caught against the comparison
// path.
TEST_F(SortListsString, PrefixCompactionMixedSingletonAndDeepRows)
{
  using StrCW = cudf::test::strings_column_wrapper;

  std::vector<std::vector<std::string>> rows;
  // Twelve "easy" rows: every element differs within the first byte, so each is its own run after
  // the first pass and the iterative passes must not touch them.
  for (int r = 0; r < 12; ++r) {
    rows.push_back({std::string(1, static_cast<char>('a' + r)) + "_zzz",
                    std::string(1, static_cast<char>('A' + r)) + "_aaa",
                    std::string(1, static_cast<char>('0' + r)) + "_mmm"});
  }
  // One "deep" row: 1500 strings sharing a thirty-two byte prefix (four windows past an 8-byte
  // key), distinct four-digit tails, shuffled. This is the only row the compaction should re-sort.
  std::string const deep_prefix(32, 'D');
  std::vector<std::string> deep;
  deep.reserve(1'500);
  for (int i = 0; i < 1'500; ++i) {
    auto tail = std::to_string(i);
    tail.insert(tail.begin(), 4 - tail.size(), '0');
    deep.push_back(deep_prefix + tail);
  }
  std::shuffle(deep.begin(), deep.end(), std::mt19937{0xD00D});
  // Place the deep row in the middle so resolved rows sit on both sides of the compacted block.
  rows.insert(rows.begin() + 6, deep);

  // Flatten to leaf strings + per-row offsets; build the host-sorted expected the same way.
  std::vector<std::string> in_strings;
  std::vector<std::string> ex_strings;
  std::vector<cudf::size_type> offsets{0};
  for (auto const& row : rows) {
    for (auto const& s : row) {
      in_strings.push_back(s);
    }
    auto sorted = row;
    std::sort(sorted.begin(), sorted.end());
    for (auto const& s : sorted) {
      ex_strings.push_back(s);
    }
    offsets.push_back(static_cast<cudf::size_type>(in_strings.size()));
  }

  auto const num_rows  = static_cast<cudf::size_type>(rows.size());
  auto const make_list = [&](std::vector<std::string> const& strs) {
    StrCW leaf{strs.begin(), strs.end()};
    cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
    return cudf::make_lists_column(num_rows, off.release(), leaf.release(), 0, {});
  };

  auto const input    = make_list(in_strings);
  auto const expected = make_list(ex_strings);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// Guards the packed-key tie-break offset. The single-uint64 first key holds only P prefix bits,
// where P follows the segment-label width S = bit_width(num_segments). The labels are dense
// segment ordinals, so a single list row is one segment: S = bit_width(1) = 1, and with nulls
// present P = 64 - 1 - 1 = 62, which is NOT a byte multiple. The key thus proves only
// floor(62/8) = 7 whole leading bytes equal plus the top six bits of byte 7 -- never byte 7's low
// two bits. The two probe strings "AAAAAA\0\x01" and "AAAAAA\0\x02" agree on their first seven
// bytes and on byte 7's top six bits (both 0x01 and 0x02 are below 4, so byte 7 >> 2 is 0 for
// each), so they tie on the packed key; they differ only in byte 7's low two bits. A correct
// tie-break resumes its byte windows at byte 7 and separates them (\x01 < \x02); a stale offset
// that skipped a fixed eight bytes would compare past both eight-byte strings, leave them tied,
// and order them arbitrarily. The row holds 200 elements -- the probe pair, two nulls, and
// distinct "B"-prefixed singletons -- but the count no longer drives S (one row fixes
// num_segments = 1); it only keeps the probe pair its own two-element prefix-tie run amid many
// first-pass singletons. Built against a host std::sort reference (raw-byte order, embedded nulls
// and all) so any offset drift is caught against the comparison path.
TEST_F(SortListsString, PrefixPackedKeyPartialByteTieOffset)
{
  using StrCW = cudf::test::strings_column_wrapper;

  std::string const probe_lo = std::string("AAAAAA") + '\0' + '\x01';  // 8 bytes, byte 7 = 0x01.
  std::string const probe_hi = std::string("AAAAAA") + '\0' + '\x02';  // 8 bytes, byte 7 = 0x02.

  // Distinct "B"-prefixed fillers: each differs within its first bytes, so they resolve in the
  // first pass and never merge into the probe pair's "A"-prefixed run. Two nulls force the layout's
  // null bit (P = 62, not 63) and must collect last under null_order::AFTER.
  std::vector<std::string> in_strings;
  std::vector<bool> in_valids;
  in_strings.push_back(probe_hi);  // Arrive out of order so the tie-break must reorder the pair.
  in_valids.push_back(true);
  in_strings.push_back(probe_lo);
  in_valids.push_back(true);
  in_strings.emplace_back("");  // Null placeholder; its value is never compared.
  in_valids.push_back(false);
  for (int i = 0; in_strings.size() < 199; ++i) {
    auto tail = std::to_string(i);
    tail.insert(tail.begin(), 4 - tail.size(), '0');
    in_strings.push_back("B" + tail);
    in_valids.push_back(true);
  }
  in_strings.emplace_back("");  // Second null placeholder.
  in_valids.push_back(false);
  // 200 elements total, but all in one list row => num_segments = 1 => S = bit_width(1) = 1, and
  // with nulls P = 62, floor(P/8) = 7 (a non-byte-multiple P, the offset this test probes).
  auto const num_elements = static_cast<cudf::size_type>(in_strings.size());

  // Host reference: non-null strings ascending by raw bytes (std::string compares unsigned bytes
  // and respects the embedded null), then the two nulls last.
  std::vector<std::string> non_null;
  for (std::size_t i = 0; i < in_strings.size(); ++i) {
    if (in_valids[i]) { non_null.push_back(in_strings[i]); }
  }
  std::sort(non_null.begin(), non_null.end());
  std::vector<std::string> ex_strings = non_null;
  std::vector<bool> ex_valids(non_null.size(), true);
  ex_strings.emplace_back("");
  ex_valids.push_back(false);
  ex_strings.emplace_back("");
  ex_valids.push_back(false);

  auto const make_single_row_list = [](std::vector<std::string> const& strs,
                                       std::vector<bool> const& valids) {
    StrCW leaf{strs.begin(), strs.end(), valids.begin()};
    auto const n = static_cast<cudf::size_type>(strs.size());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, n};
    return cudf::make_lists_column(1, offsets.release(), leaf.release(), 0, {});
  };

  auto const input    = make_single_row_list(in_strings, in_valids);
  auto const expected = make_single_row_list(ex_strings, ex_valids);
  // One list row fixes num_segments = 1 (so S = 1, P = 62); pin the element count only to keep the
  // probe pair amid a bulk of first-pass singletons rather than alone.
  EXPECT_EQ(num_elements, 200);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// One row mixes three adjacent null elements with genuine non-null tie groups: an eight-byte
// "AAAAAAAA" shared-prefix trio with distinct tails (a first-pass prefix tie the windows resolve)
// and a duplicated "apple" (an exact tie). Under null_order::AFTER the nulls collect last while
// the non-nulls sort ascending, verifying that excluding nulls from the tie-break set (they are
// position-final after the first pass, never counted tied) keeps their placement correct. Were a
// null instead dragged through the windows, this would still be correct but slower; the check here
// guards the exclusion against ever misplacing a null.
TEST_F(SortListsString, PrefixNullRunsNeverTied)
{
  using StrCW = cudf::test::strings_column_wrapper;

  std::vector<std::string> const in_strings{
    "AAAAAAAA3", "", "", "", "AAAAAAAA1", "AAAAAAAA2", "zebra", "apple", "apple"};
  std::vector<bool> const in_valids{true, false, false, false, true, true, true, true, true};

  // Non-nulls ascending by raw bytes ('A' 0x41 < 'a' 0x61 < 'z' 0x7A), then the three nulls last.
  std::vector<std::string> const ex_strings{
    "AAAAAAAA1", "AAAAAAAA2", "AAAAAAAA3", "apple", "apple", "zebra", "", "", ""};
  std::vector<bool> const ex_valids{true, true, true, true, true, true, false, false, false};

  auto const make_single_row_list = [](std::vector<std::string> const& strs,
                                       std::vector<bool> const& valids) {
    StrCW leaf{strs.begin(), strs.end(), valids.begin()};
    auto const n = static_cast<cudf::size_type>(strs.size());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, n};
    return cudf::make_lists_column(1, offsets.release(), leaf.release(), 0, {});
  };

  auto const input    = make_single_row_list(in_strings, in_valids);
  auto const expected = make_single_row_list(ex_strings, ex_valids);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// One row is nothing but ten copies of a single non-empty string -- a whole segment collapsing to
// one all-duplicate tie run (equal keys and equal bytes, so their order is immaterial) -- beside
// ordinary rows. A known suite gap: exercises the tie-break on a segment-spanning run of exact
// duplicates, which must leave every row correctly ordered.
TEST_F(SortListsString, PrefixWholeSegmentDuplicate)
{
  using StrCW = cudf::test::strings_column_wrapper;

  std::vector<std::string> const dup_row(10, "duplicate");
  std::vector<std::vector<std::string>> const rows{
    {"cherry", "apple", "banana"}, dup_row, {"melon", "date", "fig"}};

  std::vector<std::string> in_strings;
  std::vector<std::string> ex_strings;
  std::vector<cudf::size_type> offsets{0};
  for (auto const& row : rows) {
    for (auto const& s : row) {
      in_strings.push_back(s);
    }
    auto sorted = row;
    std::sort(sorted.begin(), sorted.end());
    for (auto const& s : sorted) {
      ex_strings.push_back(s);
    }
    offsets.push_back(static_cast<cudf::size_type>(in_strings.size()));
  }

  auto const num_rows  = static_cast<cudf::size_type>(rows.size());
  auto const make_list = [&](std::vector<std::string> const& strs) {
    StrCW leaf{strs.begin(), strs.end()};
    cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
    return cudf::make_lists_column(num_rows, off.release(), leaf.release(), 0, {});
  };

  auto const input    = make_list(in_strings);
  auto const expected = make_list(ex_strings);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// Forces the comparison cleanup's heapsort branch, which no other test reaches: >= 40 elements in
// ONE row share a prefix long enough that every iterative window sees only shared bytes, so the
// whole run survives to the cleanup as one run >= TIE_HEAPSORT_THRESHOLD (32) and is sorted by
// heapsort rather than insertion. Layout arithmetic for this test: one list row => num_segments = 1
// => S = bit_width(1) = 1; no nulls => P = 64 - 1 = 63, so the first pass proves floor(63/8) = 7
// whole bytes. The MAX_RADIX_PASSES (8) windows then consume 8 * prefix_bytes (8) = 64 more, so the
// last window ends at byte 7 + 64 = 71. An 80-byte shared prefix exceeds 71, so no window can
// separate any element and the distinct two-digit tails (at byte 80, beyond every window) are
// resolved only by the cleanup. Two exact duplicates check the heapsort's handling of equal keys.
TEST_F(SortListsString, PrefixTrueHeapsortRun)
{
  using StrCW = cudf::test::strings_column_wrapper;

  std::string const prefix(80, 'Z');  // 80 > 7 + 8*8 = 71, so ties survive every iterative window.
  std::vector<std::string> non_null;
  // Exceeding STRINGS_GRAD_WARP_CAP (> 64 elements) sends the column down the prefix path
  // (not the graduated-warp path), so the tie run genuinely reaches TIE_HEAPSORT_THRESHOLD.
  for (int i = 0; i < 70; ++i) {
    non_null.push_back(prefix + std::to_string(i / 10) + std::to_string(i % 10));
  }
  // Two exact duplicates land among the distinct strings; their relative order is immaterial.
  non_null.push_back(prefix + "05");
  non_null.push_back(prefix + "17");

  // The fast path is unstable, so the input must arrive unsorted to prove the heapsort orders it.
  auto shuffled = non_null;
  std::shuffle(shuffled.begin(), shuffled.end(), std::mt19937{0x8EA7});

  // Host reference: ascending by raw bytes; all share the prefix, so ordered by the two-digit tail.
  auto sorted = non_null;
  std::sort(sorted.begin(), sorted.end());

  auto const make_single_row_list = [](std::vector<std::string> const& strs) {
    StrCW leaf{strs.begin(), strs.end()};
    auto const n = static_cast<cudf::size_type>(strs.size());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, n};
    return cudf::make_lists_column(1, offsets.release(), leaf.release(), 0, {});
  };

  auto const input    = make_single_row_list(shuffled);
  auto const expected = make_single_row_list(sorted);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

namespace {
// Builds a LIST<STRING> column from per-row (string, validity) data: flattens the rows into one
// leaf strings column carrying the validity plus a matching offsets column. Embedded NUL bytes ride
// through unchanged (a string is a byte sequence), so callers pass explicit-length `std::string`s.
std::unique_ptr<cudf::column> make_string_lists(std::vector<std::vector<std::string>> const& rows,
                                                std::vector<std::vector<bool>> const& valids)
{
  std::vector<std::string> flat;
  std::vector<bool> flat_v;
  std::vector<cudf::size_type> offsets{0};
  for (std::size_t r = 0; r < rows.size(); ++r) {
    flat.insert(flat.end(), rows[r].begin(), rows[r].end());
    flat_v.insert(flat_v.end(), valids[r].begin(), valids[r].end());
    offsets.push_back(static_cast<cudf::size_type>(flat.size()));
  }
  cudf::test::strings_column_wrapper leaf(flat.begin(), flat.end(), flat_v.begin());
  cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
  return cudf::make_lists_column(
    static_cast<cudf::size_type>(rows.size()), off.release(), leaf.release(), 0, {});
}

// Drives one set of LIST<STRING> rows through all four (order, null_order) combos, checking the
// fast path and the comparison path against a per-row host oracle (`host_sorted_row`, unsigned-byte
// order with `std::string`'s length tie-break -- the same order cudf uses). Each combo folds a
// distinct polarity into the keys, so a wrong descending byte complement, exhausted-run side, or
// null side shows up as a mismatch here.
void expect_string_polarity_matrix(std::vector<std::vector<std::string>> const& rows,
                                   std::vector<std::vector<bool>> const& valids)
{
  constexpr std::array<std::pair<cudf::order, cudf::null_order>, 4> combos{
    {{cudf::order::ASCENDING, cudf::null_order::AFTER},
     {cudf::order::DESCENDING, cudf::null_order::AFTER},
     {cudf::order::ASCENDING, cudf::null_order::BEFORE},
     {cudf::order::DESCENDING, cudf::null_order::BEFORE}}};
  auto const input = make_string_lists(rows, valids);
  for (auto const& [ord, np] : combos) {
    std::vector<std::vector<std::string>> ex_rows(rows.size());
    std::vector<std::vector<bool>> ex_valids(rows.size());
    for (std::size_t r = 0; r < rows.size(); ++r) {
      auto sorted_row = host_sorted_row(rows[r], valids[r], ord, np);
      ex_rows[r]      = std::move(sorted_row.first);
      ex_valids[r]    = std::move(sorted_row.second);
    }
    auto const expected = make_string_lists(ex_rows, ex_valids);
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view(), ord, np);
  }
}
}  // namespace

// ===== graduated-warp string path: per-band polarity coverage (segments within the 64-element cap)
// The graduated in-warp sort is the default string fast path once every segment fits the largest
// warp tile, so these drive its bands -- (0,8] one item per lane, (8,16] W8, W16, W32 -- through
// the full (order, null_order) matrix against the comparison-path oracle. Rows are shuffled so the
// sort must actively reorder; a bug that left the input order (e.g. a comparator collapsing valid
// pairs to "equal") shows up as a mismatch.

// Segment sizes pinned at and across every graduated band boundary -- 1..16 (W8), 17..32 (W16),
// 33..64 (W32) -- including 8/9 (the (0,8]/(8,16] split edge), 15/16/17, 31/32/33, 63/64 and an
// empty row, all in one column so the bands run side by side. The 16/32/64 rows fill their W*IPT
// tiles exactly (zero pads), as does the 8 row for the one-item-per-lane band. All-distinct
// strings, no nulls, so the descending combos exercise the packed-window byte complement across
// every band.
TEST_F(SortListsString, GradPolarityMatrixBandBoundarySizes)
{
  std::vector<cudf::size_type> const sizes{1, 2, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 0};
  std::mt19937 rng{0xBEEF};
  std::vector<std::vector<std::string>> rows;
  std::vector<std::vector<bool>> valids;
  for (auto const n : sizes) {
    std::vector<std::string> row(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      // Distinct within a row: 97 is coprime to 1000, so the numeric part never repeats for i < 64.
      row[i] = "k" + std::to_string(i * 97 % 1000) + static_cast<char>('a' + i % 26);
    }
    std::shuffle(row.begin(), row.end(), rng);
    valids.emplace_back(row.size(), true);
    rows.push_back(std::move(row));
  }
  expect_string_polarity_matrix(rows, valids);
}

// Null handling in every band, and the direct regression for the class-bit polarity resolution: the
// two nulls-first combos (ASC/BEFORE and DESC/AFTER on this null-bearing column) put the valid
// class at ordinal 1, so a comparator hardcoding the valid class to 0 would collapse every
// valid-vs-valid pair to "equal" and leave the shuffled input order -- caught here as a mismatch.
// Row 0 (W8 band) mixes 10 valids (one duplicate) with 2 nulls; row 1's 17 elements have only 2
// valid (its W16 tile holds 15 nulls then 15 pad slots, the pad-boundary stressor a pad class
// failing to rank above tier_null would break); row 5 is all null; rows of 5 and 8 put nulls in the
// (0,8] one-item-per-lane band.
TEST_F(SortListsString, GradPolarityMatrixNullsAcrossBands)
{
  std::vector<cudf::size_type> const sizes{12, 17, 20, 40, 64, 16, 5, 8};
  std::mt19937 rng{0xD00D};
  std::vector<std::vector<std::string>> rows;
  std::vector<std::vector<bool>> valids;
  for (std::size_t r = 0; r < sizes.size(); ++r) {
    auto const n = sizes[r];
    std::vector<std::pair<std::string, bool>> cells(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      // Row 1: only elements 3 and 11 valid (2 valid / 15 null). Row 5: all null. Others: every
      // (i % 5 == 2) element is null.
      bool const v = (r == 1) ? (i == 3 || i == 11) : (r == 5 ? false : (i % 5 != 2));
      cells[i]     = {"v" + std::to_string(i * 97 % 1000), v};
    }
    if (n > 1 && cells[0].second && cells[1].second) {
      cells[1].first = cells[0].first;  // A duplicate among the valids.
    }
    std::shuffle(cells.begin(), cells.end(), rng);  // Interleave nulls; don't pre-group them.
    std::vector<std::string> row(n);
    std::vector<bool> valid(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      row[i]   = cells[i].first;
      valid[i] = cells[i].second;
    }
    rows.push_back(std::move(row));
    valids.push_back(std::move(valid));
  }
  expect_string_polarity_matrix(rows, valids);
}

// Ties and zero-extension families across bands: exact duplicates, strings colliding on the packed
// eight-byte window through its zero-fill (`S` vs `S + "\0"`, separable only by the length
// tie-break
// -- the exhausted-window case whose side the descending complement flips), embedded NUL bytes
// ordering before printable bytes, and eight-byte shared prefixes forcing every pair onto the
// suffix fallback. Row 0 is a W8-band family, row 1 a W16-band shared prefix, row 2 a W32-band
// shared prefix, row 3 pure duplicates.
TEST_F(SortListsString, GradPolarityMatrixTiesAndZeroExtension)
{
  using namespace std::string_literals;
  std::mt19937 rng{0xACE5};
  std::vector<std::string> row0{""s, "a"s, "ab"s, "ab"s, "ab\0"s, "ab\0c"s, "abc"s, "abcd"s, "b"s};
  std::vector<std::string> row1;
  for (int i = 0; i < 16; ++i) {
    row1.push_back("PPPPPPPP" + std::to_string(i));
  }
  row1.push_back("PPPPPPPP"s);
  row1.push_back("PPPPPPPP\0"s);
  row1.push_back("PPPPPPPP3"s);  // duplicate
  row1.push_back("PPPPPPPP7"s);  // duplicate
  std::vector<std::string> row2;
  for (int i = 0; i < 38; ++i) {
    row2.push_back("SAMEPREF" + std::string(1, static_cast<char>('0' + (i % 10))) +
                   std::to_string(i));
  }
  row2.push_back("SAMEPREF"s);
  row2.push_back("SAMEPREF\0"s);
  std::vector<std::string> row3(10, "dup"s);

  std::vector<std::vector<std::string>> rows;
  std::vector<std::vector<bool>> valids;
  for (auto* row : {&row0, &row1, &row2, &row3}) {
    std::shuffle(row->begin(), row->end(), rng);
    valids.emplace_back(row->size(), true);
    rows.push_back(*row);
  }
  expect_string_polarity_matrix(rows, valids);
}

// UTF-8 multibyte ordering: raw unsigned-byte order across the two-, three-, and four-byte lead
// classes in a W8-band row, and a W32-band row where every string shares a two-byte multibyte
// prefix so the prekey ties past the lead bytes. Polarity is orthogonal to encoding; the descending
// pass is a cheap byte-complement sanity check.
TEST_F(SortListsString, GradPolarityMatrixUtf8Multibyte)
{
  std::mt19937 rng{0xFACE};
  std::vector<std::string> row0{
    "\xE2\x82\xAC\x75\x72\x6F", "zoo", "\xF0\x9F\x98\x80", "\xC3\xA9\x70\xC3\xA9\x65", "apple"};
  std::vector<std::string> row1;
  for (int i = 0; i < 36; ++i) {
    row1.push_back("\xC3\xA9" + std::to_string(i * 97 % 1000));
  }
  std::vector<std::vector<std::string>> rows;
  std::vector<std::vector<bool>> valids;
  for (auto* row : {&row0, &row1}) {
    std::shuffle(row->begin(), row->end(), rng);
    valids.emplace_back(row->size(), true);
    rows.push_back(*row);
  }
  expect_string_polarity_matrix(rows, valids);
}

// Nonzero column offset under every combo: slicing off row 0 gives the surviving rows' leaf strings
// a genuine nonzero offset, which the graduated key builders and comparators must honor through
// `column_device_view::element` / `is_null`. The null slot carries "x" (never compared); the paired
// stable_sort_lists assertion re-verifies each hand-built expectation against the comparison path.
TEST_F(SortListsString, GradSlicedWithNulls)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  StrLCW l{StrLCW{"zz", "aa"},
           StrLCW{{"banana", "apple", "x", "cherry"}, null_at(2)},  // element 2 ("x") is null
           StrLCW{"b", "a"},
           StrLCW{"x"}};
  auto const sliced = cudf::slice(l, {1, 4})[0];  // drops row 0 -> nonzero child offset
  {                                               // ASC / AFTER: values ascending, null last
    StrLCW expected{
      StrLCW{{"apple", "banana", "cherry", "x"}, null_at(3)}, StrLCW{"a", "b"}, StrLCW{"x"}};
    expect_both_sort_paths_match(cudf::lists_column_view{sliced}, expected);
  }
  {  // ASC / BEFORE: null first, then values ascending
    StrLCW expected{
      StrLCW{{"x", "apple", "banana", "cherry"}, null_at(0)}, StrLCW{"a", "b"}, StrLCW{"x"}};
    expect_both_sort_paths_match(
      cudf::lists_column_view{sliced}, expected, cudf::order::ASCENDING, cudf::null_order::BEFORE);
  }
  {  // DESC / AFTER: the comparator swap places the null first, then values descending
    StrLCW expected{
      StrLCW{{"x", "cherry", "banana", "apple"}, null_at(0)}, StrLCW{"b", "a"}, StrLCW{"x"}};
    expect_both_sort_paths_match(
      cudf::lists_column_view{sliced}, expected, cudf::order::DESCENDING, cudf::null_order::AFTER);
  }
  {  // DESC / BEFORE: values descending, then the null last
    StrLCW expected{
      StrLCW{{"cherry", "banana", "apple", "x"}, null_at(3)}, StrLCW{"b", "a"}, StrLCW{"x"}};
    expect_both_sort_paths_match(
      cudf::lists_column_view{sliced}, expected, cudf::order::DESCENDING, cudf::null_order::BEFORE);
  }
}

// The admission gate's negative side under every combo: one 65-element segment exceeds the
// 64-element warp cap, disqualifying the whole column, so the graduated path is rejected and the
// prefix path produces the result. The only observable contract is an unchanged, correct order
// under each polarity -- which the matrix checks against the comparison-path oracle.
TEST_F(SortListsString, GradOversizedSegmentFallsThrough)
{
  std::mt19937 rng{0xF00D};
  std::vector<std::string> big(65);
  for (int i = 0; i < 65; ++i) {
    big[i] = "q" + std::to_string(i * 97 % 1000);
  }
  std::shuffle(big.begin(), big.end(), rng);
  std::vector<std::vector<std::string>> const rows{big, {"b", "a", "c"}};
  std::vector<std::vector<bool>> const valids{std::vector<bool>(big.size(), true),
                                              {true, true, true}};
  expect_string_polarity_matrix(rows, valids);
}

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

// DECIMAL128 value-ordering at the extremes. Both cases are null-bearing, so both route to the
// tiered kernel (24-byte tiered_key128); they differ only in segment length and null position. Each
// covers
// +/-10^37 reaching the high value words (10^37 ~ 2^123), the sign boundary at zero, a duplicated
// low extreme, and a null. The __int128 rep is sign-flipped (XOR bit 127) then split big-endian, so
// +10^37 sorts above every non-negative, -10^37 flips to a small key and sorts first, and the null
// collects last under AFTER.
TEST_F(SortListsInt, NumericPackedDecimal128Bounds)
{
  auto const pow10_37 = [] {
    __int128_t v = 1;
    for (int i = 0; i < 37; ++i) {
      v *= 10;
    }
    return v;
  }();
  auto constexpr scale = numeric::scale_type{0};

  {  // A longer row whose null routes it to the tiered kernel. Ascending order of the non-null
    // values: the low extreme (duplicated), a run crossing zero, and the high extreme.
    std::vector<__int128_t> ascending{-pow10_37, -pow10_37};
    for (int i = -5; i <= 25; ++i) {
      ascending.push_back(i);
    }
    ascending.push_back(pow10_37);
    auto const null_pos = static_cast<cudf::size_type>(ascending.size());

    // Input feeds the values reversed (so the sort does real work) with the leaf null first;
    // expected is the ascending values with the null last. The null-slot value is ignored.
    std::vector<__int128_t> in_vals;
    in_vals.push_back(0);
    in_vals.insert(in_vals.end(), ascending.rbegin(), ascending.rend());
    std::vector<__int128_t> ex_vals(ascending.begin(), ascending.end());
    ex_vals.push_back(0);

    auto input    = as_single_row_list(cudf::test::fixed_point_column_wrapper<__int128_t>(
                                      in_vals.begin(), in_vals.end(), null_at(0), scale)
                                      .release());
    auto expected = as_single_row_list(cudf::test::fixed_point_column_wrapper<__int128_t>(
                                         ex_vals.begin(), ex_vals.end(), null_at(null_pos), scale)
                                         .release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
  {  // A short row whose null again routes it to the tiered kernel, covering the shorter-segment
    // shape.
    std::vector<__int128_t> const in{pow10_37, -pow10_37, -1, 0, 1, -pow10_37, 0};
    std::vector<__int128_t> const ex{-pow10_37, -pow10_37, -1, 0, 1, pow10_37, 0};
    auto input = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(in.begin(), in.end(), null_at(3), scale)
        .release());
    auto expected = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(ex.begin(), ex.end(), null_at(6), scale)
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

// The tiered kernel's class-composition seam: one int32 column whose row sizes straddle both
// internal tier boundaries -- network (<=8): 0, 1, 3, 8; warp (9..64): 9, 33, 64; radix (>64): 65,
// 600, 3000 -- so the Batcher-network kernel, the full-warp kernel, and the compact large-segment
// radix fallback all run and must merge into one gather map with no cross-row contamination. The
// sizes include both exact caps (8 and 64) and their first over-cap neighbours (9 and 65).
// Scattered element nulls make the column null-bearing, which is what routes it to the tiered
// kernel regardless of its average size; each row is reverse-sorted with a duplicate, and the
// expected column is an independent host sort per row (ascending, nulls last). This is the
// likeliest bug site for a tiered design.
TEST_F(SortListsInt, NumericTieredThreeSizeClasses)
{
  std::vector<cudf::size_type> const sizes{0, 1, 3, 8, 9, 33, 64, 65, 600, 3'000};
  std::vector<int32_t> in_vals;
  std::vector<bool> in_valids;
  std::vector<int32_t> ex_vals;
  std::vector<bool> ex_valids;
  std::vector<cudf::size_type> offsets{0};
  for (auto const s : sizes) {
    std::vector<int32_t> rv(s);
    std::vector<bool> rok(s);
    for (cudf::size_type i = 0; i < s; ++i) {
      rv[i]  = static_cast<int32_t>(s - i);  // distinct, descending -> the sort must fully reorder
      rok[i] = (i % 7 != 3);                 // scattered element nulls
    }
    if (s >= 2) { rv[1] = rv[0]; }  // a duplicate value among the non-nulls
    std::vector<int32_t> nn;
    for (cudf::size_type i = 0; i < s; ++i) {
      in_vals.push_back(rv[i]);
      in_valids.push_back(rok[i]);
      if (rok[i]) { nn.push_back(rv[i]); }
    }
    std::sort(nn.begin(), nn.end());
    for (auto const v : nn) {
      ex_vals.push_back(v);
      ex_valids.push_back(true);
    }
    for (cudf::size_type k = static_cast<cudf::size_type>(nn.size()); k < s; ++k) {
      ex_vals.push_back(0);
      ex_valids.push_back(false);
    }
    offsets.push_back(static_cast<cudf::size_type>(in_vals.size()));
  }
  auto const num_rows  = static_cast<cudf::size_type>(sizes.size());
  auto const make_list = [&](std::vector<int32_t> const& vals, std::vector<bool> const& valids) {
    cudf::test::fixed_width_column_wrapper<int32_t> leaf(vals.begin(), vals.end(), valids.begin());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
    return cudf::make_lists_column(num_rows, off.release(), leaf.release(), 0, {});
  };
  auto const input    = make_list(in_vals, in_valids);
  auto const expected = make_list(ex_vals, ex_valids);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// Every other tiered-kernel test keeps each tier's segment count small enough that `grid_1d` needs
// only one block (TIERED_BLOCK_THREADS is 128, and the warp tier packs 128/32 = 4 virtual warps per
// block). This drives 200 network-tier and 10 warp-tier segments into one no-null int32 column
// (which routes to the tiered kernel), so both kernels' grids span >= 2 blocks, exercising
// `global_thread_id<BLOCK_THREADS>()` across a block boundary.
TEST_F(SortListsInt, NumericTieredMultiBlockGrid)
{
  std::vector<cudf::size_type> sizes;
  for (int i = 0; i < 200; ++i) {
    sizes.push_back(1 + (i % 8));
  }  // 200 network-tier segments
  for (int i = 0; i < 10; ++i) {
    sizes.push_back(16 + (i % 40));
  }  // 10 warp-tier segments
  std::vector<int32_t> in_vals;
  std::vector<int32_t> ex_vals;
  std::vector<cudf::size_type> offsets{0};
  for (auto const s : sizes) {
    std::vector<int32_t> rv(s);
    for (cudf::size_type i = 0; i < s; ++i) {
      rv[i] = static_cast<int32_t>(s - i);
    }
    for (auto const v : rv) {
      in_vals.push_back(v);
    }
    std::sort(rv.begin(), rv.end());
    for (auto const v : rv) {
      ex_vals.push_back(v);
    }
    offsets.push_back(static_cast<cudf::size_type>(in_vals.size()));
  }
  auto const num_rows  = static_cast<cudf::size_type>(sizes.size());
  auto const make_list = [&](std::vector<int32_t> const& vals) {
    cudf::test::fixed_width_column_wrapper<int32_t> leaf(vals.begin(), vals.end());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
    return cudf::make_lists_column(num_rows, off.release(), leaf.release(), 0, {});
  };
  auto const input    = make_list(in_vals);
  auto const expected = make_list(ex_vals);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// Zero-one principle, in-repo: the eight-slot Batcher network sorts every 0/1 pattern at every list
// size 1 through 8 (510 lists, 3586 elements; a no-null int32 column of this shape routes to the
// tiered kernel with every segment in the network tier). A comparison network that sorts every 0/1
// input sorts every totally ordered input, so this permanently pins the hand-unrolled
// 19-comparator sequence against future edits.
TEST_F(SortListsInt, NumericTieredNetworkZeroOneExhaustive)
{
  std::vector<int32_t> vals;
  std::vector<int32_t> ex_vals;
  std::vector<cudf::size_type> offsets{0};
  for (int n = 1; n <= 8; ++n) {
    for (int pattern = 0; pattern < (1 << n); ++pattern) {
      int ones = 0;
      for (int b = 0; b < n; ++b) {
        int const bit = (pattern >> b) & 1;
        vals.push_back(bit);
        ones += bit;
      }
      for (int b = 0; b < n; ++b) {
        ex_vals.push_back(b < n - ones ? 0 : 1);
      }
      offsets.push_back(static_cast<cudf::size_type>(vals.size()));
    }
  }
  auto const num_lists = static_cast<cudf::size_type>(offsets.size() - 1);
  auto const make_list = [&](std::vector<int32_t> const& v) {
    cudf::test::fixed_width_column_wrapper<int32_t> leaf(v.begin(), v.end());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
    return cudf::make_lists_column(num_lists, off.release(), leaf.release(), 0, {});
  };
  auto const input    = make_list(vals);
  auto const expected = make_list(ex_vals);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// Nonzero column offset (from cudf::slice) combined with a retained null element: slicing off row 0
// gives the surviving rows' child column a genuine nonzero column_view::offset() (via
// lists_column_view::get_sliced_child), and row 1's null must still be read correctly through
// column_device_view::is_null() at that offset by the tiered engine (a null-bearing int column
// routes there). SortListsInt.Sliced covers the offset alone (no nulls); this adds the offset +
// null combo.
TEST_F(SortListsInt, SlicedWithNulls)
{
  using T = int;
  std::vector<bool> const valids{true, false, true};  // row 1's middle element (5) is null
  LCW<T> l{{3, 2, 1, 4}, {{7, 5, 6}, valids.begin()}, {8, 9}, {10}};

  auto const sliced_list = cudf::slice(l, {1, 4})[0];  // drops row 0 -> nonzero child offset
  std::vector<bool> const ex_valids{true, true, false};
  auto const expected = LCW<T>{{{6, 7, 0}, ex_valids.begin()}, {8, 9}, {10}};
  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{sliced_list}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// Nonzero column offset for the strings prefix path: slicing off row 0 gives the surviving rows'
// leaf strings column a genuine nonzero offset, which the packed-key builder, the window tie-break,
// and the null handling must all honor. Mirrors SlicedWithNulls (the tiered engine's offset + null
// combo) for the string engine.
TEST_F(SortListsString, PrefixSlicedWithNulls)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  StrLCW l{StrLCW{"zz", "aa"},
           StrLCW{{"banana", "apple", "x", "cherry"}, null_at(2)},
           StrLCW{"b", "a"},
           StrLCW{"x"}};
  auto const sliced_list = cudf::slice(l, {1, 4})[0];  // drops row 0 -> nonzero child offset
  StrLCW expected{
    StrLCW{{"apple", "banana", "cherry", "x"}, null_at(3)}, StrLCW{"a", "b"}, StrLCW{"x"}};
  expect_both_sort_paths_match(cudf::lists_column_view{sliced_list}, expected);
}

// The tiered kernel's warp tier in isolation: three single-row columns within the warp cap (64) and
// no larger segment, so the radix fallback never runs and the full-warp kernels alone must be
// correct. Scattered element nulls route each column to the tiered kernel. An int64
// sixty-four-element row fills the 32*2 warp tile exactly (no pads); a DECIMAL128 fifty-element row
// leaves pad slots; a double sixty-four-element row fills the generic tiered_warp_sort_kernel's
// tile exactly (zero pads on the shared non-int32/int64 path). Each is reverse-sorted with a
// duplicate; the expected column is an independent host sort (ascending, nulls last).
TEST_F(SortListsInt, NumericTieredWarpSegments)
{
  auto constexpr scale = numeric::scale_type{0};
  {  // int64 warp tier.
    cudf::size_type const n = 64;
    std::vector<int64_t> in(n);
    std::vector<bool> in_v(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] =
        static_cast<int64_t>(n - i) * 1'000'000'007LL;  // distinct, descending, beyond 32 bits
      in_v[i] = (i % 5 != 2);                           // scattered nulls
    }
    in[1] = in[0];  // a duplicate value among the non-nulls
    std::vector<int64_t> nn;
    for (cudf::size_type i = 0; i < n; ++i) {
      if (in_v[i]) { nn.push_back(in[i]); }
    }
    std::sort(nn.begin(), nn.end());
    std::vector<int64_t> ex = nn;
    std::vector<bool> ex_v(nn.size(), true);
    ex.resize(n, int64_t{0});
    ex_v.resize(n, false);
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int64_t>(in.begin(), in.end(), in_v.begin())
        .release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int64_t>(ex.begin(), ex.end(), ex_v.begin())
        .release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
  {  // decimal128 warp tier (24-byte key).
    cudf::size_type const n = 50;
    std::vector<__int128_t> in(n);
    std::vector<bool> in_v(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i]   = static_cast<__int128_t>(n - i) * (i % 2 == 0 ? 1 : -1);  // mixed sign, distinct
      in_v[i] = (i % 6 != 4);
    }
    in[3] = in[2];  // a duplicate value
    std::vector<__int128_t> nn;
    for (cudf::size_type i = 0; i < n; ++i) {
      if (in_v[i]) { nn.push_back(in[i]); }
    }
    std::sort(nn.begin(), nn.end());
    std::vector<__int128_t> ex = nn;
    std::vector<bool> ex_v(nn.size(), true);
    ex.resize(n, __int128_t{0});
    ex_v.resize(n, false);
    auto input = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(in.begin(), in.end(), in_v.begin(), scale)
        .release());
    auto expected = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(ex.begin(), ex.end(), ex_v.begin(), scale)
        .release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
  {  // double warp tier at exactly TIERED_WARP_CAP: zero pad slots, full-tile occupancy of the
     // generic tiered_warp_sort_kernel (the path float/double/chrono/decimal128 share).
    cudf::size_type const n = 64;
    std::vector<double> in(n);
    std::vector<bool> in_v(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i]   = static_cast<double>(n - i);  // distinct, descending
      in_v[i] = (i % 5 != 2);                // scattered nulls
    }
    in[1] = in[0];  // a duplicate value among the non-nulls
    std::vector<double> nn;
    for (cudf::size_type i = 0; i < n; ++i) {
      if (in_v[i]) { nn.push_back(in[i]); }
    }
    std::sort(nn.begin(), nn.end());
    std::vector<double> ex = nn;
    std::vector<bool> ex_v(nn.size(), true);
    ex.resize(n, 0.0);
    ex_v.resize(n, false);
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<double>(in.begin(), in.end(), in_v.begin()).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<double>(ex.begin(), ex.end(), ex_v.begin()).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
}

// int64 no-null mid band on the tiered kernel's network and warp tiers for the eight-byte rep. A
// single row of eight elements lands in the network tier (Batcher-8); one of sixteen in the warp
// tier, where a no-null int64 sorts via the register bitonic (raw key). Values exceed INT32_MAX and
// span the sign boundary; both paths must reproduce the comparison sort's ascending order. The
// no-null int64 mid band no longer prefers CUB `DeviceSegmentedSort` -- the tiered warp kernels
// beat it, so `choose_fixed_width_sort_path` routes the whole no-null int64 range below the
// packed-radix cutoff to the tiered path.
TEST_F(SortListsInt, NumericMidBandInt64Tiered)
{
  auto constexpr big          = int64_t{5} * 1'000 * 1'000 * 1'000;  // > INT32_MAX
  auto const check_single_row = [big](cudf::size_type n) {
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
  };
  check_single_row(8);   // one segment of eight -> network tier (Batcher-8)
  check_single_row(16);  // one segment of sixteen -> warp tier, register bitonic (raw key)
}

// No-null 8-byte chrono (TIMESTAMP/DURATION) in the mid band. `dispatch_storage_type` does not
// reduce chrono to its integer rep, so the CUB fast path (`column_fast_sort_fn`) cannot sort it;
// routing an 8-byte no-null chrono to CUB (as an integer of the same width would go) throws
// CUDF_FAIL. A single row of sixteen (average eight, past the tiered tiny cutoff) must therefore
// stay on the tiered kernel, which encodes chrono via `radix_encode_*`. Guards that routing for a
// timestamp and a duration rep against the comparison oracle.
TEST_F(SortListsInt, NumericMidBandChronoNoNull)
{
  std::vector<int64_t> in(16);
  for (cudf::size_type i = 0; i < 16; ++i) {
    in[i] = (8 - i) * int64_t{1'000'000'000};
  }
  std::vector<int64_t> ex(in);
  std::sort(ex.begin(), ex.end());
  {  // TIMESTAMP_MILLISECONDS (int64 rep)
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms>(in.begin(), in.end()).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms>(ex.begin(), ex.end()).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
  {  // DURATION_MICROSECONDS (int64 rep)
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<cudf::duration_us>(in.begin(), in.end()).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<cudf::duration_us>(ex.begin(), ex.end()).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
}

// DECIMAL128 no-null mid band, the tiered / CUB / tiered trio: the shortest lists take the tiered
// kernel, a mid band the lifted CUB `DeviceSegmentedSort` over the __int128 rep (its shape gate
// passes for a single short segment), and lists above the mid band but below the packed-radix
// cutoff the tiered kernel again. The CUB block also guards the __int128 fast-sort lift against the
// comparison oracle. Values are distinct multiples of 10^30, spanning the sign boundary into the
// high value words.
TEST_F(SortListsInt, NumericMidBandDecimal128Cub)
{
  auto constexpr scale = numeric::scale_type{0};
  auto const k         = [] {
    __int128_t v = 1;
    for (int i = 0; i < 30; ++i) {
      v *= 10;
    }
    return v;
  }();
  auto const check_single_row = [&](cudf::size_type n) {
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
  };
  check_single_row(8);   // average four: tiered kernel
  check_single_row(16);  // average eight: DECIMAL128 mid band -> lifted CUB DeviceSegmentedSort
  check_single_row(40);  // average twenty: above the CUB band, below packed radix -> tiered kernel
  // The exact gate cells: average five is the first cell past the tiered tiny gate (avg > 4),
  // average sixteen the last in-band CUB cell, average seventeen the first above-band tiered cell.
  check_single_row(10);
  check_single_row(32);
  check_single_row(34);
}

// `decimal128_cub_segment_shape_ok` (the DECIMAL128 mid-band CUB-vs-tiered shape gate) is reached
// only by NumericMidBandDecimal128Cub above, which always builds a single segment -- so its
// "oversized segments are sparse" ratio (`oversized * 32 <= num_segments`) is never exercised
// beyond one segment. This pins both outcomes at the same mid-band average list size (in (4, 16]):
// a many-small-plus-two-large shape whose ratio holds (-> lifted CUB), and a few-small shape whose
// ratio fails (-> tiered). Both must match the comparison oracle.
TEST_F(SortListsInt, NumericMidBandDecimal128CubShapeGate)
{
  auto constexpr scale = numeric::scale_type{0};
  auto const check =
    [&](int num_small, cudf::size_type small_sz, int num_large, cudf::size_type large_sz) {
      std::vector<__int128_t> in_vals;
      std::vector<__int128_t> ex_vals;
      std::vector<cudf::size_type> offsets{0};
      auto const add_row = [&](cudf::size_type sz) {
        std::vector<__int128_t> row(sz);
        for (cudf::size_type i = 0; i < sz; ++i) {
          row[i] = static_cast<__int128_t>(sz) - i;
        }
        for (auto const v : row) {
          in_vals.push_back(v);
        }
        std::sort(row.begin(), row.end());
        for (auto const v : row) {
          ex_vals.push_back(v);
        }
        offsets.push_back(static_cast<cudf::size_type>(in_vals.size()));
      };
      for (int r = 0; r < num_small; ++r) {
        add_row(small_sz);
      }
      for (int r = 0; r < num_large; ++r) {
        add_row(large_sz);
      }
      auto const num_rows  = static_cast<cudf::size_type>(num_small + num_large);
      auto const make_list = [&](std::vector<__int128_t> const& vals) {
        cudf::test::fixed_point_column_wrapper<__int128_t> leaf(vals.begin(), vals.end(), scale);
        cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
        return cudf::make_lists_column(num_rows, off.release(), leaf.release(), 0, {});
      };
      auto const input    = make_list(in_vals);
      auto const expected = make_list(ex_vals);
      expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
    };
  check(200, 10, 2, 40);  // avg ~10, oversized=2: 2*32=64 <= 202 -> shape OK  -> lifted CUB
  check(10, 10, 2, 40);   // avg ~15, oversized=2: 2*32=64 >  12  -> shape bad -> tiered
  check(62, 10, 2, 40);   // exact ratio cell: 2*32=64 == 64 segments -> shape still OK (<=) -> CUB
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

// No-null INT32 warp tier: the register bitonic across its three sub-bands (<=16, <=32, <=64). One
// column of segments straddling every tier boundary -- network (8), the bitonic bands (9, 16, 17,
// 32, 33, 48, 64), and a radix outlier (65) -- with INT32_MAX (which radix-encodes to the raw-key
// pad sentinel, so the bitonic pad tie-break must keep it inside the segment), INT32_MIN, and a
// duplicate. The 8/9, 32/33 and 64/65 boundaries are all present in one column. Verified against
// the oracle.
TEST_F(SortListsInt, NumericTieredBitonicNoNullInt32)
{
  std::vector<cudf::size_type> const sizes{8, 9, 16, 17, 32, 33, 48, 64, 65};
  auto constexpr lo = std::numeric_limits<int32_t>::min();
  auto constexpr hi = std::numeric_limits<int32_t>::max();
  std::vector<int32_t> in_vals;
  std::vector<int32_t> ex_vals;
  std::vector<cudf::size_type> offsets{0};
  for (auto const s : sizes) {
    std::vector<int32_t> rv(s);
    for (cudf::size_type i = 0; i < s; ++i) {
      rv[i] = static_cast<int32_t>(s - i);
    }
    rv[0] = hi;                     // a real INT32_MAX (radix-encodes to the raw-key pad sentinel)
    if (s >= 2) { rv[1] = lo; }     // INT32_MIN
    if (s >= 4) { rv[2] = rv[3]; }  // a duplicate value
    for (auto const v : rv) {
      in_vals.push_back(v);
    }
    std::sort(rv.begin(), rv.end());
    for (auto const v : rv) {
      ex_vals.push_back(v);
    }
    offsets.push_back(static_cast<cudf::size_type>(in_vals.size()));
  }
  auto const num_rows  = static_cast<cudf::size_type>(sizes.size());
  auto const make_list = [&](std::vector<int32_t> const& vals) {
    cudf::test::fixed_width_column_wrapper<int32_t> leaf(vals.begin(), vals.end());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
    return cudf::make_lists_column(num_rows, off.release(), leaf.release(), 0, {});
  };
  auto const input    = make_list(in_vals);
  auto const expected = make_list(ex_vals);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// No-null INT64 warp tier, pinning the bitonic/narrow split: the register bitonic handles segments
// up to 32, a raw-key `WarpMergeSort` (narrow) segments 33..64. Sizes 9, 32, 33, 64, 65 pin the
// 32/33 (bitonic->narrow) and 64/65 (warp->radix) boundaries; 40 and 48 add interior narrow-band
// shapes where pad slots coexist with the INT64_MAX element. Values exceed INT32_MAX and include
// INT64_MAX (which encodes to exactly the raw-key pad sentinel, so the merge must keep the real
// element inside the valid range whatever pads exist), INT64_MIN, and a duplicate. Verified against
// the comparison oracle.
TEST_F(SortListsInt, NumericTieredBitonicNarrowNoNullInt64)
{
  std::vector<cudf::size_type> const sizes{9, 32, 33, 40, 48, 64, 65};
  auto constexpr lo  = std::numeric_limits<int64_t>::min();
  auto constexpr hi  = std::numeric_limits<int64_t>::max();
  auto constexpr big = int64_t{5} * 1'000 * 1'000 * 1'000;  // > INT32_MAX
  std::vector<int64_t> in_vals;
  std::vector<int64_t> ex_vals;
  std::vector<cudf::size_type> offsets{0};
  for (auto const s : sizes) {
    std::vector<int64_t> rv(s);
    for (cudf::size_type i = 0; i < s; ++i) {
      rv[i] = (static_cast<int64_t>(s) - i) * big;
    }
    rv[0] = hi;                     // a real INT64_MAX (encodes to the raw-key pad sentinel)
    if (s >= 2) { rv[1] = lo; }     // INT64_MIN
    if (s >= 4) { rv[2] = rv[3]; }  // a duplicate value
    for (auto const v : rv) {
      in_vals.push_back(v);
    }
    std::sort(rv.begin(), rv.end());
    for (auto const v : rv) {
      ex_vals.push_back(v);
    }
    offsets.push_back(static_cast<cudf::size_type>(in_vals.size()));
  }
  auto const num_rows  = static_cast<cudf::size_type>(sizes.size());
  auto const make_list = [&](std::vector<int64_t> const& vals) {
    cudf::test::fixed_width_column_wrapper<int64_t> leaf(vals.begin(), vals.end());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
    return cudf::make_lists_column(num_rows, off.release(), leaf.release(), 0, {});
  };
  auto const input    = make_list(in_vals);
  auto const expected = make_list(ex_vals);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// Null-bearing INT32 and INT64 warp tier: a null-flagged column routes to the tiered path, whose
// warp tier uses the full-warp packed-key `WarpMergeSort` at one item per lane up to 32 then two to
// 64 (the w32x1 shape) for both widths. One column per width with segments 8, 9, 32, 33, 64, 65
// (spanning the network / warp / radix tiers and the 8/9, 32/33, 64/65 boundaries), scattered nulls
// placed last, and INT_MIN/MAX plus a duplicate among the non-nulls. Verified against the
// comparison oracle.
TEST_F(SortListsInt, NumericTieredW32x1Nulls)
{
  std::vector<cudf::size_type> const sizes{8, 9, 32, 33, 64, 65};
  auto const run = [&](auto tag) {
    using T           = decltype(tag);
    auto constexpr lo = std::numeric_limits<T>::min();
    auto constexpr hi = std::numeric_limits<T>::max();
    std::vector<T> in_vals;
    std::vector<bool> in_valids;
    std::vector<T> ex_vals;
    std::vector<bool> ex_valids;
    std::vector<cudf::size_type> offsets{0};
    for (auto const s : sizes) {
      std::vector<T> rv(s);
      std::vector<bool> rok(s);
      for (cudf::size_type i = 0; i < s; ++i) {
        rv[i]  = static_cast<T>(s - i);
        rok[i] = (i % 5 != 2);  // scattered element nulls
      }
      rv[0] = hi;
      if (s >= 2) { rv[1] = lo; }
      if (s >= 4) { rv[2] = rv[3]; }  // a duplicate among the non-nulls
      std::vector<T> nn;
      for (cudf::size_type i = 0; i < s; ++i) {
        in_vals.push_back(rv[i]);
        in_valids.push_back(rok[i]);
        if (rok[i]) { nn.push_back(rv[i]); }
      }
      std::sort(nn.begin(), nn.end());
      for (auto const v : nn) {
        ex_vals.push_back(v);
        ex_valids.push_back(true);
      }
      for (cudf::size_type k = static_cast<cudf::size_type>(nn.size()); k < s; ++k) {
        ex_vals.push_back(T{0});
        ex_valids.push_back(false);
      }
      offsets.push_back(static_cast<cudf::size_type>(in_vals.size()));
    }
    auto const num_rows  = static_cast<cudf::size_type>(sizes.size());
    auto const make_list = [&](std::vector<T> const& vals, std::vector<bool> const& valids) {
      cudf::test::fixed_width_column_wrapper<T> leaf(vals.begin(), vals.end(), valids.begin());
      cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
      return cudf::make_lists_column(num_rows, off.release(), leaf.release(), 0, {});
    };
    auto const input    = make_list(in_vals, in_valids);
    auto const expected = make_list(ex_vals, ex_valids);
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  };
  run(int32_t{});
  run(int64_t{});
}

// DECIMAL128 no-null long lists route to the full-column packed radix (Site A), which picks the
// narrowest lossless key from the value range: a range under 2^32 -> min-biased uint64 (eight
// passes), fitting int64 -> prefix_key96 (twelve passes), else the two-phase hi64/lo64 sort. One
// single row of 250 values per regime (avg 125 >= 100 -> Site A), including a wide row engineered
// with two sign clusters that share a high word so the two-phase second pass fires. Verified
// against the oracle.
TEST_F(SortListsInt, NumericDecimal128ComposedSiteA)
{
  auto constexpr scale    = numeric::scale_type{0};
  cudf::size_type const n = 250;  // single row, avg 125 >= 100 -> full-column packed radix (Site A)
  auto const check        = [&](std::vector<__int128_t> const& in) {
    std::vector<__int128_t> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(in.begin(), in.end(), scale).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(ex.begin(), ex.end(), scale).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  };
  {  // small: value range < 2^32 -> min-biased uint64 key
    std::vector<__int128_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = static_cast<__int128_t>(n / 2 - i);
    }
    check(in);
  }
  {  // fits int64 (range > 2^32) -> prefix_key96 key
    auto const step = static_cast<__int128_t>(int64_t{1} << 40);  // 2^40 > 2^32
    std::vector<__int128_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = static_cast<__int128_t>(n / 2 - i) * step;
    }
    check(in);
  }
  {  // wide, distinct high words -> two-phase, phase one resolves (no tie pass)
    auto const step = static_cast<__int128_t>(1) << 90;  // > int64, distinct hi64
    std::vector<__int128_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = static_cast<__int128_t>(n / 2 - i) * step;
    }
    check(in);
  }
  {  // wide with shared high words -> two-phase second pass fires (two sign clusters, distinct
     // lo64)
    auto const base = static_cast<__int128_t>(1) << 100;
    std::vector<__int128_t> in;
    for (int i = 0; i < 125; ++i) {
      in.push_back(-base + i);
    }
    for (int i = 0; i < 125; ++i) {
      in.push_back(base + i);
    }  // 250 total -> avg 125 -> Site A
    check(in);
  }
}

// DECIMAL128 with nulls routes to the tiered path; a segment above the warp cap (64) becomes a
// radix outlier sorted by the compact-large-segment path (Site B), which uses the same range-gated
// key selection. One single row of 200 elements: null-mixed with a wide two-cluster range
// (two-phase second pass fires, nulls excluded and placed last), and an all-null row (degenerate
// range -> the cheapest key, every element a position-final null). Verified against the comparison
// oracle.
TEST_F(SortListsInt, NumericDecimal128ComposedSiteB)
{
  auto constexpr scale    = numeric::scale_type{0};
  cudf::size_type const n = 200;  // one segment > 64 -> radix-tier outlier -> compact path (Site B)
  auto const check        = [&](std::vector<__int128_t> const& in, std::vector<bool> const& valid) {
    std::vector<__int128_t> nn;
    for (cudf::size_type i = 0; i < n; ++i) {
      if (valid[i]) { nn.push_back(in[i]); }
    }
    std::sort(nn.begin(), nn.end());
    std::vector<__int128_t> ex = nn;
    std::vector<bool> ex_v(nn.size(), true);
    ex.resize(n, __int128_t{0});
    ex_v.resize(n, false);
    auto input = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(in.begin(), in.end(), valid.begin(), scale)
        .release());
    auto expected = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(ex.begin(), ex.end(), ex_v.begin(), scale)
        .release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  };
  {  // null-mixed wide with shared high words: two-phase second pass fires, nulls placed last
    auto const base = static_cast<__int128_t>(1) << 100;
    std::vector<__int128_t> in;
    std::vector<bool> valid;
    for (int i = 0; i < 100; ++i) {
      in.push_back(-base + i);
      valid.push_back(i % 7 != 3);
    }
    for (int i = 0; i < 100; ++i) {
      in.push_back(base + i);
      valid.push_back(i % 5 != 2);
    }
    check(in, valid);
  }
  {  // all-null: degenerate range -> cheapest key; every element sorts as a position-final null
    std::vector<__int128_t> const in(n, __int128_t{0});
    std::vector<bool> const valid(n, false);
    check(in, valid);
  }
}

// The Site A range gate's 2^32 boundary: a span of exactly 2^32 - 1 takes the min-biased uint64 key
// (the biased value still fits uint32), exactly 2^32 the wider prefix_key96 (it would overflow).
// One single row of 250 no-null values (avg 125 >= 100 -> Site A) with min 0 and max at each
// boundary; both must reproduce the comparison order, proving the biased key is lossless up to the
// boundary.
TEST_F(SortListsInt, NumericDecimal128RangeGateBoundary)
{
  auto constexpr scale    = numeric::scale_type{0};
  cudf::size_type const n = 250;  // single row, avg 125 >= 100 -> Site A range gate
  auto const check        = [&](__int128_t span) {
    std::vector<__int128_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = (span / (n - 1)) * i;
    }
    in[n - 1] = span;  // exact max, so the range is [0, span]
    std::vector<__int128_t> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(in.begin(), in.end(), scale).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(ex.begin(), ex.end(), scale).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  };
  check((static_cast<__int128_t>(1) << 32) - 1);  // span 2^32 - 1 -> min-biased uint64 (lossless)
  check(static_cast<__int128_t>(1) << 32);        // span 2^32     -> prefix_key96
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
