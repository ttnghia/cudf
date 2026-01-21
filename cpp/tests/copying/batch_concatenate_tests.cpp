/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <numeric>
#include <vector>

/**
 * @brief Test fixture for batch_concatenate tests.
 *
 * These tests focus on edge cases for the bitmask concatenation kernel,
 * particularly around word boundaries (32-bit alignment).
 */
struct BatchConcatenateTest : public cudf::test::BaseFixture {};

// Helper to create a column with specific validity pattern
template <typename T>
cudf::test::fixed_width_column_wrapper<T> make_column_with_validity(
  std::vector<T> const& values, std::vector<bool> const& validity)
{
  return cudf::test::fixed_width_column_wrapper<T>(values.begin(), values.end(), validity.begin());
}

// ============================================================================
// Basic functionality tests
// ============================================================================

TEST_F(BatchConcatenateTest, SingleColumn)
{
  auto col = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3, 4, 5}, {1, 0, 1, 0, 1});
  std::vector<cudf::column_view> columns = {col};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TEST_F(BatchConcatenateTest, TwoColumnsNoNulls)
{
  auto col1 = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3});
  auto col2 = cudf::test::fixed_width_column_wrapper<int32_t>({4, 5, 6});
  std::vector<cudf::column_view> columns = {col1, col2};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TEST_F(BatchConcatenateTest, TwoColumnsWithNulls)
{
  auto col1 = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3}, {1, 0, 1});
  auto col2 = cudf::test::fixed_width_column_wrapper<int32_t>({4, 5, 6}, {0, 1, 0});
  std::vector<cudf::column_view> columns = {col1, col2};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

// ============================================================================
// Word boundary edge cases (32-bit alignment)
// ============================================================================

TEST_F(BatchConcatenateTest, ExactlyOneWord)
{
  // 32 elements = exactly 1 word
  std::vector<int32_t> values(32);
  std::iota(values.begin(), values.end(), 0);
  std::vector<bool> validity(32, true);
  for (int i = 0; i < 32; i += 2) {
    validity[i] = false;
  }

  auto col                               = make_column_with_validity(values, validity);
  std::vector<cudf::column_view> columns = {col};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TEST_F(BatchConcatenateTest, TwoColumnsExactlyOneWordEach)
{
  // Each column has exactly 32 elements (1 word)
  std::vector<int32_t> values1(32), values2(32);
  std::iota(values1.begin(), values1.end(), 0);
  std::iota(values2.begin(), values2.end(), 32);

  std::vector<bool> validity1(32), validity2(32);
  for (int i = 0; i < 32; ++i) {
    validity1[i] = (i % 3) != 0;
    validity2[i] = (i % 5) != 0;
  }

  auto col1                              = make_column_with_validity(values1, validity1);
  auto col2                              = make_column_with_validity(values2, validity2);
  std::vector<cudf::column_view> columns = {col1, col2};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TEST_F(BatchConcatenateTest, ColumnSizesNotAlignedToWord)
{
  // Column 1: 17 elements (not word-aligned)
  // Column 2: 23 elements (not word-aligned)
  // Total: 40 elements = 1 full word + 8 bits
  std::vector<int32_t> values1(17), values2(23);
  std::iota(values1.begin(), values1.end(), 0);
  std::iota(values2.begin(), values2.end(), 17);

  std::vector<bool> validity1(17, true);
  std::vector<bool> validity2(23, true);
  // Set some nulls
  validity1[5] = validity1[10] = validity1[16] = false;
  validity2[0] = validity2[11] = validity2[22] = false;

  auto col1                              = make_column_with_validity(values1, validity1);
  auto col2                              = make_column_with_validity(values2, validity2);
  std::vector<cudf::column_view> columns = {col1, col2};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TEST_F(BatchConcatenateTest, OutputWordSpansTwoColumns)
{
  // Column 1: 30 elements (output word 0 has bits 0-29 from col1)
  // Column 2: 10 elements (output word 0 has bits 30-31 from col2, word 1 has bits 0-7 from col2)
  std::vector<int32_t> values1(30), values2(10);
  std::iota(values1.begin(), values1.end(), 0);
  std::iota(values2.begin(), values2.end(), 30);

  std::vector<bool> validity1(30, true);
  std::vector<bool> validity2(10, true);
  // Nulls at the boundary
  validity1[28] = validity1[29] = false;  // Last 2 bits of col1
  validity2[0] = validity2[1] = false;    // First 2 bits of col2

  auto col1                              = make_column_with_validity(values1, validity1);
  auto col2                              = make_column_with_validity(values2, validity2);
  std::vector<cudf::column_view> columns = {col1, col2};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TEST_F(BatchConcatenateTest, OutputWordSpansThreeColumns)
{
  // Three small columns that fit in one output word
  // Column 1: 10 elements
  // Column 2: 12 elements
  // Column 3: 8 elements
  // Total: 30 elements < 32
  std::vector<int32_t> values1(10), values2(12), values3(8);
  std::iota(values1.begin(), values1.end(), 0);
  std::iota(values2.begin(), values2.end(), 10);
  std::iota(values3.begin(), values3.end(), 22);

  std::vector<bool> validity1(10, true);
  std::vector<bool> validity2(12, false);  // All nulls
  std::vector<bool> validity3(8, true);

  auto col1                              = make_column_with_validity(values1, validity1);
  auto col2                              = make_column_with_validity(values2, validity2);
  auto col3                              = make_column_with_validity(values3, validity3);
  std::vector<cudf::column_view> columns = {col1, col2, col3};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

// ============================================================================
// Empty column edge cases
// ============================================================================

TEST_F(BatchConcatenateTest, EmptyColumnInMiddle)
{
  auto col1  = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3}, {1, 0, 1});
  auto empty = cudf::test::fixed_width_column_wrapper<int32_t>{};
  auto col2  = cudf::test::fixed_width_column_wrapper<int32_t>({4, 5, 6}, {0, 1, 0});
  std::vector<cudf::column_view> columns = {col1, empty, col2};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TEST_F(BatchConcatenateTest, MultipleEmptyColumns)
{
  auto col1   = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2}, {1, 0});
  auto empty1 = cudf::test::fixed_width_column_wrapper<int32_t>{};
  auto empty2 = cudf::test::fixed_width_column_wrapper<int32_t>{};
  auto col2   = cudf::test::fixed_width_column_wrapper<int32_t>({3, 4}, {0, 1});
  std::vector<cudf::column_view> columns = {col1, empty1, empty2, col2};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TEST_F(BatchConcatenateTest, EmptyColumnAtWordBoundary)
{
  // 32 elements followed by empty column followed by more elements
  std::vector<int32_t> values1(32), values2(10);
  std::iota(values1.begin(), values1.end(), 0);
  std::iota(values2.begin(), values2.end(), 32);
  std::vector<bool> validity1(32, true);
  std::vector<bool> validity2(10, true);
  validity1[31] = false;
  validity2[0]  = false;

  auto col1                              = make_column_with_validity(values1, validity1);
  auto empty                             = cudf::test::fixed_width_column_wrapper<int32_t>{};
  auto col2                              = make_column_with_validity(values2, validity2);
  std::vector<cudf::column_view> columns = {col1, empty, col2};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

// ============================================================================
// Mixed null mask scenarios
// ============================================================================

TEST_F(BatchConcatenateTest, MixedNullAndNoNullColumns)
{
  // Column without nulls (null_mask == nullptr)
  auto col_no_null = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3, 4, 5});
  // Column with nulls
  auto col_with_null = cudf::test::fixed_width_column_wrapper<int32_t>({6, 7, 8}, {1, 0, 1});
  // Another column without nulls
  auto col_no_null2 = cudf::test::fixed_width_column_wrapper<int32_t>({9, 10});

  std::vector<cudf::column_view> columns = {col_no_null, col_with_null, col_no_null2};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TEST_F(BatchConcatenateTest, NullMaskColumnFollowedByNoMask)
{
  // This tests the case where we read all 1s for columns without masks
  auto col_with_null = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3}, {0, 1, 0});
  auto col_no_null   = cudf::test::fixed_width_column_wrapper<int32_t>({4, 5, 6, 7, 8});

  std::vector<cudf::column_view> columns = {col_with_null, col_no_null};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

// ============================================================================
// Source bit offset tests (sliced columns)
// ============================================================================

TEST_F(BatchConcatenateTest, SlicedColumnWithOffset)
{
  // Create a column and slice it to create an offset
  std::vector<int32_t> values(100);
  std::iota(values.begin(), values.end(), 0);
  std::vector<bool> validity(100);
  for (int i = 0; i < 100; ++i) {
    validity[i] = (i % 2) == 0;
  }

  auto full_col = make_column_with_validity(values, validity);
  auto slices   = cudf::slice(full_col, {10, 50});  // Offset of 10 bits

  std::vector<cudf::column_view> columns = {slices[0]};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TEST_F(BatchConcatenateTest, TwoSlicedColumnsWithDifferentOffsets)
{
  std::vector<int32_t> values1(100), values2(100);
  std::iota(values1.begin(), values1.end(), 0);
  std::iota(values2.begin(), values2.end(), 100);
  std::vector<bool> validity1(100), validity2(100);
  for (int i = 0; i < 100; ++i) {
    validity1[i] = (i % 3) != 0;
    validity2[i] = (i % 5) != 0;
  }

  auto full_col1 = make_column_with_validity(values1, validity1);
  auto full_col2 = make_column_with_validity(values2, validity2);

  // Different offsets: 7 and 13 (both not word-aligned)
  auto slices1 = cudf::slice(full_col1, {7, 47});   // 40 elements, offset 7
  auto slices2 = cudf::slice(full_col2, {13, 63});  // 50 elements, offset 13

  std::vector<cudf::column_view> columns = {slices1[0], slices2[0]};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TEST_F(BatchConcatenateTest, SlicedColumnCrossingSourceWordBoundary)
{
  // Create a slice that requires reading from two source words
  std::vector<int32_t> values(100);
  std::iota(values.begin(), values.end(), 0);
  std::vector<bool> validity(100, true);
  // Set nulls at word boundaries
  validity[31] = validity[32] = validity[63] = validity[64] = false;

  auto full_col = make_column_with_validity(values, validity);
  // Slice starting at bit 20, so first output word reads bits 20-51 from source
  // which spans words 0 and 1
  auto slices = cudf::slice(full_col, {20, 80});

  std::vector<cudf::column_view> columns = {slices[0]};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

// ============================================================================
// Large column tests
// ============================================================================

TEST_F(BatchConcatenateTest, LargeColumnMultipleWords)
{
  // 1000 elements = 31.25 words
  std::vector<int32_t> values(1000);
  std::iota(values.begin(), values.end(), 0);
  std::vector<bool> validity(1000);
  for (int i = 0; i < 1000; ++i) {
    validity[i] = (i % 7) != 0;  // Every 7th is null
  }

  auto col                               = make_column_with_validity(values, validity);
  std::vector<cudf::column_view> columns = {col};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TEST_F(BatchConcatenateTest, ManySmallColumns)
{
  // Many columns of varying small sizes
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> column_wrappers;
  std::vector<cudf::column_view> columns;

  int value = 0;
  for (int size = 1; size <= 50; ++size) {
    std::vector<int32_t> values(size);
    std::vector<bool> validity(size);
    for (int i = 0; i < size; ++i) {
      values[i]   = value++;
      validity[i] = (i + size) % 2 == 0;
    }
    column_wrappers.push_back(make_column_with_validity(values, validity));
  }

  for (auto& w : column_wrappers) {
    columns.push_back(w);
  }

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

// ============================================================================
// Specific bit pattern tests
// ============================================================================

TEST_F(BatchConcatenateTest, AllNulls)
{
  std::vector<int32_t> values(64, 0);
  std::vector<bool> validity(64, false);

  auto col                               = make_column_with_validity(values, validity);
  std::vector<cudf::column_view> columns = {col};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TEST_F(BatchConcatenateTest, AllValid)
{
  std::vector<int32_t> values(64);
  std::iota(values.begin(), values.end(), 0);
  std::vector<bool> validity(64, true);

  auto col                               = make_column_with_validity(values, validity);
  std::vector<cudf::column_view> columns = {col};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TEST_F(BatchConcatenateTest, AlternatingValidNull)
{
  std::vector<int32_t> values(64);
  std::iota(values.begin(), values.end(), 0);
  std::vector<bool> validity(64);
  for (int i = 0; i < 64; ++i) {
    validity[i] = (i % 2) == 0;
  }

  auto col                               = make_column_with_validity(values, validity);
  std::vector<cudf::column_view> columns = {col};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

// ============================================================================
// Struct column tests
// ============================================================================

TEST_F(BatchConcatenateTest, StructColumnWithNulls)
{
  // Struct with two int32 children
  auto child1_col1 = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3}, {1, 1, 0});
  auto child2_col1 = cudf::test::fixed_width_column_wrapper<int32_t>({10, 20, 30}, {1, 0, 1});
  auto struct_col1 = cudf::test::structs_column_wrapper({child1_col1, child2_col1}, {1, 0, 1});

  auto child1_col2 = cudf::test::fixed_width_column_wrapper<int32_t>({4, 5}, {0, 1});
  auto child2_col2 = cudf::test::fixed_width_column_wrapper<int32_t>({40, 50}, {1, 1});
  auto struct_col2 = cudf::test::structs_column_wrapper({child1_col2, child2_col2}, {1, 1});

  std::vector<cudf::column_view> columns = {struct_col1, struct_col2};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

// ============================================================================
// Edge case: Single element columns
// ============================================================================

TEST_F(BatchConcatenateTest, ManySingleElementColumns)
{
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> column_wrappers;
  std::vector<cudf::column_view> columns;

  // 100 single-element columns with alternating validity
  for (int i = 0; i < 100; ++i) {
    column_wrappers.push_back(cudf::test::fixed_width_column_wrapper<int32_t>({i}, {i % 2 == 0}));
  }

  for (auto& w : column_wrappers) {
    columns.push_back(w);
  }

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

// ============================================================================
// Typed tests for different data types
// ============================================================================

template <typename T>
struct BatchConcatenateTypedTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(BatchConcatenateTypedTest, cudf::test::FixedWidthTypes);

TYPED_TEST(BatchConcatenateTypedTest, BasicConcatenation)
{
  using T = TypeParam;

  auto col1 = cudf::test::fixed_width_column_wrapper<T, int32_t>({1, 2, 3, 4, 5}, {1, 0, 1, 0, 1});
  auto col2 = cudf::test::fixed_width_column_wrapper<T, int32_t>({6, 7, 8}, {0, 1, 1});
  std::vector<cudf::column_view> columns = {col1, col2};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TYPED_TEST(BatchConcatenateTypedTest, WordBoundaryTest)
{
  using T = TypeParam;

  // 31 elements + 33 elements = 64 elements (2 words)
  // Column boundary at bit 31 (inside first word)
  std::vector<int32_t> values1(31), values2(33);
  std::iota(values1.begin(), values1.end(), 0);
  std::iota(values2.begin(), values2.end(), 31);

  std::vector<bool> validity1(31, true);
  std::vector<bool> validity2(33, true);
  validity1[30] = false;  // Last bit of first column
  validity2[0]  = false;  // First bit of second column

  auto col1 = cudf::test::fixed_width_column_wrapper<T, int32_t>(
    values1.begin(), values1.end(), validity1.begin());
  auto col2 = cudf::test::fixed_width_column_wrapper<T, int32_t>(
    values2.begin(), values2.end(), validity2.begin());

  std::vector<cudf::column_view> columns = {col1, col2};

  auto result = cudf::detail::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto expected = cudf::concatenate(columns);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}
