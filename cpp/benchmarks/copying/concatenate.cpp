/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <benchmarks/common/generate_input.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <random>
#include <vector>

static void bench_concatenate(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const nulls    = static_cast<cudf::size_type>(state.get_float64("nulls"));

  auto input = create_sequence_table(
    cycle_dtypes({cudf::type_to_id<int64_t>()}, num_cols), row_count{num_rows}, nulls);
  auto input_columns = input->view();
  auto column_views  = std::vector<cudf::column_view>(input_columns.begin(), input_columns.end());

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_global_memory_reads<int64_t>(num_rows * num_cols);
  state.add_global_memory_writes<int64_t>(num_rows * num_cols);

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { auto result = cudf::concatenate(column_views); });
}

NVBENCH_BENCH(bench_concatenate)
  .set_name("concatenate")
  .add_int64_axis("num_rows", {64, 512, 4096, 32768, 262144})
  .add_int64_axis("num_cols", {2, 8, 64, 512, 1024})
  .add_float64_axis("nulls", {0.0, 0.3});

static void bench_concatenate_strings(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols  = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const nulls     = static_cast<cudf::size_type>(state.get_float64("nulls"));

  data_profile const profile =
    data_profile_builder()
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, 0, row_width)
      .null_probability(nulls);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  auto const input  = column->view();

  auto column_views = std::vector<cudf::column_view>(num_cols, input);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const sv = cudf::strings_column_view(input);
  state.add_global_memory_reads<int8_t>(sv.chars_size(stream) * num_cols);
  state.add_global_memory_writes<int64_t>(sv.chars_size(stream) * num_cols);

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { auto result = cudf::concatenate(column_views); });
}

NVBENCH_BENCH(bench_concatenate_strings)
  .set_name("concatenate_strings")
  .add_int64_axis("num_rows", {100000, 200000, 500000, 1000000})
  .add_int64_axis("num_cols", {2, 8, 64, 256})
  .add_int64_axis("row_width", {32})
  .add_float64_axis("nulls", {0.0, 0.3});

// Helper to create a struct column with fixed-width children
static std::unique_ptr<cudf::column> create_struct_column(cudf::size_type num_rows,
                                                          cudf::size_type num_children,
                                                          cudf::size_type depth,
                                                          double null_probability)
{
  using Type           = int32_t;
  using column_wrapper = cudf::test::fixed_width_column_wrapper<Type>;

  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 100);

  // Create leaf columns with values in the range [0,100)
  std::vector<column_wrapper> columns;
  columns.reserve(num_children);
  std::generate_n(std::back_inserter(columns), num_children, [&]() {
    auto const elements = cudf::detail::make_counting_transform_iterator(
      0, [&](auto row) { return distribution(generator); });
    if (null_probability == 0.0) { return column_wrapper(elements, elements + num_rows); }
    auto valids = cudf::detail::make_counting_transform_iterator(
      0, [&](auto i) { return distribution(generator) >= (null_probability * 100); });
    return column_wrapper(elements, elements + num_rows, valids);
  });

  std::vector<std::unique_ptr<cudf::column>> cols;
  std::transform(columns.begin(), columns.end(), std::back_inserter(cols), [](column_wrapper& col) {
    return col.release();
  });

  std::vector<std::unique_ptr<cudf::column>> child_cols = std::move(cols);

  // Nest the child columns in structs up to the desired depth
  for (cudf::size_type i = 0; i < depth; i++) {
    std::vector<bool> struct_validity;
    std::uniform_int_distribution<int> bool_distribution(0, 100);
    std::generate_n(std::back_inserter(struct_validity), num_rows, [&]() {
      return null_probability == 0.0 || bool_distribution(generator) >= (null_probability * 100);
    });
    cudf::test::structs_column_wrapper struct_col(std::move(child_cols), struct_validity);
    child_cols = std::vector<std::unique_ptr<cudf::column>>{};
    child_cols.push_back(struct_col.release());
  }

  return std::move(child_cols[0]);
}

// Benchmark for concatenating plain fixed-width types (uses batch_concatenate path)
static void bench_concatenate_fixed_width(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const nulls    = state.get_float64("nulls");

  // Create a table with multiple int32 columns
  auto input = create_sequence_table(
    cycle_dtypes({cudf::type_to_id<int32_t>()}, num_cols), row_count{num_rows}, nulls);
  auto input_columns = input->view();
  auto column_views  = std::vector<cudf::column_view>(input_columns.begin(), input_columns.end());

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_global_memory_reads<int32_t>(num_rows * num_cols);
  state.add_global_memory_writes<int32_t>(num_rows * num_cols);

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { auto result = cudf::concatenate(column_views); });
}

NVBENCH_BENCH(bench_concatenate_fixed_width)
  .set_name("concatenate_fixed_width")
  .add_int64_axis("num_rows", {100000, 200000, 500000, 1000000})
  .add_int64_axis("num_cols", {64, 512, 1024})
  .add_float64_axis("nulls", {0.0, 0.3});

// Benchmark for concatenating struct columns (uses batch_concatenate path)
static void bench_concatenate_structs(nvbench::state& state)
{
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols     = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const num_children = static_cast<cudf::size_type>(state.get_int64("num_children"));
  auto const depth        = static_cast<cudf::size_type>(state.get_int64("depth"));
  auto const nulls        = state.get_float64("nulls");

  // Create struct columns
  auto column = create_struct_column(num_rows, num_children, depth, nulls);
  auto input  = column->view();

  auto column_views = std::vector<cudf::column_view>(num_cols, input);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  // Estimate memory: struct column with num_children int32 columns, nested depth times
  // Each leaf column has num_rows * sizeof(int32_t) bytes
  auto const leaf_bytes = static_cast<int64_t>(num_rows) * sizeof(int32_t) * num_children;
  state.add_global_memory_reads<int8_t>(leaf_bytes * num_cols);
  state.add_global_memory_writes<int8_t>(leaf_bytes * num_cols);

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { auto result = cudf::concatenate(column_views); });
}

NVBENCH_BENCH(bench_concatenate_structs)
  .set_name("concatenate_structs")
  .add_int64_axis("num_rows", {100000, 200000, 500000})
  .add_int64_axis("num_cols", {100})
  .add_int64_axis("num_children", {2, 4, 8})
  .add_int64_axis("depth", {1, 2, 4})
  .add_float64_axis("nulls", {0.0, 0.3});

// Helper to create a nested list column with specified nesting level
// nesting_level=1 -> LIST<INT32>
// nesting_level=2 -> LIST<LIST<INT32>>
// nesting_level=3 -> LIST<LIST<LIST<INT32>>>
// etc.
static std::unique_ptr<cudf::column> create_nested_list_column(cudf::size_type num_rows,
                                                               cudf::size_type avg_list_size,
                                                               cudf::size_type nesting_level,
                                                               double null_probability)
{
  using int_column_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;

  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 100);
  std::uniform_int_distribution<int> size_distribution(1, avg_list_size * 2 - 1);

  // Start with innermost level (leaf INT32 values)
  // For simplicity, create fixed-size lists at each level
  auto const total_leaf_elements =
    static_cast<cudf::size_type>(std::pow(avg_list_size, nesting_level)) * num_rows;

  // Create leaf int32 values
  auto const elements = cudf::detail::make_counting_transform_iterator(
    0, [&](auto row) { return distribution(generator); });

  std::unique_ptr<cudf::column> current_column;
  if (null_probability == 0.0) {
    int_column_wrapper leaf_col(elements, elements + total_leaf_elements);
    current_column = leaf_col.release();
  } else {
    auto valids = cudf::detail::make_counting_transform_iterator(
      0, [&](auto i) { return distribution(generator) >= (null_probability * 100); });
    int_column_wrapper leaf_col(elements, elements + total_leaf_elements, valids);
    current_column = leaf_col.release();
  }

  // Wrap in list levels from innermost to outermost
  auto current_num_elements = total_leaf_elements;
  for (cudf::size_type level = 0; level < nesting_level; ++level) {
    auto const num_lists = current_num_elements / avg_list_size;

    // Create offsets for this level
    std::vector<cudf::size_type> offsets;
    offsets.reserve(num_lists + 1);
    offsets.push_back(0);
    for (cudf::size_type i = 0; i < num_lists; ++i) {
      offsets.push_back(offsets.back() + avg_list_size);
    }

    auto offsets_col =
      cudf::test::fixed_width_column_wrapper<cudf::size_type>(offsets.begin(), offsets.end());

    // Create validity mask for lists
    rmm::device_buffer null_mask;
    cudf::size_type null_count = 0;
    if (null_probability > 0.0) {
      std::vector<bool> validity;
      validity.reserve(num_lists);
      for (cudf::size_type i = 0; i < num_lists; ++i) {
        validity.push_back(distribution(generator) >= (null_probability * 100));
        if (!validity.back()) { ++null_count; }
      }
      auto validity_iter = validity.begin();
      auto [mask, count] =
        cudf::test::detail::make_null_mask(validity_iter, validity_iter + num_lists);
      null_mask = std::move(mask);
    }

    current_column       = cudf::make_lists_column(num_lists,
                                             offsets_col.release(),
                                             std::move(current_column),
                                             null_count,
                                             std::move(null_mask));
    current_num_elements = num_lists;
  }

  return current_column;
}

// Benchmark for concatenating list columns (uses batch_concatenate path)
static void bench_concatenate_lists(nvbench::state& state)
{
  auto const num_rows      = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols      = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const avg_list_size = static_cast<cudf::size_type>(state.get_int64("avg_list_size"));
  auto const nesting_level = static_cast<cudf::size_type>(state.get_int64("nesting_level"));
  auto const nulls         = state.get_float64("nulls");

  // Create nested list column
  auto column = create_nested_list_column(num_rows, avg_list_size, nesting_level, nulls);
  auto input  = column->view();

  auto column_views = std::vector<cudf::column_view>(num_cols, input);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  // Estimate memory: leaf data size
  auto const leaf_elements =
    static_cast<int64_t>(std::pow(avg_list_size, nesting_level)) * num_rows;
  auto const leaf_bytes = leaf_elements * sizeof(int32_t);
  state.add_global_memory_reads<int8_t>(leaf_bytes * num_cols);
  state.add_global_memory_writes<int8_t>(leaf_bytes * num_cols);

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { auto result = cudf::concatenate(column_views); });
}

NVBENCH_BENCH(bench_concatenate_lists)
  .set_name("concatenate_lists")
  .add_int64_axis("num_rows", {50000, 100000, 200000})
  .add_int64_axis("num_cols", {64, 128})
  .add_int64_axis("avg_list_size", {5})
  .add_int64_axis("nesting_level", {1, 2, 3})
  .add_float64_axis("nulls", {0.0});
