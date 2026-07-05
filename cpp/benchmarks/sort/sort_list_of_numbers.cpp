/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <nvbench/nvbench.cuh>

#include <cstdint>

// Benchmark for cudf::lists::sort_lists on a LIST<numeric> column -- the operation Spark array_sort
// lowers to. The type axis spans the integral, floating-point, and decimal128 fixed-width paths.
// The segmented sort routes in a single split today: a no-null integral or non-DECIMAL128
// fixed-point column goes to CUB DeviceSegmentedSort while the average list size stays under 100 or
// the total row count under 2^18; everything else (float, double, DECIMAL128, or any nulls) takes
// the generic fallback, a lexicographic comparator sort over a prepended segment-id column. The
// shape axes bracket that boundary: max_list_size 4 and 32 keep eligible types on the CUB path at
// both row counts; 256 (average ~128) stays on it only at 100k rows (total-size arm) and falls back
// at 1M rows. null_frequency 0.1 forces every type onto the fallback; 0 exercises the no-null
// routing. The order and null_order axes complete the sort-parameter matrix: the CUB path honors
// order (its gate excludes nulls) and the fallback honors both.
template <typename Type>
void bench_sort_list_of_numbers(nvbench::state& state, nvbench::type_list<Type>)
{
  auto const num_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const max_list_size  = static_cast<cudf::size_type>(state.get_int64("max_list_size"));
  auto const null_frequency = state.get_float64("null_frequency");
  auto const column_order =
    state.get_string("order") == "DESC" ? cudf::order::DESCENDING : cudf::order::ASCENDING;
  auto const null_precedence =
    state.get_string("null_order") == "BEFORE" ? cudf::null_order::BEFORE : cudf::null_order::AFTER;

  // Build a LIST<Type> column: list length uniform in [0, max_list_size], leaf values uniform in a
  // fixed range (cardinality 0 leaves the distinct count uncapped). null_frequency applies nulls at
  // both the list-row and element levels, so a non-zero value gives the sort leaf nulls to place
  // per the null_order axis.
  data_profile profile =
    data_profile_builder()
      .list_type(cudf::type_to_id<Type>())
      .list_depth(1)
      .distribution(cudf::type_id::LIST, distribution_id::UNIFORM, 0, max_list_size)
      .null_probability(null_frequency)
      .cardinality(0);
  // Leaf value distribution: the bound type selects the generator overload, so it must match the
  // leaf -- integral bounds populate the integer distribution, floating point needs double bounds,
  // and a fixed-point leaf needs the five-argument scale overload; a mismatched call is a no-op.
  if constexpr (cudf::is_floating_point<Type>()) {
    profile.set_distribution_params(
      cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0., 1'000'000.);
  } else if constexpr (cudf::is_fixed_point<Type>()) {
    profile.set_distribution_params(
      cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0, 1'000'000, numeric::scale_type{0});
  } else {
    profile.set_distribution_params(
      cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0, 1'000'000);
  }

  auto const table = create_random_table({cudf::type_id::LIST}, row_count{num_rows}, profile);

  auto const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  auto const input = cudf::lists_column_view{table->view().column(0)};

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    cudf::lists::sort_lists(
      input, column_order, null_precedence, stream, cudf::get_current_device_resource_ref());
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

// Chrono is not on the type axis: timestamps are neither integral nor fixed-point to the fast-path
// gate, so they take the same comparator fallback the float/double cells already measure.
// decimal128 is likewise excluded from the CUB path (128-bit storage exceeds its fast-sort
// support) and needs its own type-string mapping.
NVBENCH_DECLARE_TYPE_STRINGS(numeric::decimal128, "decimal128", "decimal128");

NVBENCH_BENCH_TYPES(
  bench_sort_list_of_numbers,
  NVBENCH_TYPE_AXES(
    nvbench::type_list<std::int32_t, std::int64_t, float, double, numeric::decimal128>))
  .set_name("sort_list_of_numbers")
  .add_int64_axis("num_rows", {100'000, 1'000'000})
  // 4 and 32 keep eligible no-null types on the CUB fast path (average under its 100 cutoff); 256
  // (average ~128) stays on it only at 100k rows via the total-size arm and falls back at 1M.
  .add_int64_axis("max_list_size", {4, 32, 256})
  // No-null vs a realistic null rate; any nulls disqualify the CUB fast path, so 0.1 forces every
  // type onto the comparator fallback.
  .add_float64_axis("null_frequency", {0, 0.1})
  // Full (order, null_order) matrix: the CUB path honors order (no nulls by its gate); the
  // fallback honors both.
  .add_string_axis("order", {"ASC", "DESC"})
  .add_string_axis("null_order", {"AFTER", "BEFORE"});
