/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

// Overwrites the first `min(plen, byte_length)` bytes of every non-null leaf string of a
// LIST<STRING> column with the constant byte 'A' (0x41), returning a new LIST<STRING> column. The
// strings keep their original lengths and offsets, so they stay distinct in their tails while now
// sharing a `plen`-byte leading prefix -- the regime where the packed-prefix key is uninformative
// and the tie-break does the real work. This is data setup, not a measured operation, so a host
// round-trip is acceptable and keeps the helper host-compilable (the benchmark is a .cpp). Offset
// values may be int32 or int64; both are normalized to int64 on the host. The leaf is assumed
// unsliced (offset 0), which holds for the freshly generated benchmark column.
static std::unique_ptr<cudf::column> apply_shared_prefix(cudf::column_view const& list_col,
                                                         cudf::size_type plen,
                                                         rmm::cuda_stream_view stream)
{
  auto const mr = cudf::get_current_device_resource_ref();

  auto const lists    = cudf::lists_column_view{list_col};
  auto const scv      = cudf::strings_column_view{lists.child()};
  auto const num_leaf = scv.size();

  // Pull chars, offsets, and validity to the host so the mutation is a plain host loop.
  auto const chars_bytes = static_cast<std::size_t>(scv.chars_size(stream));
  auto host_chars        = cudf::detail::make_host_vector<char>(
    cudf::device_span<char const>{scv.chars_begin(stream), chars_bytes}, stream);

  // Normalize offsets (int32 or int64) to a single int64 host array of length num_leaf + 1.
  auto const offsets_cv = scv.offsets();
  std::vector<int64_t> offsets(static_cast<std::size_t>(num_leaf) + 1);
  if (offsets_cv.type().id() == cudf::type_id::INT64) {
    auto const host_offsets = cudf::detail::make_host_vector<int64_t>(
      cudf::device_span<int64_t const>{offsets_cv.data<int64_t>(),
                                       static_cast<std::size_t>(num_leaf) + 1},
      stream);
    std::copy(host_offsets.begin(), host_offsets.end(), offsets.begin());
  } else {
    auto const host_offsets = cudf::detail::make_host_vector<cudf::size_type>(
      cudf::device_span<cudf::size_type const>{offsets_cv.data<cudf::size_type>(),
                                               static_cast<std::size_t>(num_leaf) + 1},
      stream);
    std::copy(host_offsets.begin(), host_offsets.end(), offsets.begin());
  }

  // Copy the leaf null mask to the host (if any) to skip null elements.
  std::vector<cudf::bitmask_type> host_mask;
  if (scv.null_mask() != nullptr) {
    auto const host_mask_words = cudf::detail::make_host_vector<cudf::bitmask_type>(
      cudf::device_span<cudf::bitmask_type const>{
        scv.null_mask(), static_cast<std::size_t>(cudf::num_bitmask_words(num_leaf))},
      stream);
    host_mask.assign(host_mask_words.begin(), host_mask_words.end());
  }
  auto const* mask_ptr = host_mask.empty() ? nullptr : host_mask.data();

  for (cudf::size_type i = 0; i < num_leaf; ++i) {
    if (!cudf::bit_value_or(mask_ptr, i, true)) { continue; }  // Skip null elements.
    auto const len = offsets[i + 1] - offsets[i];
    auto const k   = std::min<int64_t>(plen, len);
    std::fill_n(host_chars.begin() + offsets[i], k, 'A');
  }

  // Rebuild the leaf STRING column from the mutated chars plus copies of the original offsets and
  // null mask, then rebuild the LIST column around it from copies of the list offsets and mask.
  auto new_chars = rmm::device_buffer{host_chars.data(), chars_bytes, stream, mr};
  auto new_leaf  = cudf::make_strings_column(num_leaf,
                                            std::make_unique<cudf::column>(offsets_cv, stream, mr),
                                            std::move(new_chars),
                                            scv.null_count(),
                                            cudf::copy_bitmask(scv.parent(), stream, mr));

  auto result = cudf::make_lists_column(lists.size(),
                                        std::make_unique<cudf::column>(lists.offsets(), stream, mr),
                                        std::move(new_leaf),
                                        list_col.null_count(),
                                        cudf::copy_bitmask(list_col, stream, mr));

  // The host->device chars copy above runs on `stream` and reads only `host_chars`; synchronize
  // before it goes out of scope so the copy cannot read freed host memory (`offsets`/`host_mask`
  // are fully consumed by the synchronous loop above and carry no such dependency).
  stream.synchronize();
  return result;
}

// The list generator forces its final offset to the child size ("always include all elements"), so
// the last row absorbs every leftover element -- an artifact row often far above `max_list_size`
// that would disqualify the whole column from the graduated-warp path (its gate requires every
// segment within the 64-element warp tile) and silently demote it to the prefix path. Trimming that
// one row back to `max_list_size` restores the axis's declared [0, max_list_size] regime; it runs
// outside the timed region. Returns nullptr when the generated last row already fits, so the caller
// keeps the original column. The generator purges nonempty nulls, so an oversized last row is never
// a null row and the trim cannot create a nonempty null.
static std::unique_ptr<cudf::column> trim_forced_last_row(cudf::column_view const& list_col,
                                                          cudf::size_type max_list_size,
                                                          rmm::cuda_stream_view stream)
{
  auto const mr       = cudf::get_current_device_resource_ref();
  auto const lists    = cudf::lists_column_view{list_col};
  auto const num_rows = lists.size();

  // Normalize the list offsets (int32 or int64) to one int64 host array of length num_rows + 1,
  // mirroring `apply_shared_prefix`, so a wider offset column is read at its true width instead of
  // reinterpreted.
  auto const offsets_cv = lists.offsets();
  std::vector<int64_t> offsets(static_cast<std::size_t>(num_rows) + 1);
  if (offsets_cv.type().id() == cudf::type_id::INT64) {
    auto const host_offsets = cudf::detail::make_host_vector<int64_t>(
      cudf::device_span<int64_t const>{offsets_cv.data<int64_t>(),
                                       static_cast<std::size_t>(num_rows) + 1},
      stream);
    std::copy(host_offsets.begin(), host_offsets.end(), offsets.begin());
  } else {
    auto const host_offsets = cudf::detail::make_host_vector<cudf::size_type>(
      cudf::device_span<cudf::size_type const>{offsets_cv.data<cudf::size_type>(),
                                               static_cast<std::size_t>(num_rows) + 1},
      stream);
    std::copy(host_offsets.begin(), host_offsets.end(), offsets.begin());
  }
  if (num_rows == 0 or offsets[num_rows] - offsets[num_rows - 1] <= max_list_size) {
    return nullptr;
  }

  auto new_offsets          = offsets;
  new_offsets[num_rows]     = new_offsets[num_rows - 1] + max_list_size;
  auto const new_child_size = static_cast<cudf::size_type>(new_offsets[num_rows]);

  // Rebuild the offsets column at the input's own width so the trimmed column keeps the
  // generator's offset type. Both host staging vectors outlive the trailing synchronize, so the
  // async H2D copy never reads freed host memory.
  std::vector<cudf::size_type> new_offsets_i32;
  std::unique_ptr<cudf::column> offsets_col;
  if (offsets_cv.type().id() == cudf::type_id::INT64) {
    offsets_col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT64},
      num_rows + 1,
      rmm::device_buffer{new_offsets.data(), new_offsets.size() * sizeof(int64_t), stream, mr},
      rmm::device_buffer{},
      0);
  } else {
    new_offsets_i32.assign(new_offsets.begin(), new_offsets.end());
    offsets_col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      num_rows + 1,
      rmm::device_buffer{
        new_offsets_i32.data(), new_offsets_i32.size() * sizeof(cudf::size_type), stream, mr},
      rmm::device_buffer{},
      0);
  }
  auto new_leaf =
    std::make_unique<cudf::column>(cudf::slice(lists.child(), {0, new_child_size})[0], stream, mr);
  auto result = cudf::make_lists_column(num_rows,
                                        std::move(offsets_col),
                                        std::move(new_leaf),
                                        list_col.null_count(),
                                        cudf::copy_bitmask(list_col, stream, mr));
  // The offsets H2D copy reads the host offset buffers on `stream`; synchronize before they go out
  // of scope.
  stream.synchronize();
  return result;
}

// Benchmark for cudf::lists::sort_lists on a LIST<STRING> column, the operation Spark
// array_sort(array<string>) lowers to and which a plain table sort does not exercise. The axes span
// the shape space a reviewer needs to see the whole picture of the string fast path: list-length
// regimes, string width, and -- the tie-break stressor -- how many leading bytes the elements
// share. The order and null_order axes complete the sort-parameter matrix.
static void bench_sort_list_of_strings(nvbench::state& state)
{
  auto const num_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const max_list_size  = static_cast<cudf::size_type>(state.get_int64("max_list_size"));
  auto const row_width      = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const null_frequency = state.get_float64("null_frequency");
  // shared_prefix_len > 0 overwrites the first that-many bytes of every leaf string with a constant
  // so the strings share a leading prefix (the realistic regime array_sort hits on keyed data); 0
  // leaves the random data unchanged (the control). Applied below, outside the timed region.
  auto const shared_prefix_len = static_cast<cudf::size_type>(state.get_int64("shared_prefix_len"));
  auto const column_order =
    state.get_string("order") == "DESC" ? cudf::order::DESCENDING : cudf::order::ASCENDING;
  auto const null_precedence =
    state.get_string("null_order") == "BEFORE" ? cudf::null_order::BEFORE : cudf::null_order::AFTER;

  // Skip when estimated leaf chars (rows x mean list x mean width) exceed the int32 char cap.
  auto const estimated_chars =
    static_cast<double>(num_rows) * (max_list_size / 2.0) * (row_width / 2.0);
  if (estimated_chars > static_cast<double>(std::numeric_limits<cudf::size_type>::max())) {
    state.skip("Skip benchmarks greater than size_type limit");
    return;
  }

  // Build a LIST<STRING> column: list length is uniform in [0, max_list_size] and each string's
  // width is normally distributed in [0, row_width]. Leaf strings are all-distinct (cardinality 0)
  // to match the near-unique elements of the real array_sort workload -- the hardest case for the
  // tie-break, and the one the shared_prefix_len axis stresses further; leaf cardinality is
  // therefore not a separate axis. Nulls are applied at both the list-row and string-element levels
  // so the sort exercises null ordering at the leaf.
  data_profile const profile =
    data_profile_builder()
      .list_type(cudf::type_id::STRING)
      .list_depth(1)
      .distribution(cudf::type_id::LIST, distribution_id::UNIFORM, 0, max_list_size)
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, 0, row_width)
      .null_probability(null_frequency)
      .cardinality(0);

  auto const table = create_random_table({cudf::type_id::LIST}, row_count{num_rows}, profile);

  auto const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  // Trim the generator's forced-last-row artifact (untimed data setup) so the data matches the
  // declared [0, max_list_size] regime; otherwise that one oversized row disqualifies the whole
  // column from the graduated-warp path and silently demotes it to the prefix path. See
  // `trim_forced_last_row`.
  auto const trimmed  = trim_forced_last_row(table->view().column(0), max_list_size, stream);
  auto const base_col = trimmed ? trimmed->view() : table->view().column(0);

  // Optionally force a shared leading prefix on the leaf strings (untimed data setup). When
  // shared_prefix_len == 0 the input is the (trimmed) random column; otherwise it is a new
  // LIST<STRING> column whose leaf strings all begin with `shared_prefix_len` copies of 'A'.
  auto const prefixed =
    shared_prefix_len > 0 ? apply_shared_prefix(base_col, shared_prefix_len, stream) : nullptr;
  auto const input = cudf::lists_column_view{prefixed ? prefixed->view() : base_col};
  stream.synchronize();  // Ensure the host round-trip and rebuild complete before timing.

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    cudf::lists::sort_lists(
      input, column_order, null_precedence, stream, cudf::get_current_device_resource_ref());
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_sort_list_of_strings)
  .set_name("sort_list_of_strings")
  .add_int64_axis("num_rows", {1'000'000})
  // Tiny / typical / large list lengths: tiny stresses per-segment overhead, large amortizes it.
  .add_int64_axis("max_list_size", {4, 32, 256})
  // Short strings mostly fit the packed key; wide strings push work into the byte-window tie-break.
  .add_int64_axis("row_width", {16, 64})
  // 0 control; 8 shares exactly the packed-key width (first tie-break window); 32 spans several.
  .add_int64_axis("shared_prefix_len", {0, 8, 32})
  // No-null vs a realistic null rate to exercise the leaf null ordering.
  .add_float64_axis("null_frequency", {0, 0.1})
  // Full (order, null_order) matrix: the fallback sort takes the requested polarity as sort
  // parameters, so every combination exercises the column.
  .add_string_axis("order", {"ASC", "DESC"})
  .add_string_axis("null_order", {"AFTER", "BEFORE"});
