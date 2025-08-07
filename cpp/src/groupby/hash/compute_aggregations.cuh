/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "compute_aggregations.hpp"
#include "compute_global_memory_aggs.hpp"
#include "compute_mapping_indices.hpp"
#include "compute_shared_memory_aggs.hpp"
#include "create_sparse_results_table.hpp"
#include "flatten_single_pass_aggs.hpp"
#include "helpers.cuh"
#include "single_pass_functors.cuh"

#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuco/static_set.cuh>
#include <cuda/std/atomic>
#include <thrust/for_each.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace cudf::groupby::detail::hash {
template <typename SetType>
CUDF_KERNEL void gm_fallback_kernel(int64_t total_size,
                                    size_type num_rows,
                                    SetType set,
                                    table_device_view input_values,
                                    mutable_table_device_view output_values,
                                    aggregation::Kind const* __restrict__ aggs,
                                    cudf::size_type* block_cardinality,
                                    cudf::size_type stride,
                                    bitmask_type const* __restrict__ row_bitmask,
                                    bool skip_rows_with_nulls)
{
  auto i = cudf::detail::grid_1d::global_thread_id();
  if (i >= total_size) return;

  auto const row_idx  = static_cast<size_type>(i % num_rows);
  auto const block_id = (row_idx % stride) / GROUPBY_BLOCK_SIZE;
  if (block_cardinality[block_id] >= GROUPBY_CARDINALITY_THRESHOLD and
      (not skip_rows_with_nulls or cudf::bit_is_set(row_bitmask, row_idx))) {
    auto const result  = set.insert_and_find(row_idx);
    auto const col_idx = static_cast<size_type>(i / num_rows);
    cudf::detail::aggregate_row(col_idx, output_values, *result.first, input_values, row_idx, aggs);
  }
}

/**
 * @brief Computes all aggregations from `requests` that require a single pass
 * over the data and stores the results in `sparse_results`
 */
template <typename SetType>
rmm::device_uvector<cudf::size_type> compute_aggregations(
  int64_t num_rows,
  bool skip_rows_with_nulls,
  bitmask_type const* row_bitmask,
  SetType& global_set,
  cudf::host_span<cudf::groupby::aggregation_request const> requests,
  cudf::detail::result_cache* sparse_results,
  rmm::cuda_stream_view stream)
{
  // flatten the aggs to a table that can be operated on by aggregate_row
  auto [flattened_values, agg_kinds, aggs] = flatten_single_pass_aggs(requests, stream);
  auto const d_agg_kinds                   = cudf::detail::make_device_uvector_async(
    agg_kinds, stream, rmm::mr::get_current_device_resource());

  auto const grid_size =
    max_occupancy_grid_size<typename SetType::ref_type<cuco::insert_and_find_tag>>(num_rows);
  auto const available_shmem_size = get_available_shared_memory_size(grid_size);
  auto const offsets_buffer_size  = compute_shmem_offsets_size(flattened_values.num_columns()) * 2;
  auto const data_buffer_size     = available_shmem_size - offsets_buffer_size;

  // Check if any aggregation is SUM_WITH_OVERFLOW, which should always use global memory
  auto const has_sum_with_overflow =
    std::any_of(agg_kinds.begin(), agg_kinds.end(), [](aggregation::Kind k) {
      return k == aggregation::SUM_WITH_OVERFLOW;
    });

  auto const is_shared_memory_compatible =
    !has_sum_with_overflow &&
    std::all_of(
      requests.begin(), requests.end(), [&](cudf::groupby::aggregation_request const& request) {
        if (cudf::is_dictionary(request.values.type())) { return false; }
        // Ensure there is enough buffer space to store local aggregations up to the max cardinality
        // for shared memory aggregations
        auto const size = cudf::type_dispatcher<cudf::dispatch_storage_type>(request.values.type(),
                                                                             size_of_functor{});
        return data_buffer_size >= (size * GROUPBY_CARDINALITY_THRESHOLD);
      });

  // Performs naive global memory aggregations when the workload is not compatible with shared
  // memory, such as when aggregating dictionary columns, when there is insufficient dynamic
  // shared memory for shared memory aggregations, or when SUM_WITH_OVERFLOW aggregations are
  // present.
  if (!is_shared_memory_compatible) {
    return compute_global_memory_aggs(num_rows,
                                      skip_rows_with_nulls,
                                      row_bitmask,
                                      flattened_values,
                                      d_agg_kinds.data(),
                                      agg_kinds,
                                      global_set,
                                      aggs,
                                      sparse_results,
                                      stream);
  }

  // 'populated_keys' contains inserted row_indices (keys) of global hash set
  rmm::device_uvector<cudf::size_type> populated_keys(num_rows, stream);
  // 'local_mapping_index' maps from the global row index of the input table to its block-wise rank
  rmm::device_uvector<cudf::size_type> local_mapping_index(num_rows, stream);
  // 'global_mapping_index' maps from the block-wise rank to the row index of global aggregate table
  rmm::device_uvector<cudf::size_type> global_mapping_index(grid_size * GROUPBY_SHM_MAX_ELEMENTS,
                                                            stream);
  rmm::device_uvector<cudf::size_type> block_cardinality(grid_size, stream);

  // Flag indicating whether a global memory aggregation fallback is required or not
  rmm::device_scalar<cuda::std::atomic_flag> needs_global_memory_fallback(stream);

  auto global_set_ref = global_set.ref(cuco::op::insert_and_find);

  compute_mapping_indices(grid_size,
                          num_rows,
                          global_set_ref,
                          row_bitmask,
                          skip_rows_with_nulls,
                          local_mapping_index.data(),
                          global_mapping_index.data(),
                          block_cardinality.data(),
                          needs_global_memory_fallback.data(),
                          stream);

  cuda::std::atomic_flag h_needs_fallback;
  // Cannot use `device_scalar::value` as it requires a copy constructor, which
  // `atomic_flag` doesn't have.
  CUDF_CUDA_TRY(cudaMemcpyAsync(&h_needs_fallback,
                                needs_global_memory_fallback.data(),
                                sizeof(cuda::std::atomic_flag),
                                cudaMemcpyDefault,
                                stream.value()));
  stream.synchronize();
  auto const needs_fallback = h_needs_fallback.test();

  // make table that will hold sparse results
  cudf::table sparse_table = create_sparse_results_table(flattened_values,
                                                         d_agg_kinds.data(),
                                                         agg_kinds,
                                                         needs_fallback,
                                                         global_set,
                                                         populated_keys,
                                                         stream);
  // prepare to launch kernel to do the actual aggregation
  auto d_values       = table_device_view::create(flattened_values, stream);
  auto d_sparse_table = mutable_table_device_view::create(sparse_table, stream);

  compute_shared_memory_aggs(grid_size,
                             available_shmem_size,
                             num_rows,
                             row_bitmask,
                             skip_rows_with_nulls,
                             local_mapping_index.data(),
                             global_mapping_index.data(),
                             block_cardinality.data(),
                             *d_values,
                             *d_sparse_table,
                             d_agg_kinds.data(),
                             stream);

  // The shared memory groupby is designed so that each thread block can handle up to 128 unique
  // keys. When a block reaches this cardinality limit, shared memory becomes insufficient to store
  // the temporary aggregation results. In these situations, we must fall back to a global memory
  // aggregator to process the remaining aggregation requests.
  if (needs_fallback) {
    auto const stride     = GROUPBY_BLOCK_SIZE * grid_size;
    auto const total_size = num_rows * static_cast<int64_t>(flattened_values.num_columns());
    cudf::detail::grid_1d grid{total_size, GROUPBY_BLOCK_SIZE};
    gm_fallback_kernel<<<grid.num_blocks, GROUPBY_BLOCK_SIZE, 0, stream.value()>>>(
      total_size,
      num_rows,
      global_set_ref,
      *d_values,
      *d_sparse_table,
      d_agg_kinds.data(),
      block_cardinality.data(),
      stride,
      row_bitmask,
      skip_rows_with_nulls);

    extract_populated_keys(global_set, populated_keys, stream);
  }

  // Add results back to sparse_results cache
  auto sparse_result_cols = sparse_table.release();
  for (size_t i = 0; i < aggs.size(); i++) {
    // Note that the cache will make a copy of this temporary aggregation
    sparse_results->add_result(
      flattened_values.column(i), *aggs[i], std::move(sparse_result_cols[i]));
  }

  return populated_keys;
}
}  // namespace cudf::groupby::detail::hash
