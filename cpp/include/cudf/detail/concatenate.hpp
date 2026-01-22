/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <vector>

namespace CUDF_EXPORT cudf {
//! Inner interfaces and implementations
namespace detail {
/**
 * @copydoc cudf::concatenate(host_span<column_view const>,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> concatenate(host_span<column_view const> columns_to_concat,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::concatenate(host_span<table_view const>,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> concatenate(host_span<table_view const> tables_to_concat,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr);

/**
 * @brief Batch concatenates columns using optimized batched memory operations.
 *
 * This function performs concatenation of columns using batched memory copy operations
 * (cub::DeviceMemcpy::Batched) and batched mask concatenation kernels. It produces
 * identical results to cudf::concatenate but is optimized for reducing kernel launch
 * overhead by processing all nesting levels in fewer operations.
 *
 * Supported types:
 * - Fixed-width types (int, float, etc.)
 * - Struct types (recursively supports all nested types)
 * - String types
 * - List types (recursively supports nested lists and all child types)
 * - Dictionary types (delegated to specialized implementation)
 *
 * @param columns_to_concat Columns to concatenate
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used for allocating the returned column's device memory.
 * @return A single column containing all rows from the input columns.
 *
 * @throws cudf::logic_error if column types don't match
 * @throws std::overflow_error if total row count exceeds size_type limits
 */
std::unique_ptr<column> batch_concatenate(host_span<column_view const> columns_to_concat,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr);

/**
 * @brief Checks if batch_concatenate can be used for the given columns.
 *
 * Returns true if all columns are of supported types: fixed-width, struct, string,
 * list, or dictionary types.
 *
 * @param columns Columns to check
 * @return true if batch_concatenate can be used, false otherwise
 */
bool can_use_batch_concatenate(host_span<column_view const> columns);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
