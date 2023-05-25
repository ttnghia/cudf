/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

std::unique_ptr<table> copy_if(table_view const& input,
                               device_span<bool const> boolean_mask,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(static_cast<std::size_t>(input.num_rows()) == boolean_mask.size(),
               "Input size mismatch");
  if (input.is_empty()) { return empty_like(input); }

  return detail::copy_if(
    input,
    [boolean_mask = boolean_mask.begin()] __device__(auto const idx) { return boolean_mask[idx]; },
    stream,
    mr);
}

}  // namespace detail

std::unique_ptr<table> copy_if(table_view const& input,
                               device_span<bool const> boolean_mask,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_if(input, boolean_mask, cudf::get_default_stream(), mr);
}
}  // namespace cudf
