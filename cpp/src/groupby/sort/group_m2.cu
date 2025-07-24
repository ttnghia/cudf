/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace {

template <typename ResultType, typename Iterator>
struct m2_transform {
  column_device_view const d_values;
  Iterator const values_iter;
  ResultType const* d_means;
  size_type const* d_group_labels;

  __device__ ResultType operator()(size_type const idx) const noexcept
  {
    if (d_values.is_null(idx)) { return 0.0; }

    auto const x         = static_cast<ResultType>(values_iter[idx]);
    auto const group_idx = d_group_labels[idx];
    auto const mean      = d_means[group_idx];
    auto const diff      = x - mean;
    return diff * diff;
  }
};

template <typename ResultType, typename Iterator>
void compute_m2_fn(column_device_view const& values,
                   Iterator values_iter,
                   cudf::device_span<size_type const> group_labels,
                   ResultType const* d_means,
                   ResultType* d_result,
                   rmm::cuda_stream_view stream)
{
  auto m2_fn = m2_transform<ResultType, decltype(values_iter)>{
    values, values_iter, d_means, group_labels.data()};
  auto const itr = thrust::counting_iterator<size_type>(0);
  // Using a temporary buffer for intermediate transform results instead of
  // using the transform-iterator directly in thrust::reduce_by_key
  // improves compile-time significantly.
  auto m2_vals = rmm::device_uvector<ResultType>(values.size(), stream);
  thrust::transform(rmm::exec_policy(stream), itr, itr + values.size(), m2_vals.begin(), m2_fn);

  thrust::reduce_by_key(rmm::exec_policy(stream),
                        group_labels.begin(),
                        group_labels.end(),
                        m2_vals.begin(),
                        thrust::make_discard_iterator(),
                        d_result);
}

struct m2_functor {
  template <typename T>
  std::unique_ptr<column> operator()(column_view const& values,
                                     column_view const& group_means,
                                     cudf::device_span<size_type const> group_labels,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
    requires(std::is_arithmetic_v<T>)
  {
    using result_type = cudf::detail::target_type_t<T, aggregation::Kind::M2>;
    auto result       = make_numeric_column(data_type(type_to_id<result_type>()),
                                      group_means.size(),
                                      mask_state::UNALLOCATED,
                                      stream,
                                      mr);

    auto const values_dv_ptr = column_device_view::create(values, stream);
    auto const d_values      = *values_dv_ptr;
    auto const d_means       = group_means.data<result_type>();
    auto const d_result      = result->mutable_view().data<result_type>();

    if (!cudf::is_dictionary(values.type())) {
      auto const values_iter = d_values.begin<T>();
      compute_m2_fn(d_values, values_iter, group_labels, d_means, d_result, stream);
    } else {
      auto const values_iter =
        cudf::dictionary::detail::make_dictionary_iterator<T>(*values_dv_ptr);
      compute_m2_fn(d_values, values_iter, group_labels, d_means, d_result, stream);
    }

    // M2 column values should have the same bitmask as means's.
    if (group_means.nullable()) {
      result->set_null_mask(cudf::detail::copy_bitmask(group_means, stream, mr),
                            group_means.null_count());
    }

    return result;
  }

  template <typename T, typename... Args>
  std::unique_ptr<column> operator()(Args&&...) requires(!std::is_arithmetic_v<T>)
  {
    CUDF_FAIL("Only numeric types are supported in M2 groupby aggregation");
  }
};

template <typename InputType, typename ResultType>
struct m2_transform_new {
  InputType const* values_iter;
  ResultType const* d_means;
  size_type const* d_group_labels;

  __device__ ResultType operator()(size_type const idx) const noexcept
  {
    auto const x         = static_cast<ResultType>(values_iter[idx]);
    auto const group_idx = d_group_labels[idx];
    auto const mean      = d_means[group_idx];
    auto const diff      = x - mean;
    return diff * diff;
  }
};

template <typename InputType, typename ResultType>
void compute_m2_fn_new(rmm::device_uvector<InputType> const& values,
                       rmm::device_uvector<size_type> const& group_labels,
                       ResultType const* d_means,
                       ResultType* d_result,
                       rmm::cuda_stream_view stream)
{
  auto m2_fn =
    m2_transform_new<InputType, ResultType>{values.begin(), d_means, group_labels.data()};
  auto const itr = thrust::counting_iterator<size_type>(0);
  // Using a temporary buffer for intermediate transform results instead of
  // using the transform-iterator directly in thrust::reduce_by_key
  // improves compile-time significantly.
  auto m2_vals = rmm::device_uvector<ResultType>(values.size(), stream);
  thrust::transform(rmm::exec_policy(stream), itr, itr + values.size(), m2_vals.begin(), m2_fn);

  thrust::reduce_by_key(rmm::exec_policy_nosync(stream),
                        group_labels.begin(),
                        group_labels.end(),
                        m2_vals.begin(),
                        thrust::make_discard_iterator(),
                        d_result);
  stream.synchronize();
}

struct m2_functor_new {
  template <typename T>
  std::unique_ptr<column> operator()(column_view const& values,
                                     cudf::device_span<size_type const> key_indices,
                                     cudf::device_span<size_type const> key_arranged_map,
                                     cudf::size_type num_groups,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
    requires(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>)
  {
    using result_type = cudf::detail::target_type_t<T, aggregation::Kind::M2>;
    auto result       = make_numeric_column(
      data_type(type_to_id<result_type>()), num_groups, mask_state::UNALLOCATED, stream, mr);

    if (num_groups == 0) { return result; }

    rmm::device_uvector<size_type> group_labels(values.size(), stream);
    rmm::device_uvector<T> grouped_values(values.size(), stream);

    {
      cudf::scoped_range range{"gather value"};

      thrust::for_each(rmm::exec_policy_nosync(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       thrust::make_counting_iterator<size_type>(values.size()),
                       [values           = values.begin<T>(),
                        labels           = group_labels.begin(),
                        grouped_values   = grouped_values.begin(),
                        key_indices      = key_indices.begin(),
                        key_arranged_map = key_arranged_map.begin()] __device__(size_type idx) {
                         auto const grouped_idx = key_arranged_map[idx];
                         labels[idx]            = key_indices[grouped_idx];
                         grouped_values[idx]    = static_cast<T>(values[grouped_idx]);
                       });
      stream.synchronize();
    }

    //    auto h_l = cudf::detail::make_std_vector(group_labels, stream);
    //    printf("group_labels: \n");
    //    for (auto i : h_l) {
    //      printf("%d, ", i);
    //    }
    //    printf("\n\n\n");
    //    auto h_v = cudf::detail::make_std_vector(grouped_values, stream);
    //    printf("grouped values: \n");
    //    for (auto i : h_v) {
    //      printf("%f, ", (double)i);
    //    }
    //    printf("\n\n\n");

    rmm::device_uvector<double> mean(num_groups, stream);
    {
      cudf::scoped_range range{"comp mean"};

      rmm::device_uvector<int64_t> count(num_groups, stream);
      rmm::device_uvector<double> sum(num_groups, stream);

      {
        cudf::scoped_range range{"count"};
        thrust::reduce_by_key(rmm::exec_policy_nosync(stream),
                              group_labels.begin(),
                              group_labels.end(),
                              thrust::make_constant_iterator(1),
                              thrust::make_discard_iterator(),
                              count.begin());
        stream.synchronize();
      }
      {
        cudf::scoped_range range{"sum"};
        thrust::reduce_by_key(rmm::exec_policy_nosync(stream),
                              group_labels.begin(),
                              group_labels.end(),
                              grouped_values.begin(),
                              thrust::make_discard_iterator(),
                              sum.begin(),
                              cuda::std::equal_to{},
                              cuda::std::plus<double>{});
        stream.synchronize();
      }
      {
        cudf::scoped_range range{"mean"};
        thrust::transform(rmm::exec_policy_nosync(stream),
                          sum.begin(),
                          sum.end(),
                          count.begin(),
                          mean.begin(),
                          [] __device__(double s, int64_t c) { return c == 0 ? 0.0 : s / c; });
        stream.synchronize();
      }
    }

    auto const d_result = result->mutable_view().data<result_type>();

    {
      cudf::scoped_range range{"comp m2"};
      compute_m2_fn_new(grouped_values, group_labels, mean.begin(), d_result, stream);
    }
    return result;
  }

  template <typename T, typename... Args>
  std::unique_ptr<column> operator()(Args&&...)
    requires(!std::is_arithmetic_v<T> || std::is_same_v<T, bool>)
  {
    CUDF_FAIL("Only numeric types are supported in M2 groupby aggregation");
  }
};

}  // namespace

std::unique_ptr<column> group_m2(column_view const& values,
                                 column_view const& group_means,
                                 cudf::device_span<size_type const> group_labels,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto values_type = cudf::is_dictionary(values.type())
                       ? dictionary_column_view(values).keys().type()
                       : values.type();

  return type_dispatcher(values_type, m2_functor{}, values, group_means, group_labels, stream, mr);
}

std::unique_ptr<column> group_m2_new(column_view const& values,
                                     cudf::device_span<size_type const> key_indices,
                                     cudf::device_span<size_type const> key_arranged_map,
                                     cudf::size_type num_groups,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto values_type = cudf::is_dictionary(values.type())
                       ? dictionary_column_view(values).keys().type()
                       : values.type();

  return type_dispatcher(
    values_type, m2_functor_new{}, values, key_indices, key_arranged_map, num_groups, stream, mr);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
