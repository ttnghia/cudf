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

#include "compression_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/io_metadata_utilities.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/io/orc_metadata.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>

namespace {
enum class output_limit : std::size_t {};
enum class input_limit : std::size_t {};
enum class output_row_granularity : cudf::size_type {};

// Global environment for temporary files
auto const temp_env = reinterpret_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

using int32s_col       = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64s_col       = cudf::test::fixed_width_column_wrapper<int64_t>;
using doubles_col      = cudf::test::fixed_width_column_wrapper<double>;
using strings_col      = cudf::test::strings_column_wrapper;
using structs_col      = cudf::test::structs_column_wrapper;
using int32s_lists_col = cudf::test::lists_column_wrapper<int32_t>;

auto write_file(std::vector<std::unique_ptr<cudf::column>>& input_columns,
                std::string const& filename,
                bool nullable                    = false,
                std::size_t stripe_size_bytes    = cudf::io::default_stripe_size_bytes,
                cudf::size_type stripe_size_rows = cudf::io::default_stripe_size_rows)
{
  if (nullable) {
    // Generate deterministic bitmask instead of random bitmask for easy computation of data size.
    auto const valid_iter = cudf::detail::make_counting_transform_iterator(
      0, [](cudf::size_type i) { return i % 4 != 3; });
    cudf::size_type offset{0};
    for (auto& col : input_columns) {
      auto const [null_mask, null_count] =
        cudf::test::detail::make_null_mask(valid_iter + offset, valid_iter + col->size() + offset);
      col = cudf::structs::detail::superimpose_and_sanitize_nulls(
        static_cast<cudf::bitmask_type const*>(null_mask.data()),
        null_count,
        std::move(col),
        cudf::get_default_stream(),
        cudf::get_current_device_resource_ref());

      // Shift nulls of the next column by one position, to avoid having all nulls
      // in the same table rows.
      ++offset;
    }
  }

  auto input_table = std::make_unique<cudf::table>(std::move(input_columns));
  auto filepath =
    temp_env->get_temp_filepath(nullable ? filename + "_nullable.orc" : filename + ".orc");

  auto const write_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, *input_table)
      .stripe_size_bytes(stripe_size_bytes)
      .stripe_size_rows(stripe_size_rows)
      .build();
  cudf::io::write_orc(write_opts);

  return std::pair{std::move(input_table), std::move(filepath)};
}

// NOTE: By default, output_row_granularity=10'000 rows.
// This means if the input file has more than 10k rows then the output chunk will never
// have less than 10k rows.
auto chunked_read(std::string const& filepath,
                  output_limit output_limit_bytes           = output_limit{0},
                  input_limit input_limit_bytes             = input_limit{0},
                  output_row_granularity output_granularity = output_row_granularity{10'000})
{
  auto const read_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath}).build();
  auto reader = cudf::io::chunked_orc_reader(static_cast<std::size_t>(output_limit_bytes),
                                             static_cast<std::size_t>(input_limit_bytes),
                                             static_cast<cudf::size_type>(output_granularity),
                                             read_opts);

  auto num_chunks = 0;
  auto out_tables = std::vector<std::unique_ptr<cudf::table>>{};

  // TODO: remove this scope, when we get rid of mem stat in the reader.
  // This is to avoid use-after-free of memory resource created by the mem stat object.
  auto mr = cudf::get_current_device_resource_ref();

  static bool printed = false;
  do {
    auto chunk = reader.read_chunk();
    // If the input file is empty, the first call to `read_chunk` will return an empty table.
    // Thus, we only check for non-empty output table from the second call.
    if (num_chunks > 0) {
      CUDF_EXPECTS(chunk.tbl->num_rows() != 0, "Number of rows in the new chunk is zero.");
    }
    ++num_chunks;

    if (!printed) {
      printed    = true;
      auto& meta = chunk.metadata;
      int count  = 0;
      for (auto& col : meta.schema_info) {
        std::cout << count++ << ", column name: " << col.name << std::endl;
      }
    }
    out_tables.emplace_back(std::move(chunk.tbl));
  } while (reader.has_next());

  if (num_chunks > 1) {
    CUDF_EXPECTS(out_tables.front()->num_rows() != 0, "Number of rows in the new chunk is zero.");
  }

  if (num_chunks == 1) { return std::pair(std::move(out_tables.front()), num_chunks); }

  auto out_tviews = std::vector<cudf::table_view>{};
  for (auto const& tbl : out_tables) {
    out_tviews.emplace_back(tbl->view());
  }

  // return std::pair(cudf::concatenate(out_tviews), num_chunks);

  // TODO: remove this
  return std::pair(cudf::concatenate(out_tviews, cudf::get_default_stream(), mr), num_chunks);
}

}  // namespace

#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

std::vector<std::string> find_orc_files(const std::string& root)
{
  std::vector<std::string> result;
  fs::path root_path(root);  // convert string to filesystem::path
  for (const auto& entry : fs::recursive_directory_iterator(root_path)) {
    if (entry.is_regular_file() && entry.path().extension() == ".orc") {
      result.push_back(fs::absolute(entry.path()).string());
    }
  }
  return result;
}

struct OrcChunkedReaderTest : public cudf::test::BaseFixture {};

using OrcChunkedDecompressionTest = DecompressionTest<OrcChunkedReaderTest>;

#if 0
TEST_F(OrcChunkedReaderTest, ListFiles)
{
  auto const path  = "/home/nghiat/tmp/store_sales/";
  auto const files = find_orc_files(path);
  for (auto& f : files) {
    std::cout << f << std::endl;
  }
}
#endif

#include <unordered_map>

TEST_F(OrcChunkedReaderTest, TestFiles)
{
  auto const path  = "/home/nghiat/tmp/store_sales/";
  auto const files = find_orc_files(path);

  std::unique_ptr<cudf::column> expected{nullptr};
  std::unordered_map<std::string, int> file_rows;
  std::unordered_map<std::string, int> file_distinct_counts;
  std::unordered_map<std::string, int> file_distinct_counts_nulls;

  for (int iter = 0; iter < 100; ++iter) {
    printf("Iter: %d\n", iter);
    fflush(stdout);
    for (auto& f : files) {
      // std::cout << f << std::endl;
      auto const [result, num_chunks] = chunked_read(f);

      auto agg_null =
        cudf::make_nunique_aggregation<cudf::reduce_aggregation>(cudf::null_policy::INCLUDE);
      auto agg =
        cudf::make_nunique_aggregation<cudf::reduce_aggregation>(cudf::null_policy::EXCLUDE);

      auto size_data_type = cudf::data_type(cudf::type_to_id<cudf::size_type>());
      auto const dcount   = cudf::reduce(result->get_column(11).view(), *agg, size_data_type);
      auto const hcount =
        dynamic_cast<cudf::numeric_scalar<cudf::size_type>*>(dcount.get())->value();

      auto const dcount_nulls =
        cudf::reduce(result->get_column(11).view(), *agg_null, size_data_type);
      auto const hcount_nulls =
        dynamic_cast<cudf::numeric_scalar<cudf::size_type>*>(dcount_nulls.get())->value();

      if (file_rows.contains(f)) {
        EXPECT_EQ(result->num_rows(), file_rows[f]);
        EXPECT_TRUE(file_distinct_counts.contains(f));
        EXPECT_TRUE(file_distinct_counts_nulls.contains(f));
        CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(),
                                       result->get_column(11).view(),
                                       cudf::test::debug_output_level::ALL_ERRORS);
        fflush(stdout);

        if (file_distinct_counts[f] != hcount) {
          printf(
            "... difference w/o nulls: %d vs %d, %s\n", file_distinct_counts[f], hcount, f.c_str());

          fflush(stdout);
          exit(0);
        }
        if (file_distinct_counts_nulls[f] != hcount_nulls) {
          printf("... difference with nulls: %d vs %d, %s\n",
                 file_distinct_counts_nulls[f],
                 hcount_nulls,
                 f.c_str());
          fflush(stdout);
          exit(0);
        }

        EXPECT_EQ(file_distinct_counts[f], hcount);
        EXPECT_EQ(file_distinct_counts_nulls[f], hcount_nulls);
      } else {
        file_rows[f]                  = result->num_rows();
        file_distinct_counts[f]       = hcount;
        file_distinct_counts_nulls[f] = hcount_nulls;

        if (!expected) { expected = std::move((result->release())[11]); }
      }
      // std::cout << "    Number of rows: " << result->num_rows() << ", num. chunks: " <<
      // num_chunks
      //           << std::endl;
      // std::cout << std::endl << std::endl << std::endl;
    }
  }
}

TEST_F(OrcChunkedReaderTest, TestChunkedReadNoData)
{
  std::vector<std::unique_ptr<cudf::column>> input_columns;
  input_columns.emplace_back(int32s_col{}.release());
  input_columns.emplace_back(int64s_col{}.release());

  auto const [expected, filepath] = write_file(input_columns, "chunked_read_empty");
  auto const [result, num_chunks] = chunked_read(filepath, output_limit{1'000});
  EXPECT_EQ(num_chunks, 1);
  EXPECT_EQ(result->num_rows(), 0);
  EXPECT_EQ(result->num_columns(), 2);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
}
