/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <superdex_recorder/episode_reader.h>

#include "rrd_chunks.h"

#include <superdex_recorder/formatted_error.h>

#include <algorithm>
#include <map>
#include <tuple>
#include <utility>

namespace superdex::recorder {
namespace {

// This reader reads any 1.x version of the episode schema.
constexpr std::string_view kReadableSchemaName = "superdex.episode";
constexpr std::string_view kReadableSchemaVersionMajor = "1";

/* One chunk's rows of one component, with the chunk's timelines, which static chunks lack. */
struct ChunkColumn {
  std::shared_ptr<arrow::ListArray> values;
  std::shared_ptr<arrow::Int64Array> frames;
  std::shared_ptr<arrow::Int64Array> simulationTimes;
};

struct ComponentOrder {
  bool operator()(EpisodeComponent const& a, EpisodeComponent const& b) const {
    return std::tie(a.entityPath, a.component, a.isStatic) <
        std::tie(b.entityPath, b.component, b.isStatic);
  }
};

std::shared_ptr<arrow::Int64Array> FindTimeline(
    detail::RrdChunk const& chunk,
    std::string const& name) {
  auto const timeline = chunk.timelines.find(name);
  return timeline == chunk.timelines.end() ? nullptr : timeline->second;
}

template <typename ArrowType>
bool TryAppendAsDouble(
    arrow::Array const& values,
    int64_t begin,
    int64_t end,
    std::vector<double>& output) {
  if (values.type_id() != ArrowType::type_id) {
    return false;
  }
  auto const* const data = static_cast<arrow::NumericArray<ArrowType> const&>(values).raw_values();
  output.insert(output.end(), data + begin, data + end);
  return true;
}

/* Appends one row's values, flattening fixed-size lists such as XYZ triples. Returns false if the
 * column does not hold floats or doubles. */
bool AppendNumbers(arrow::ListArray const& column, int64_t row, std::vector<double>& output) {
  int64_t begin = column.value_offset(row);
  int64_t end = column.value_offset(row + 1);
  arrow::Array const* values = column.values().get();
  if (values->type_id() == arrow::Type::FIXED_SIZE_LIST) {
    auto const& lists = static_cast<arrow::FixedSizeListArray const&>(*values);
    if (lists.null_count() != 0) {
      return false;
    }
    begin = lists.value_offset(begin);
    end = lists.value_offset(end);
    values = lists.values().get();
  }
  return values->null_count() == 0 &&
      (TryAppendAsDouble<arrow::DoubleType>(*values, begin, end, output) ||
       TryAppendAsDouble<arrow::FloatType>(*values, begin, end, output));
}

/* Appends one row's strings. Returns false if the column does not hold text. */
bool AppendTexts(arrow::ListArray const& column, int64_t row, std::vector<std::string>& output) {
  auto const& values = *column.values();
  if (values.type_id() != arrow::Type::STRING || values.null_count() != 0) {
    return false;
  }
  auto const& strings = static_cast<arrow::StringArray const&>(values);
  for (int64_t i = column.value_offset(row); i < column.value_offset(row + 1); ++i) {
    output.emplace_back(strings.GetView(i));
  }
  return true;
}

void SetMissingComponentError(
    std::string_view entityPath,
    std::string_view component,
    bool isStatic,
    mochi::Error& error) {
  SUPERDEX_ERROR_SET(
      error,
      "Episode has no %s rows of '%.*s' on '%.*s'.",
      isStatic ? "static" : "temporal",
      static_cast<int>(component.size()),
      component.data(),
      static_cast<int>(entityPath.size()),
      entityPath.data());
}

void SetWrongTypeError(
    std::string_view entityPath,
    std::string_view component,
    char const* expected,
    mochi::Error& error) {
  SUPERDEX_ERROR_SET(
      error,
      "Component '%.*s' on '%.*s' does not hold %s.",
      static_cast<int>(component.size()),
      component.data(),
      static_cast<int>(entityPath.size()),
      entityPath.data(),
      expected);
}

void ValidateSchema(
    EpisodeReader const& reader,
    std::filesystem::path const& path,
    mochi::Error& error) {
  auto const name = reader.ReadStaticTexts("episode", "superdex.episode.schema_name", error);
  auto const version = reader.ReadStaticTexts("episode", "superdex.episode.schema_version", error);
  MOCHI_ERROR_RETURN(error);
  std::string const nameText = name.size() == 1 ? name[0] : "";
  std::string const versionText = version.size() == 1 ? version[0] : "";
  if (nameText != kReadableSchemaName ||
      versionText.substr(0, versionText.find('.')) != kReadableSchemaVersionMajor) {
    SUPERDEX_ERROR_SET(
        error,
        "'%s' has episode schema '%s' version '%s', but this reader reads '%.*s' version %.*s.",
        path.string().c_str(),
        nameText.c_str(),
        versionText.c_str(),
        static_cast<int>(kReadableSchemaName.size()),
        kReadableSchemaName.data(),
        static_cast<int>(kReadableSchemaVersionMajor.size()),
        kReadableSchemaVersionMajor.data());
  }
}

} // namespace

struct EpisodeReader::Impl {
  // The chunks that hold each stored component, in file order.
  std::map<EpisodeComponent, std::vector<ChunkColumn>, ComponentOrder> chunks;
  std::vector<EpisodeComponent> components;

  std::vector<ChunkColumn> const* Find(
      std::string_view entityPath,
      std::string_view component,
      bool isStatic,
      mochi::Error& error) const {
    MOCHI_ERROR_RETURN(error, nullptr);
    auto const found = chunks.find({std::string{entityPath}, std::string{component}, isStatic});
    if (found == chunks.end()) {
      SetMissingComponentError(entityPath, component, isStatic, error);
      return nullptr;
    }
    return &found->second;
  }

  template <typename Value, typename AppendFn>
  std::vector<Value> ReadStatic(
      std::string_view entityPath,
      std::string_view component,
      AppendFn append,
      char const* valueType,
      mochi::Error& error) const {
    auto const* const columns = Find(entityPath, component, true, error);
    MOCHI_ERROR_RETURN(error, {});
    // As in Rerun, the latest static write wins. A null entry is a row that does not update it.
    for (auto chunk = columns->rbegin(); chunk != columns->rend(); ++chunk) {
      for (int64_t row = chunk->values->length() - 1; row >= 0; --row) {
        if (chunk->values->IsNull(row)) {
          continue;
        }
        std::vector<Value> values;
        if (!append(*chunk->values, row, values)) {
          SetWrongTypeError(entityPath, component, valueType, error);
        }
        return values;
      }
    }
    SetMissingComponentError(entityPath, component, true, error);
    return {};
  }
};

EpisodeReader::EpisodeReader(std::unique_ptr<Impl> impl) : _impl{std::move(impl)} {}

EpisodeReader::~EpisodeReader() = default;

std::unique_ptr<EpisodeReader> EpisodeReader::Open(
    std::filesystem::path const& path,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error, nullptr);
  auto impl = std::make_unique<Impl>();
  for (auto& chunk : detail::ReadRrdChunks(path, error)) {
    auto const frames = FindTimeline(chunk, "frame");
    auto const simulationTimes = FindTimeline(chunk, "sim_time");
    for (auto& [component, values] : chunk.components) {
      impl->chunks[{chunk.entityPath, component, chunk.timelines.empty()}].push_back(
          {std::move(values), frames, simulationTimes});
    }
  }
  for (auto const& [component, columns] : impl->chunks) {
    impl->components.push_back(component);
  }
  std::unique_ptr<EpisodeReader> reader{new EpisodeReader{std::move(impl)}};
  ValidateSchema(*reader, path, error);
  MOCHI_ERROR_RETURN(error, nullptr);
  return reader;
}

std::vector<EpisodeComponent> const& EpisodeReader::GetComponents() const {
  return _impl->components;
}

bool EpisodeReader::HasComponent(
    std::string_view entityPath,
    std::string_view component,
    bool isStatic) const {
  return _impl->chunks.contains({std::string{entityPath}, std::string{component}, isStatic});
}

std::vector<std::string> EpisodeReader::ReadStaticTexts(
    std::string_view entityPath,
    std::string_view component,
    mochi::Error& error) const {
  return _impl->ReadStatic<std::string>(entityPath, component, AppendTexts, "text", error);
}

std::vector<double> EpisodeReader::ReadStaticScalars(
    std::string_view entityPath,
    std::string_view component,
    mochi::Error& error) const {
  return _impl->ReadStatic<double>(
      entityPath, component, AppendNumbers, "floats or doubles", error);
}

ScalarRows EpisodeReader::ReadScalars(
    std::string_view entityPath,
    std::string_view component,
    mochi::Error& error) const {
  auto const* const columns = _impl->Find(entityPath, component, false, error);
  MOCHI_ERROR_RETURN(error, {});
  struct Row {
    int64_t frame;
    ChunkColumn const* chunk;
    int64_t index;
  };
  std::vector<Row> rows;
  for (auto const& chunk : *columns) {
    if (chunk.frames == nullptr || chunk.simulationTimes == nullptr) {
      SUPERDEX_ERROR_SET(
          error,
          "Rows of '%.*s' on '%.*s' lack the frame or sim_time timeline.",
          static_cast<int>(component.size()),
          component.data(),
          static_cast<int>(entityPath.size()),
          entityPath.data());
      return {};
    }
    for (int64_t index = 0; index < chunk.values->length(); ++index) {
      if (chunk.values->IsValid(index)) {
        rows.push_back({chunk.frames->Value(index), &chunk, index});
      }
    }
  }
  std::ranges::stable_sort(rows, {}, &Row::frame);

  ScalarRows result;
  for (auto const& [frame, chunk, index] : rows) {
    if (!AppendNumbers(*chunk->values, index, result.values)) {
      SetWrongTypeError(entityPath, component, "floats or doubles", error);
      return {};
    }
    result.frames.push_back(frame);
    result.simulationTimesNs.push_back(chunk->simulationTimes->Value(index));
    result.offsets.push_back(result.values.size());
  }
  return result;
}

} // namespace superdex::recorder
