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

#pragma once

#include <superdex_recorder/series.h>

#include <mochi_core/utils/error.h>
#include <mochi_core/utils/span.h>

#include <rerun.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace superdex::recorder {

inline constexpr std::string_view kSchemaName = "superdex.episode";
inline constexpr uint32_t kSchemaVersionMajor = 1;
inline constexpr uint32_t kSchemaVersionMinor = 0;
inline constexpr std::string_view kSchemaCoordinateSpace = "FLU";
inline constexpr double kSchemaUnitsPerMeter = 1.0;

struct RecorderConfig {
  std::filesystem::path outputDirectory;
};

struct EpisodeMetadata {
  std::map<std::string, std::string> properties;
};

/* @brief Owns one transactional RRD and the constants and sample series buffered until commit. */
class Episode final {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(Episode)

 public:
  ~Episode();

  ScalarSeries* CreateScalarSeries(
      std::string_view entityPath,
      std::vector<ComponentField> fields,
      mochi::Error& error);
  TransformSeries* CreateTransformSeries(std::string_view entityPath, mochi::Error& error);
  void WriteConstantText(
      std::string_view entityPath,
      ComponentField const& field,
      std::string_view value,
      mochi::Error& error);
  void WriteConstantTexts(
      std::string_view entityPath,
      ComponentField const& field,
      mochi::Span<std::string const> values,
      mochi::Error& error);
  void WriteConstantScalars(
      std::string_view entityPath,
      ComponentField const& field,
      mochi::Span<double const> values,
      mochi::Error& error);
  void WriteConstantPose(
      std::string_view entityPath,
      ComponentField const& translationField,
      ComponentField const& quaternionField,
      std::array<double, 3> const& translation,
      std::array<double, 4> const& quaternionXyzw,
      mochi::Error& error);
  void WriteConstantPose(
      std::string_view entityPath,
      ComponentField const& translationField,
      ComponentField const& quaternionField,
      mochi::TransformRT const& transform,
      mochi::Error& error);
  void WriteMesh(
      std::string_view entityPath,
      mochi::Span<float const> vertexPositions,
      mochi::Span<uint32_t const> triangleIndices,
      std::array<uint8_t, 4> const& color,
      mochi::Error& error);

  /* Every sample stays in memory until Commit, which blocks the calling thread while it writes the
   * episode; a two-arm teleoperation scene records about 1 MB/s, so keep episodes to minutes. */
  void Commit(mochi::Error& error);
  void Abort(std::string_view reason, mochi::Error& error);

  [[nodiscard]] std::string const& GetEpisodeId() const;
  [[nodiscard]] std::filesystem::path const& GetOutputPath() const;
  [[nodiscard]] bool IsActive() const;

  static std::string EncodeEntityPathSegment(std::string_view segment);

 private:
  friend class Recorder;

  enum class State {
    Active,
    CommitFailed,
    Committed,
    Aborted,
  };

  struct StaticRows;

  Episode(
      std::string episodeId,
      std::filesystem::path temporaryPath,
      std::filesystem::path outputPath,
      std::unique_ptr<rerun::RecordingStream> stream);

  void ValidateWrite(mochi::Error& error) const;
  void ValidateSeriesPath(std::string_view entityPath, mochi::Error& error) const;
  void SealSeries();
  void DiscardTemporaryFile(mochi::Error* error);

  std::string _episodeId;
  std::filesystem::path _temporaryPath;
  std::filesystem::path _outputPath;
  std::unique_ptr<rerun::RecordingStream> _stream;
  std::unique_ptr<StaticRows> _staticRows;
  std::vector<std::unique_ptr<Series>> _series;
  State _state = State::Active;
};

/* @brief Starts transactional episode recordings in one output directory. */
class Recorder final {
 public:
  explicit Recorder(RecorderConfig config);

  std::unique_ptr<Episode> BeginEpisode(EpisodeMetadata const& metadata, mochi::Error& error) const;

 private:
  RecorderConfig _config;
};

} // namespace superdex::recorder
