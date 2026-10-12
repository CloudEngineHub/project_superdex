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

#include <superdex_recorder/recorder.h>
#include <superdex_recorder/series.h>

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <random>
#include <regex>
#include <string>
#include <vector>

namespace superdex::recorder {
namespace {

class TemporaryDirectory final {
 public:
  TemporaryDirectory() {
    auto const suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    _path = std::filesystem::temp_directory_path() /
        ("superdex_recorder_test_" + std::to_string(suffix) + "_" +
         std::to_string(std::random_device{}()));
    std::filesystem::create_directories(_path);
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(_path, error);
  }

  [[nodiscard]] std::filesystem::path const& Path() const {
    return _path;
  }

 private:
  std::filesystem::path _path;
};

uint64_t UnixMilliseconds() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::system_clock::now().time_since_epoch())
                                   .count());
}

void AppendSample(Episode& episode, mochi::Error& error) {
  auto* const series = episode.CreateScalarSeries(
      "data/test", {{"superdex.archetypes.Test", "superdex.test.value"}}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(series, nullptr);
  std::array<mochi::real, 1> const value{static_cast<mochi::real>(42)};
  std::array<mochi::Span<mochi::real const>, 1> const fields{value};
  series->Append({.frame = 0, .simulationTimeSec = 0.0}, fields, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
}

TEST(SuperDexRecorderTest, CommitPublishesOnlyFinalFile) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;

  auto episode = recorder.BeginEpisode({.properties = {{"task", "test"}}}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  auto const outputPath = episode->GetOutputPath();
  EXPECT_FALSE(std::filesystem::exists(outputPath));
  ASSERT_NO_FATAL_FAILURE(AppendSample(*episode, error));

  episode->Commit(error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_FALSE(episode->IsActive());
  std::vector<std::filesystem::path> const expectedFiles{outputPath};
  std::vector<std::filesystem::path> const files{
      std::filesystem::directory_iterator{directory.Path()}, std::filesystem::directory_iterator{}};
  EXPECT_EQ(files, expectedFiles);
  EXPECT_GT(std::filesystem::file_size(outputPath), 0);
}

TEST(SuperDexRecorderTest, CommitFailureLeavesNoPartialFile) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  std::filesystem::path outputPath;
  {
    auto episode = recorder.BeginEpisode({}, error);
    ASSERT_TRUE(error.IsOK()) << error.ToString();
    ASSERT_NE(episode, nullptr);
    ASSERT_NO_FATAL_FAILURE(AppendSample(*episode, error));
    outputPath = episode->GetOutputPath();
    // No platform renames a file over a non-empty directory.
    std::filesystem::create_directories(outputPath / "blocker");

    episode->Commit(error);

    EXPECT_FALSE(error.IsOK());
    EXPECT_FALSE(episode->IsActive());
  }

  std::vector<std::filesystem::path> const expectedFiles{outputPath};
  std::vector<std::filesystem::path> const files{
      std::filesystem::directory_iterator{directory.Path()}, std::filesystem::directory_iterator{}};
  EXPECT_EQ(files, expectedFiles);
}

TEST(SuperDexRecorderTest, AbortDiscardsEpisode) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;

  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  auto const outputPath = episode->GetOutputPath();

  episode->Abort("rejected", error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_FALSE(episode->IsActive());
  EXPECT_FALSE(std::filesystem::exists(outputPath));
  EXPECT_TRUE(std::filesystem::is_empty(directory.Path()));
}

TEST(SuperDexRecorderTest, AbortDiscardsEpisodeAfterRecordingError) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  error.SetFirstError("recording failed", __FILE__, __LINE__);

  episode->Abort("rejected after recording failure", error);

  EXPECT_FALSE(episode->IsActive());
  EXPECT_TRUE(std::filesystem::is_empty(directory.Path()));
}

TEST(SuperDexRecorderTest, AppendAfterAbortIsRejected) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  auto* const series = episode->CreateScalarSeries(
      "data/test", {{"superdex.archetypes.Test", "superdex.test.value"}}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(series, nullptr);
  episode->Abort("rejected", error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  std::array<mochi::real, 1> const value{static_cast<mochi::real>(42)};
  std::array<mochi::Span<mochi::real const>, 1> const fields{value};

  series->Append({.frame = 0, .simulationTimeSec = 0.0}, fields, error);

  EXPECT_FALSE(error.IsOK());
}

TEST(SuperDexRecorderTest, DestructionDiscardsUncommittedEpisode) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;

  {
    auto episode = recorder.BeginEpisode({}, error);
    ASSERT_TRUE(error.IsOK()) << error.ToString();
    ASSERT_NE(episode, nullptr);
    ASSERT_NO_FATAL_FAILURE(AppendSample(*episode, error));
  }

  EXPECT_TRUE(std::filesystem::is_empty(directory.Path()));
}

TEST(SuperDexRecorderTest, EpisodesHaveIndependentIdsAndFiles) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;

  auto first = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(first, nullptr);
  auto second = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(second, nullptr);
  EXPECT_NE(first->GetEpisodeId(), second->GetEpisodeId());

  first->Commit(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  second->Commit(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_TRUE(std::filesystem::exists(first->GetOutputPath()));
  EXPECT_TRUE(std::filesystem::exists(second->GetOutputPath()));
}

TEST(SuperDexRecorderTest, EpisodeIdIsUuidV7WithStartTimestamp) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;

  uint64_t const before = UnixMilliseconds();
  auto episode = recorder.BeginEpisode({}, error);
  uint64_t const after = UnixMilliseconds();
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);

  std::string const& id = episode->GetEpisodeId();
  ASSERT_TRUE(
      std::regex_match(
          id, std::regex{"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"}))
      << id;
  uint64_t const timestamp = std::stoull(id.substr(0, 8) + id.substr(9, 4), nullptr, 16);
  EXPECT_GE(timestamp, before);
  EXPECT_LE(timestamp, after);
}

TEST(SuperDexRecorderTest, BeginEpisodeRejectsEmptyPropertyKeyBeforeCreatingDirectory) {
  TemporaryDirectory directory;
  auto const outputDirectory = directory.Path() / "recordings";
  Recorder recorder{{.outputDirectory = outputDirectory}};
  mochi::Error error;

  auto episode = recorder.BeginEpisode({.properties = {{"", "value"}}}, error);

  EXPECT_EQ(episode, nullptr);
  EXPECT_FALSE(error.IsOK());
  EXPECT_FALSE(std::filesystem::exists(outputDirectory));
}

TEST(SuperDexRecorderTest, ScalarSeriesRejectsRepeatedFrames) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  auto* const series = episode->CreateScalarSeries(
      "data/test", {{"superdex.archetypes.Test", "superdex.test.value"}}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(series, nullptr);
  std::array<mochi::real, 1> const value{static_cast<mochi::real>(42)};
  std::array<mochi::Span<mochi::real const>, 1> const fields{value};
  series->Append({.frame = 1, .simulationTimeSec = 0.1}, fields, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  mochi::Error appendError;
  series->Append({.frame = 1, .simulationTimeSec = 0.2}, fields, appendError);

  EXPECT_FALSE(appendError.IsOK());
}

TEST(SuperDexRecorderTest, CreateSeriesRejectsExistingEntityPath) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  ASSERT_NE(episode->CreateTransformSeries("world/test", error), nullptr);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  mochi::Error scalarError;
  auto* const scalarSeries = episode->CreateScalarSeries(
      "world/test", {{"superdex.archetypes.Test", "superdex.test.value"}}, scalarError);
  mochi::Error transformError;
  auto* const transformSeries = episode->CreateTransformSeries("world/test", transformError);
  mochi::Error arrowError;
  auto* const arrowSeries = episode->CreateArrow3DSeries("world/test", arrowError);

  EXPECT_EQ(scalarSeries, nullptr);
  EXPECT_FALSE(scalarError.IsOK());
  EXPECT_EQ(transformSeries, nullptr);
  EXPECT_FALSE(transformError.IsOK());
  EXPECT_EQ(arrowSeries, nullptr);
  EXPECT_FALSE(arrowError.IsOK());
}

TEST(SuperDexRecorderTest, WriteMeshRejectsOutOfRangeIndices) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  std::array<float, 9> const vertices{0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
  std::array<uint32_t, 3> const triangles{0, 1, 3};

  episode->WriteMesh("world/test", vertices, triangles, {255, 255, 255, 255}, error);

  EXPECT_FALSE(error.IsOK());
}

} // namespace
} // namespace superdex::recorder
