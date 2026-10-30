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

#include <mochi_core/utils/error.h>
#include <mochi_core/utils/span.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace superdex::recorder {

/* @brief Names one component column of one entity. */
struct EpisodeComponent {
  /* As the recorder writes it, such as "data/physics/frames". */
  std::string entityPath;
  std::string component;
  /* Static rows have no frame. */
  bool isStatic = false;
};

/* @brief The rows of one component on one entity, ordered by frame.
 * Rows at the same frame keep their recorded order. Row i holds the values from offsets[i] up to
 * offsets[i + 1]. */
struct ScalarRows {
  std::vector<int64_t> frames;
  std::vector<int64_t> simulationTimesNs;
  std::vector<size_t> offsets{0};
  std::vector<double> values;

  [[nodiscard]] size_t GetNumRows() const {
    return frames.size();
  }

  [[nodiscard]] mochi::Span<double const> GetRow(size_t row) const {
    return {values.data() + offsets[row], offsets[row + 1] - offsets[row]};
  }
};

/* @brief Reads a SuperDex episode RRD without Rerun.
 * Open reads and validates the whole file, so reads do no file I/O, and const methods are safe to
 * call concurrently. Float and double components read as double, and a row of XYZ triples or
 * quaternions reads as one flat row. Reads only files written by the Rerun release that the
 * recorder links; after a Rerun upgrade, `rerun rrd migrate` brings older episodes up to date. */
class EpisodeReader final {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(EpisodeReader)

 public:
  static std::unique_ptr<EpisodeReader> Open(
      std::filesystem::path const& path,
      mochi::Error& error);
  ~EpisodeReader();

  /* Ordered by entity path, then component, with temporal entries before static ones. */
  [[nodiscard]] std::vector<EpisodeComponent> const& GetComponents() const;
  [[nodiscard]] bool
  HasComponent(std::string_view entityPath, std::string_view component, bool isStatic) const;

  /* Reads the latest static row. */
  [[nodiscard]] std::vector<std::string> ReadStaticTexts(
      std::string_view entityPath,
      std::string_view component,
      mochi::Error& error) const;
  [[nodiscard]] std::vector<double> ReadStaticScalars(
      std::string_view entityPath,
      std::string_view component,
      mochi::Error& error) const;

  /* Reads every temporal row. Fails if a row lacks the frame or sim_time timeline. */
  [[nodiscard]] ScalarRows
  ReadScalars(std::string_view entityPath, std::string_view component, mochi::Error& error) const;

 private:
  struct Impl;

  explicit EpisodeReader(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> _impl;
};

} // namespace superdex::recorder
