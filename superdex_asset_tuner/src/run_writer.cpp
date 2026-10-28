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

#include "superdex_asset_tuner/run_writer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace superdex::tuner {

namespace {

std::string MakeTimestampUtc() {
  auto const now = std::chrono::system_clock::now();
  auto const day = std::chrono::floor<std::chrono::days>(now);
  std::chrono::year_month_day const ymd{day};
  std::chrono::hh_mm_ss const hms{std::chrono::floor<std::chrono::seconds>(now - day)};
  char buffer[32];
  std::snprintf(
      buffer,
      sizeof(buffer),
      "%04d%02u%02u-%02d%02d%02d",
      static_cast<int>(ymd.year()),
      static_cast<unsigned>(ymd.month()),
      static_cast<unsigned>(ymd.day()),
      static_cast<int>(hms.hours().count()),
      static_cast<int>(hms.minutes().count()),
      static_cast<int>(hms.seconds().count()));
  return buffer;
}

void WriteNumber(std::ofstream& out, double value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.9g", value);
  out << buffer;
}

} // namespace

void UpdateChannelSummaries(
    Span<real const> values,
    bool first,
    Span<ChannelSummary> inOutChannels) {
  size_t const count = std::min(values.size(), inOutChannels.size());
  for (size_t i = 0; i < count; ++i) {
    ChannelSummary& channel = inOutChannels[i];
    real const v = values[i];
    channel.min = first ? v : std::min(channel.min, v);
    channel.max = first ? v : std::max(channel.max, v);
    channel.final = v;
  }
}

bool RunWriter::Open(
    std::filesystem::path const& outDir,
    std::string_view tunerName,
    std::string_view tag,
    TunerSpec const& resolvedSpec,
    std::filesystem::path const& prefabPath,
    std::string& outError) {
  std::string baseName = MakeTimestampUtc() + "_" + std::string(tunerName);
  if (!tag.empty()) {
    baseName += "_" + std::string(tag);
  }
  std::error_code ec;
  _runDir = outDir / baseName;
  for (int suffix = 1; std::filesystem::exists(_runDir, ec); ++suffix) {
    _runDir = outDir / (baseName + "_" + std::to_string(suffix));
  }
  std::filesystem::create_directories(_runDir, ec);
  if (ec) {
    outError = "Cannot create run directory '" + _runDir.string() + "': " + ec.message();
    return false;
  }
  _runDir = std::filesystem::absolute(_runDir);

  TunerSpec resolved = resolvedSpec;
  resolved.prefab = DynamicString(prefabPath.string());
  if (!SaveTunerSpecToFile(resolved, _runDir / "tuner.resolved.json", outError)) {
    return false;
  }
  _signals.open(_runDir / "signals.csv", std::ios::binary);
  if (!_signals) {
    outError = "Cannot write '" + (_runDir / "signals.csv").string() + "'.";
    return false;
  }
  return true;
}

void RunWriter::WriteHeader(Span<DynamicString const> channelNames) {
  _signals << "t";
  for (auto const& name : channelNames) {
    _signals << ',' << std::string_view(name);
  }
  _signals << '\n';
}

void RunWriter::WriteSample(Sample const& sample) {
  WriteNumber(_signals, sample.time);
  for (real const value : sample.values) {
    _signals << ',';
    WriteNumber(_signals, value);
  }
  _signals << '\n';
}

bool RunWriter::Close(RunSummary& inOutSummary, std::string& outError) {
  _signals.close();
  inOutSummary.runDir = DynamicString(_runDir.string());
  std::ofstream summary(_runDir / "summary.json", std::ios::binary);
  summary << SReflect::ToJsonString(inOutSummary, true);
  if (!summary) {
    outError = "Cannot write '" + (_runDir / "summary.json").string() + "'.";
    return false;
  }
  return true;
}

} // namespace superdex::tuner
