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

#include "superdex_asset_tuner/run_writer.h"
#include "superdex_asset_tuner/tuner_session.h"
#include "superdex_asset_tuner/tuner_spec.h"

#include <filesystem>
#include <functional>
#include <string>

namespace superdex::tuner {

struct RunOptions {
  // Run directories are created under this folder. Empty: no files are written.
  std::filesystem::path outDir;
  std::string tag;
  // Called on every post-step sample, after it is written.
  std::function<void(Span<DynamicString const> channelNames, Sample const& sample)> onSample;
};

// Name of the ground plane actor the runner adds when RunSpec::groundPlane is set.
inline constexpr std::string_view kGroundPlaneName = "TunerGroundPlane";

// Sets default gravity. Call before adding the prefab so prefab scene settings win; then call
// AddGroundPlane and ApplyRunSpecOverrides.
void PrepareScene(mochi::Scene& scene);

// Lowest point (world Z) of `actors`' geometry, including articulated actors' links, ignoring
// unbounded geometry such as planes; 0 if none is bounded. This is the automatic ground height.
[[nodiscard]] real ComputeAutoGroundHeight(Span<mochi::Actor* const> actors);

// Adds the RunSpec ground plane (if enabled) at RunSpec::groundHeight, or at the automatic height
// of `prefabActors` when unset. Returns the height used, or nullopt when no plane was added.
std::optional<real>
AddGroundPlane(mochi::Scene& scene, RunSpec const& run, Span<mochi::Actor* const> prefabActors);

// Applies RunSpec values that override the prefab (currently gravity).
void ApplyRunSpecOverrides(mochi::Scene& scene, RunSpec const& run);

// Loads the tuner's prefab into a fresh headless scene and steps it for `spec.run.duration`.
// Returns false only for setup errors; a run that diverges returns true with status "diverged".
bool RunTuner(
    TunerSpec const& spec,
    std::filesystem::path const& tunerFilePath,
    RunOptions const& options,
    RunSummary& outSummary,
    std::string& outError);

} // namespace superdex::tuner
