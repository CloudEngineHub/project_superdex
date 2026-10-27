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

#include "rendering/scene_stage.h"

#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_renderer/buffer.h>

#include <string>
#include <vector>

namespace superdex::studio {

// Mirrors the simulated actors of a staged prefab into its SceneStage: rigid/soft root transforms,
// articulated link transforms, soft/skin surface deformation and skin poses. Track/Untrack/Produce
// run on the sim thread; Apply runs on the UI thread.
class PrefabSimSync {
 public:
  explicit PrefabSimSync(mochi::CoordinateSpaceConverter const& editorToRenderer)
      : _editorToRenderer(editorToRenderer) {}

  // Tracks the actors added by prefab::AddToScene and registers the soft/skin surface queries.
  // Returns false (and logs) if a query cannot be registered.
  bool Track(mochi::Span<mochi::Actor* const> actors);
  // Cancels the surface queries. Call while the actors are still alive.
  void Untrack(mochi::Scene* scene);
  // Post-step: captures the tracked actors' state for the UI thread.
  void Produce(mochi::Scene* scene);
  // Applies the latest captured state to `stage`. Returns false if nothing new was produced.
  bool Apply(SceneStage& stage, mochi::CoordinateSpaceConverter const& editorToRenderer);
  // Drops tracked actors and any pending state (after the scene is destroyed).
  void Reset();

  // Names of the actors in the last applied state (for the Render Scene Stage debug window).
  [[nodiscard]] std::vector<std::string> const& GetAppliedActorNames() {
    return _simData.GetConsumerData().actorNames;
  }

 private:
  struct SimData {
    std::vector<std::string> actorNames;
    std::vector<mochi::TransformRT> actorTransforms;
    std::vector<SoftMeshUpdate> softMeshUpdates;
    // Per-step skin joint poses (one per skinned articulated actor), applied to each staged skin's
    // GPU-skinned render model. Each carries the actor's per-link world transforms in nested-link
    // (== GLB joint) order.
    std::vector<SkinnedPoseUpdate> skinnedPoseUpdates;
  };
  // Per-soft-actor query registration held while simulating. `name` matches the staged actor for
  // the per-frame mesh update.
  struct SoftQueryState {
    mochi::ActorHandle actor;
    mochi::QueryHandle positions;
    mochi::QueryHandle normals;
    std::string name;
  };

  mochi::CoordinateSpaceConverter const& _editorToRenderer;
  std::vector<mochi::ActorHandle> _actors;
  std::vector<SoftQueryState> _softQueryState;
  mochi_renderer::ProducerConsumerBuffer<SimData> _simData;
};

} // namespace superdex::studio
