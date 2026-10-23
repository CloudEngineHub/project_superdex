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

#include <superdex_recorder/export.h>

#include <mochi_core/utils/error.h>
#include <mochi_physics/cpp_api/mochi_scene.h>

#include <cstddef>
#include <vector>

namespace superdex::recorder {

/* @brief Recorded state of one dynamic actor at every frame, concatenated in frame order. */
struct RestoredActor {
  mochi::ActorHandle handle;
  bool isArticulated = false;
  /* xyzw per frame. */
  std::vector<mochi::real> rootRotations;
  /* Rigid actors: center-of-mass position and velocities, xyz per frame. */
  std::vector<mochi::real> comTranslations;
  std::vector<mochi::real> linearVelocities;
  std::vector<mochi::real> angularVelocities;
  /* Articulated actors: root position (xyz per frame) and one value per DoF per frame. */
  std::vector<mochi::real> rootTranslations;
  std::vector<mochi::real> jointPositions;
  std::vector<mochi::real> jointVelocities;
  /* Empty for actors without a pose controller. */
  std::vector<mochi::real> controllerTargets;
  /* External forces during the step after each frame, one value per DoF per frame. Empty for
   * actors without recorded joint efforts. */
  std::vector<mochi::real> jointEfforts;
};

/* @brief Restores a scene to the end of any recorded step.
 * Each restore sets the scene clock, each actor's state through the public setters, pose-controller
 * targets, and external forces, then takes a zero step. The integrator starts afresh from the
 * restored state. The scene must outlive the restorer. */
class SUPERDEX_RECORDER_API EpisodeRestorer final {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(EpisodeRestorer)

 public:
  EpisodeRestorer(
      mochi::Scene& scene,
      std::vector<double> simulationTimesSec,
      std::vector<double> stepDtsSec,
      std::vector<RestoredActor> actors,
      mochi::Error& error);
  ~EpisodeRestorer();

  [[nodiscard]] size_t GetNumFrames() const;

  void RestoreFrame(size_t frameIndex, mochi::Error& error);

 private:
  mochi::Scene& _scene;
  std::vector<double> _simulationTimesSec;
  std::vector<double> _stepDtsSec;
  std::vector<RestoredActor> _actors;
};

} // namespace superdex::recorder
