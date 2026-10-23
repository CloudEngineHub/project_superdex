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

#include <superdex_recorder/episode_restorer.h>

#include <mochi_physics/mochi_physics_experimental.h>

#include <numeric>
#include <utility>
#include <vector>

namespace superdex::recorder {
namespace {

mochi::Real3 GetXyz(std::vector<mochi::real> const& values, size_t frameIndex) {
  size_t const offset = 3 * frameIndex;
  return {values[offset], values[offset + 1], values[offset + 2]};
}

mochi::Quaternion GetXyzw(std::vector<mochi::real> const& values, size_t frameIndex) {
  size_t const offset = 4 * frameIndex;
  return mochi::Quaternion{
      values[offset], values[offset + 1], values[offset + 2], values[offset + 3]};
}

mochi::Span<mochi::real const>
GetFrameValues(std::vector<mochi::real> const& values, size_t frameIndex, size_t width) {
  return {values.data() + frameIndex * width, width};
}

mochi::TransformRT GetWorldFromCom(RestoredActor const& actor, size_t frameIndex) {
  return mochi::TransformRT{
      GetXyzw(actor.rootRotations, frameIndex), GetXyz(actor.comTranslations, frameIndex)};
}

mochi::TransformRT GetWorldFromRoot(RestoredActor const& actor, size_t frameIndex) {
  return mochi::TransformRT{
      GetXyzw(actor.rootRotations, frameIndex), GetXyz(actor.rootTranslations, frameIndex)};
}

bool HasFrames(std::vector<mochi::real> const& values, size_t width, size_t numFrames) {
  return values.size() == width * numFrames;
}

} // namespace

EpisodeRestorer::EpisodeRestorer(
    mochi::Scene& scene,
    std::vector<double> simulationTimesSec,
    std::vector<double> stepDtsSec,
    std::vector<RestoredActor> actors,
    mochi::Error& error)
    : _scene{scene},
      _simulationTimesSec{std::move(simulationTimesSec)},
      _stepDtsSec{std::move(stepDtsSec)},
      _actors{std::move(actors)} {
  size_t const numFrames = _simulationTimesSec.size();
  MOCHI_ERROR_IF(
      numFrames == 0 || _stepDtsSec.size() != numFrames, error, "Invalid recorded frame timing.");
  MOCHI_ERROR_RETURN(error);

  for (auto const& actor : _actors) {
    mochi::Actor const* const sceneActor = _scene.GetActor(actor.handle);
    MOCHI_ERROR_IF(sceneActor == nullptr, error, "Invalid restored actor.");
    MOCHI_ERROR_RETURN(error);
    bool isValid = HasFrames(actor.rootRotations, 4, numFrames);
    if (actor.isArticulated) {
      auto const numDofs = static_cast<size_t>(sceneActor->GetNumDofs());
      isValid = isValid && HasFrames(actor.rootTranslations, 3, numFrames) &&
          HasFrames(actor.jointPositions, numDofs, numFrames) &&
          HasFrames(actor.jointVelocities, numDofs, numFrames) &&
          (actor.controllerTargets.empty() ||
           HasFrames(actor.controllerTargets, numDofs, numFrames)) &&
          (actor.jointEfforts.empty() || HasFrames(actor.jointEfforts, numDofs, numFrames));
    } else {
      isValid = isValid && HasFrames(actor.comTranslations, 3, numFrames) &&
          HasFrames(actor.linearVelocities, 3, numFrames) &&
          HasFrames(actor.angularVelocities, 3, numFrames);
    }
    MOCHI_ERROR_IF_NOT(isValid, error, "Restored actor state must have one value set per frame.");
    MOCHI_ERROR_RETURN(error);
  }
}

EpisodeRestorer::~EpisodeRestorer() = default;

size_t EpisodeRestorer::GetNumFrames() const {
  return _simulationTimesSec.size();
}

void EpisodeRestorer::RestoreFrame(size_t frameIndex, mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(frameIndex >= GetNumFrames(), error, "Frame index is out of range.");
  MOCHI_ERROR_RETURN(error);
  mochi::experimental::SetSimulationTime(
      &_scene, _simulationTimesSec[frameIndex], _stepDtsSec[frameIndex], error);
  for (auto const& actor : _actors) {
    mochi::Actor* const sceneActor = _scene.GetActor(actor.handle);
    MOCHI_ERROR_IF(sceneActor == nullptr, error, "A restored actor was destroyed.");
    MOCHI_ERROR_RETURN(error);
    // External forces persist until replaced, so clear any left from before the restore.
    sceneActor->ClearExternalForces();
    if (!actor.isArticulated) {
      sceneActor->SetCenterOfMassTransform(GetWorldFromCom(actor, frameIndex), error);
      sceneActor->SetVelocity(
          GetXyz(actor.linearVelocities, frameIndex),
          GetXyz(actor.angularVelocities, frameIndex),
          error);
      continue;
    }

    auto const numDofs = static_cast<size_t>(sceneActor->GetNumDofs());
    sceneActor->SetRootTransform(GetWorldFromRoot(actor, frameIndex), error);
    sceneActor->SetArticulatedPoseFromJoints(
        GetFrameValues(actor.jointPositions, frameIndex, numDofs), error);
    sceneActor->SetArticulatedJointVelocities(
        GetFrameValues(actor.jointVelocities, frameIndex, numDofs), error);
    // The pose setter resets the target, so set it afterwards.
    if (!actor.controllerTargets.empty()) {
      sceneActor->SetArticulatedTargetPose(
          GetFrameValues(actor.controllerTargets, frameIndex, numDofs), error);
    }
    if (!actor.jointEfforts.empty()) {
      std::vector<int> dofIndices(numDofs);
      std::iota(dofIndices.begin(), dofIndices.end(), 0);
      sceneActor->SetExternalForcesOnDofs(
          dofIndices, GetFrameValues(actor.jointEfforts, frameIndex, numDofs), error);
    }
  }
  MOCHI_ERROR_RETURN(error);
  _scene.Step(0.0);
}

} // namespace superdex::recorder
