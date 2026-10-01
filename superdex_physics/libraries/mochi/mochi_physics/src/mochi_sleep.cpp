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

#include "mochi_sleep.h"

#include "mochi_common_components.h"
#include "mochi_differentiable.h"
#include "mochi_island.h"
#include "mochi_scene_recorder.h"

#include <mochi_core/utils/profile.h>

using namespace mochi;

void sleep::ValidateParams(
    entt::registry const& reg,
    experimental::SleepParams const& params,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF_NOT(
      params.sleepThreshold > 0_r && params.sleepThreshold <= 1_r,
      error,
      "Sleep threshold (SleepParams::sleepThreshold) must be in (0, 1].");
  MOCHI_ERROR_IF_NOT(
      params.minStepsBeforeSleep >= 2,
      error,
      "Minimum steps before sleep (SleepParams::minStepsBeforeSleep) must be at least 2.");
  MOCHI_ERROR_IF(
      params.canSleep && reg.try_ctx<TagDifferentiableScene const>() != nullptr,
      error,
      "Island sleeping (SleepParams::canSleep) is not supported in differentiable scenes.");
}

void sleep::InitIsland(entt::registry& reg, entt::entity island) {
  reg.emplace<TagIslandIsAwake>(island);
  reg.emplace<CIslandSleepState>(island);
}

real sleep::StageRestValue(NewtonSolverStatus<real> const& status) {
  if (status.convergence != ConvergenceStatus::Converged) {
    return 0_r;
  }
  if (status.numIterDone == 0) {
    return 1_r;
  }
  MOCHI_ASSERT_VERBOSE(status.resNorm0 > 0_r, "Iterations imply a positive initial residual.");
  return status.resNorm / status.resNorm0;
}

void sleep::RecordStep(entt::registry& reg, entt::entity island, real restValue) {
  auto const& params = reg.ctx<CSleepParams const>();
  auto& state = reg.get<CIslandSleepState>(island);
  if (restValue < params.sleepThreshold) {
    state.numConsecutiveRestSteps = 0;
  } else if (state.numConsecutiveRestSteps < params.minStepsBeforeSleep) {
    ++state.numConsecutiveRestSteps;
  }
}

void sleep::WakeAll(entt::registry& reg) {
  reg.ctx<CWakeUpRequests>().wakeAll = true;
}

// Tag every island that must wake up.
static void GatherWakeRequests(entt::registry& reg) {
  auto const& requests = reg.ctx<CWakeUpRequests const>();
  if (requests.wakeAll) {
    for (entt::entity island : reg.view<TagIsland>()) {
      reg.emplace_or_replace<TagWakeUp>(island);
    }
  }
}

// Wake the tagged islands. They must then stay awake for at least minStepsBeforeSleep solves.
static void ApplyWakeRequests(entt::registry& reg) {
  for (auto&& [island, state] : reg.view<TagWakeUp, CIslandSleepState>().each()) {
    state.numConsecutiveRestSteps = 0;
    reg.emplace_or_replace<TagIslandIsAwake>(island);
  }
  reg.clear<TagWakeUp>();
  reg.ctx<CWakeUpRequests>().wakeAll = false;
}

static void PutActorToSleep(entt::registry& reg, entt::entity e) {
  if (auto* convergence = reg.try_get<CConvergenceStatus>(e)) {
    convergence->stageStatus = ConvergenceStatus::None;
    convergence->stepStatus = ConvergenceStatus::None;
  }
}

static void PutIslandToSleep(entt::registry& reg, entt::entity island) {
  MOCHI_ASSERT_VERBOSE(
      reg.all_of<TagIslandIsAwake>(island), "Expected an island that was awake to begin with");

  // Remove TagIslandIsAwake which marked this island as awake. Note that it is safe to do this
  // while iterating over a view that is filtered by TagIslandIsAwake.
  reg.remove<TagIslandIsAwake>(island);

  // Prepare island members for sleep
  for (entt::entity e : reg.get<CIslandDescendants const>(island).actors) {
    PutActorToSleep(reg, e);
  }
}

// Put islands to sleep if they have been at rest long enough.
static void SleepRestingIslands(entt::registry& reg) {
  // Sleeping islands are not recorded, so nothing sleeps while recording.
  auto const& params = reg.ctx<CSleepParams const>();
  if (!params.canSleep || reg.try_ctx<TagSceneRecordingEnabled>()) {
    return;
  }
  MOCHI_ASSERT_VERBOSE(
      !reg.try_ctx<TagDifferentiableScene>(), "Sleep requires non-differentiable.");

  for (auto&& [island, state] :
       reg.view<CIslandSleepState const, TagIslandIsAwake const>().each()) {
    if (state.numConsecutiveRestSteps >= params.minStepsBeforeSleep) {
      PutIslandToSleep(reg, island);
    }
  }
}

void sleep::PreStep(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();
  GatherWakeRequests(reg);
  ApplyWakeRequests(reg);
  SleepRestingIslands(reg);
}

bool sleep::IsIslandAwake(entt::registry const& reg, entt::entity island) {
  MOCHI_ASSERT_VERBOSE(reg.all_of<TagIsland>(island), "Not an island.");
  return reg.all_of<TagIslandIsAwake>(island);
}

void sleep::InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CIslandSleepState>(reg);
  ecs::RegisterComponent<CSleepParams>(reg);
  ecs::RegisterComponent<CWakeUpRequests>(reg);
  ecs::RegisterComponent<TagIslandIsAwake>(reg);
  ecs::RegisterComponent<TagWakeUp>(reg);
  reg.set<CSleepParams>();
  reg.set<CWakeUpRequests>();
}
