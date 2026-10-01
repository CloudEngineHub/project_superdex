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

#include "mochi_differentiable.h"

namespace mochi::sleep {

void ValidateParams(
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

void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CSleepParams>(reg);
  reg.set<CSleepParams>();
}

} // namespace mochi::sleep
