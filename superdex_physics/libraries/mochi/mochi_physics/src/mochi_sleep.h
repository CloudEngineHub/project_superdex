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

#include "mochi_ecs.h"

#include <mochi_core/utils/error.h>
#include <mochi_core/utils/no_copy.h>
#include <mochi_physics/mochi_physics_experimental.h>

namespace mochi {

// Scene-global island sleeping parameters. See experimental::SleepParams.
struct CSleepParams final : public experimental::SleepParams, NoCopy {};

} // namespace mochi

namespace mochi::sleep {

// Validates the sleep parameters against the scene, reporting problems through the error.
void ValidateParams(
    entt::registry const& reg,
    experimental::SleepParams const& params,
    Error& error);

void InitializeOnce(entt::registry& reg);

} // namespace mochi::sleep
