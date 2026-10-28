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

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_physics/mochi_physics.h>

namespace superdex::tuner::details {

// All actors except articulated links (links are reached through their articulated actor).
[[nodiscard]] mochi::DynamicArray<mochi::Actor*> GetTopLevelActors(mochi::Scene& scene);

// Whether `actor` has a finite world bounding box (planes do not).
[[nodiscard]] bool IsBounded(mochi::Actor const& actor);

} // namespace superdex::tuner::details
