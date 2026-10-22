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

// "Wrap Mesh" modifier: builds a fresh closed envelope around the input (shrink wrap, convex hull,
// alpha wrap). Each is a method. This header exposes the registry entry and the Shrink Wrap method,
// which the Merge modifier offers too.

#include "meshing/processing_modifiers/processing_modifier.h"

#include <memory>

namespace superdex::studio {

ModifierRegistryEntry MakeWrapMeshEntry();

// The Shrink Wrap method's name, as stored in pipeline files.
inline constexpr char const* kShrinkWrapMethod = "Shrink Wrap";

// The "Shrink Wrap" method: one envelope deflated onto the whole input, however many pieces it has.
std::unique_ptr<MeshProcessingMethod> MakeShrinkWrapMethod();

} // namespace superdex::studio
