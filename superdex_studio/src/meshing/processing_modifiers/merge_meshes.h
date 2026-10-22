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

// "Merge" modifier: combines the connected pieces of its input (e.g. the parts of a Source from
// Parts) into one closed surface, by their boolean union or one shrink-wrap around them all.

#include "meshing/processing_modifiers/processing_modifier.h"

#include <mochi_core/geometry/mesh_data.h>

#include <vector>

namespace superdex::studio {

// The Union method's name, as stored in pipeline files. The other method is Wrap Mesh's
// kShrinkWrapMethod.
inline constexpr char const* kMergeUnionMethod = "Union";

ModifierRegistryEntry MakeMergeMeshesEntry();

// The connected pieces of @p mesh (triangles sharing vertices), each with its own vertices.
std::vector<mochi::MeshData> SplitConnectedPieces(mochi::MeshData const& mesh);

} // namespace superdex::studio
