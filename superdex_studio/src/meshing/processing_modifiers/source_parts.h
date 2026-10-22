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

// "Source from Parts" modifier: assembles a list of mesh files and primitives, each placed in the
// model's frame, into one mesh whose connected pieces are the parts (see the Merge modifier).

#include "meshing/processing_modifiers/processing_modifier.h"

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/reflection.h>

#include <memory>
#include <vector>

namespace superdex::studio {

enum class SourcePartShape { Mesh, Box, Sphere, Cylinder, Capsule, Ellipsoid };

} // namespace superdex::studio

// Enum reflection specializes a global template, so it is declared at global scope.
MOCHI_ENUM_BEGIN(superdex::studio::SourcePartShape)
MOCHI_ENUM_ITEM(Mesh)
MOCHI_ENUM_ITEM(Box)
MOCHI_ENUM_ITEM(Sphere)
MOCHI_ENUM_ITEM(Cylinder)
MOCHI_ENUM_ITEM(Capsule)
MOCHI_ENUM_ITEM(Ellipsoid)
MOCHI_ENUM_END()

namespace superdex::studio {

using mochi::operator""_r;

// One part: a mesh file or a primitive. Its placement and sizes are in Mochi's space (Z-up), the
// space of the bot's link frames; the source converts the assembled parts into the pipeline's
// renderer space.
struct SourcePart {
  SourcePartShape shape = SourcePartShape::Mesh;
  // Mesh file (OBJ, STL, DAE, GLB, or .mochi.h5), for a Mesh part.
  mochi::DynamicString path;
  // Primitive size [m]. Box: half-extents. Sphere: radius. Cylinder and capsule: radius and
  // half-length along Z. Ellipsoid: radii.
  mochi::Real3 size = {0.05_r, 0.05_r, 0.05_r};
  // Mesh scale, applied before the placement.
  mochi::Real3 scale = {1_r, 1_r, 1_r};
  mochi::Quaternion rotation = {};
  mochi::Real3 translation = {}; // [m]

  MOCHI_STRUCT_BEGIN(superdex::studio::SourcePart)
  MOCHI_FIELD(shape)
  MOCHI_FIELD(path)
  MOCHI_FIELD(size)
  MOCHI_FIELD(scale)
  MOCHI_FIELD(rotation)
  MOCHI_FIELD(translation)
  MOCHI_STRUCT_END()
};

ModifierRegistryEntry MakePartsSourceEntry();

// A "Source from Parts" modifier holding @p parts.
std::unique_ptr<MeshProcessingModifier> MakePartsSourceModifier(std::vector<SourcePart> parts);

} // namespace superdex::studio
