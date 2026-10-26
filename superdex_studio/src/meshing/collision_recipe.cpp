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

#include "meshing/collision_recipe.h"

#include "meshing/processing_modifiers/merge_meshes.h"
#include "meshing/processing_modifiers/wrap_mesh.h" // kShrinkWrapMethod

#include <picojson/picojson.h>

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_renderer/path.h>
#include <mochi_renderer/render_space.h>

#include <cmath>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace superdex::studio {

using superdex::robotics::GeomReference;
using superdex::robotics::GeomType;
using mochi::operator""_r;

namespace {

SourcePartShape ToPartShape(GeomType type) {
  switch (type) {
    case GeomType::Box:
      return SourcePartShape::Box;
    case GeomType::Sphere:
      return SourcePartShape::Sphere;
    case GeomType::Cylinder:
      return SourcePartShape::Cylinder;
    case GeomType::Capsule:
      return SourcePartShape::Capsule;
    case GeomType::Ellipsoid:
      return SourcePartShape::Ellipsoid;
    case GeomType::Mesh:
    case GeomType::Plane:
    case GeomType::Hfield:
    case GeomType::Sdf:
      break;
  }
  return SourcePartShape::Mesh;
}

// A registered modifier with method @p method and the properties in @p props (a JSON object, whose
// missing fields keep their defaults).
std::unique_ptr<MeshProcessingModifier>
MakeModifier(std::string_view name, char const* method, picojson::object const& props) {
  std::unique_ptr<MeshProcessingModifier> modifier = MakeProcessingModifier(name);
  MOCHI_ASSERT(modifier != nullptr);
  [[maybe_unused]] bool const selected = modifier->SelectMethodByName(method);
  [[maybe_unused]] int const issues =
      modifier->ActiveMethod().DeserializeProps(picojson::value(props));
  MOCHI_ASSERT(selected && issues == 0);
  return modifier;
}

picojson::value JsonArray(std::initializer_list<mochi::real> values) {
  picojson::array array;
  for (mochi::real const value : values) {
    array.emplace_back(static_cast<double>(value));
  }
  return {std::move(array)};
}

// The source that reads @p path on its own ("From File"), chosen by the file's type, or null for a
// type none of them reads.
std::unique_ptr<MeshProcessingModifier> FileSource(std::string const& path) {
  std::string const lower = mochi::Path{path}.AsLowercaseString();
  char const* name = nullptr;
  if (lower.ends_with(".stl")) {
    name = "Source from CAD Model";
  } else if (
      lower.ends_with(".obj") || lower.ends_with(".glb") || lower.ends_with(".gltf") ||
      lower.ends_with(".dae")) {
    name = "Source from Render Model";
  } else if (lower.ends_with(".mochi.h5")) {
    name = "Source from Mochi Model";
  }
  return name != nullptr ? MakeModifier(name, "From File", {{"path", picojson::value(path)}})
                         : nullptr;
}

// @p part's placement in the pipeline's renderer space. The part is placed in Mochi's space, which
// differs from the renderer's by a change of axes.
struct RenderPlacement {
  mochi::Real3 scale;
  mochi::Quaternion rotation;
  mochi::Real3 translation;
};

RenderPlacement ToRenderPlacement(SourcePart const& part) {
  mochi::CoordinateSpaceConverter const toRender(
      mochi::CoordinateSpace::Default(), mochi_renderer::RenderSpace());
  // A scale is per axis: the change of axes reorders it, but must not flip the sign of an axis
  // whose direction it reverses, so each takes the sign the reordered magnitudes carry.
  mochi::Real3 scale = toRender.DirectionToOutput(part.scale);
  mochi::Real3 const axisSigns = toRender.DirectionToOutput(mochi::Abs(part.scale));
  for (int axis = 0; axis < 3; ++axis) {
    scale[axis] = std::copysign(1_r, axisSigns[axis]) * scale[axis];
  }
  return {
      scale,
      toRender.RotationToOutput(part.rotation),
      toRender.TranslationToOutput(part.translation)};
}

// An Apply SRT Transformation placing @p part's mesh in the model's frame, or null when the
// placement is the identity.
std::unique_ptr<MeshProcessingModifier> Placement(SourcePart const& part) {
  constexpr mochi::real kTolerance = 1e-6_r;
  bool const identity = NearEqual(part.scale, mochi::Real3{1_r, 1_r, 1_r}, kTolerance) &&
      NearEqual(part.translation, mochi::Real3{}, kTolerance) &&
      NearEqual(mochi::Abs(part.rotation.ToReal4()[3]), 1_r, kTolerance);
  if (identity) {
    return nullptr;
  }
  RenderPlacement const placement = ToRenderPlacement(part);
  mochi::Real3 const& scale = placement.scale;
  mochi::Real4 const rotation = placement.rotation.ToReal4();
  mochi::Real3 const& translation = placement.translation;
  return MakeModifier(
      "Apply SRT Transformation",
      "SRT Transform",
      {{"scale", JsonArray({scale[0], scale[1], scale[2]})},
       {"rotation", JsonArray({rotation[0], rotation[1], rotation[2], rotation[3]})},
       {"translation", JsonArray({translation[0], translation[1], translation[2]})}});
}

} // namespace

SourcePart PartFromGeom(GeomReference const& geom, std::string const& meshFile) {
  SourcePart part;
  part.shape = ToPartShape(geom.type);
  mochi::TransformRT pose = geom.linkFromGeom;
  if (part.shape == SourcePartShape::Mesh) {
    part.path = meshFile;
    part.scale = geom.meshScale;
    pose = geom.linkFromGeom * geom.geomFromMesh;
  } else {
    part.size = geom.size;
  }
  part.rotation = pose.GetRotation();
  part.translation = pose.GetTranslation();
  return part;
}

CollisionRecipe MakeCollisionRecipe(
    std::vector<SourcePart> parts,
    CollisionRecipeOptions const& options,
    std::string const& exportPath) {
  // The edge size the importer has always remeshed collision models with.
  constexpr double kRemeshEdgeSize = 0.05;
  bool const merge = parts.size() > 1 || options.merge == CollisionMerge::ShrinkWrap;
  bool const loneMesh = parts.size() == 1 && parts.front().shape == SourcePartShape::Mesh;
  CollisionRecipe result;
  ModifierStack& recipe = result.modifiers;
  switch (options.source) {
    case CollisionRecipeSource::RenderModel:
      recipe.push_back(MakeModifier("Source from Render Model", "From Model Viewer", {}));
      break;
    case CollisionRecipeSource::CadModel: {
      MOCHI_ASSERT(loneMesh);
      recipe.push_back(MakeModifier("Source from CAD Model", "From Model Viewer", {}));
      RenderPlacement const placement = ToRenderPlacement(parts.front());
      result.editorState.cadScale = placement.scale;
      result.editorState.cadRotation = placement.rotation;
      result.editorState.cadTranslation = placement.translation;
      break;
    }
    case CollisionRecipeSource::Files: {
      std::unique_ptr<MeshProcessingModifier> fileSource =
          loneMesh ? FileSource(std::string(parts.front().path)) : nullptr;
      if (!fileSource) {
        recipe.push_back(MakePartsSourceModifier(std::move(parts)));
        break;
      }
      std::unique_ptr<MeshProcessingModifier> placement = Placement(parts.front());
      recipe.push_back(std::move(fileSource));
      if (placement) {
        recipe.push_back(std::move(placement));
      }
      break;
    }
  }
  if (merge) {
    recipe.push_back(MakeModifier(
        "Merge",
        options.merge == CollisionMerge::ShrinkWrap ? kShrinkWrapMethod : kMergeUnionMethod,
        {}));
  }
  if (options.remesh) {
    recipe.push_back(
        MakeModifier("Wrap Mesh", "Alpha Wrap", {{"edgeSize", picojson::value(kRemeshEdgeSize)}}));
    recipe.push_back(MakeModifier(
        "Remesh",
        "Incremental Isotropic Remesh",
        {{"edgeSize", picojson::value(kRemeshEdgeSize)}}));
  }
  recipe.push_back(MakeModifier(
      "Export Mochi Model",
      "SDF",
      {{"bakeSdf", picojson::value(options.bakeSdf)},
       {"exportPath", picojson::value(exportPath)}}));
  return result;
}

bool WriteCollisionRecipe(
    CollisionRecipe const& recipe,
    std::filesystem::path const& pipelinePath,
    bool relativePaths,
    mochi::Error& error) {
  std::string const text = SerializeProcessingPipeline(
      recipe.modifiers,
      recipe.editorState,
      {},
      relativePaths ? pipelinePath : std::filesystem::path{});
  return WriteProcessingPipelineFile(pipelinePath.string(), text, error);
}

} // namespace superdex::studio
