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

#include "meshing/processing_modifiers/source_parts.h"

#include "meshing/link_meshes.h" // GeomCollisionMesh
#include "meshing/processing_modifiers/processing_mesh_utils.h" // WeldIdenticalVertices

#include <imgui.h>
#include <imguios/fonts/icons_font_awesome5.h>

#include <superdex_robotics/utils/file_utils.h> // GeomReference

#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_renderer/render_space.h>

#include <array>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace superdex::studio {

namespace {

using superdex::robotics::GeomReference;
using superdex::robotics::GeomType;

struct SourcePartsProps {
  mochi::DynamicArray<SourcePart> parts;

  MOCHI_STRUCT_BEGIN(superdex::studio::SourcePartsProps)
  MOCHI_FIELD(parts)
  MOCHI_STRUCT_END()
};

constexpr std::array<char const*, 6> kShapeNames =
    {"Mesh", "Box", "Sphere", "Cylinder", "Capsule", "Ellipsoid"};

GeomType ToGeomType(SourcePartShape shape) {
  switch (shape) {
    case SourcePartShape::Mesh:
      return GeomType::Mesh;
    case SourcePartShape::Box:
      return GeomType::Box;
    case SourcePartShape::Sphere:
      return GeomType::Sphere;
    case SourcePartShape::Cylinder:
      return GeomType::Cylinder;
    case SourcePartShape::Capsule:
      return GeomType::Capsule;
    case SourcePartShape::Ellipsoid:
      break;
  }
  return GeomType::Ellipsoid;
}

std::string PartLabel(SourcePart const& part) {
  if (part.shape == SourcePartShape::Mesh) {
    std::string const name = std::filesystem::path(std::string(part.path)).filename().string();
    return name.empty() ? "<no file>" : name;
  }
  return kShapeNames[static_cast<std::size_t>(part.shape)];
}

class PartsSourceMethod : public ReflectedMethod<SourcePartsProps> {
 public:
  explicit PartsSourceMethod(std::vector<SourcePart> parts = {}) {
    _props.parts.assign(parts.begin(), parts.end());
  }

  char const* Name() const override {
    return "Parts";
  }
  char const* Description() const override {
    return "Assembles mesh files and primitives, each placed in the model's frame.";
  }
  bool CanGenerate(ModifierRunContext const& /*ctx*/) const override {
    return !_props.parts.empty();
  }
  // The first mesh part's file, which an Auto export is named after.
  std::string SourceFilePath(ModifierRunContext const& /*ctx*/) const override {
    for (SourcePart const& part : _props.parts) {
      if (part.shape == SourcePartShape::Mesh && !part.path.empty()) {
        return std::string(part.path);
      }
    }
    return {};
  }
  std::vector<std::string_view> PathPropKeys() const override {
    return {"parts[].path"};
  }

  void ShowParams(ModifierGuiContext const& gui) override {
    ModifierTooltip const& tooltip = gui.tooltip;
    std::optional<std::size_t> removed;
    for (std::size_t i = 0; i < _props.parts.size(); ++i) {
      SourcePart& part = _props.parts[i];
      ImGui::PushID(static_cast<int>(i));
      std::string const header = "Part " + std::to_string(i + 1) + ": " + PartLabel(part);
      bool const open = ImGui::TreeNodeEx(
          "##part",
          ImGuiTreeNodeFlags_AllowOverlap | ImGuiTreeNodeFlags_FramePadding,
          "%s",
          header.c_str());
      float const trashWidth =
          ImGui::CalcTextSize(ICON_FA_TRASH).x + 2.0f * ImGui::GetStyle().FramePadding.x;
      // Placed like the modifier header's remove button, so the trash icons line up.
      ImGui::SameLine();
      ImGui::SetCursorPosX(ImGui::GetContentRegionMax().x - trashWidth);
      if (ImGui::Button(ICON_FA_TRASH)) {
        removed = i;
      }
      tooltip("Remove this part.");
      if (open) {
        int shape = static_cast<int>(part.shape);
        if (ImGui::Combo(
                "Shape", &shape, kShapeNames.data(), static_cast<int>(kShapeNames.size()))) {
          part.shape = static_cast<SourcePartShape>(shape);
        }
        if (gui.host) {
          if (part.shape == SourcePartShape::Mesh) {
            gui.host->ModelSlot("##file", part.path, ModelSlotType::AnyModel);
            tooltip(
                "Drop a mesh here: a CAD model (STL; STEP is not supported here), a render model "
                "(OBJ, GLB, DAE), or a Mochi model.");
            gui.host->DragXYZ("Scale", part.scale, 0.01f, "%.4f");
          } else {
            gui.host->DragXYZ("Size", part.size, 0.001f, "%.4f");
            tooltip(
                "Size in meters. Box: half-extents. Sphere: radius. Cylinder and capsule: radius, "
                "half-length along Z. Ellipsoid: radii.");
          }
          gui.host->DragTransformRT("Placement", part.rotation, part.translation);
          tooltip("The part's pose in the model's frame (Z-up, meters).");
        }
        ImGui::TreePop();
      }
      ImGui::PopID();
    }
    if (removed) {
      _props.parts.erase(_props.parts.begin() + static_cast<std::ptrdiff_t>(*removed));
    }
    if (ImGui::Button("Add Part")) {
      _props.parts.push_back(SourcePart{});
    }
    tooltip("Add a mesh file or primitive. A Merge modifier below combines the parts into one.");
  }

  mochi::MeshData Run(
      mochi::MeshData const& /*input*/,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& error) const override {
    MOCHI_ERROR_IF(_props.parts.empty(), error, "Source from Parts: add a part.");
    MOCHI_ERROR_RETURN(error, {});
    mochi::MeshData out;
    out.nodesPerElement = 3;
    for (std::size_t i = 0; i < _props.parts.size(); ++i) {
      SourcePart const& part = _props.parts[i];
      GeomReference geom;
      geom.type = ToGeomType(part.shape);
      geom.size = part.size;
      geom.meshFile = part.path;
      geom.meshScale = part.scale;
      geom.linkFromGeom = mochi::TransformRT{part.rotation, part.translation};
      std::vector<std::string> warnings;
      mochi::MeshData piece = GeomCollisionMesh(geom, /*inLinkFrame=*/true, warnings);
      if (piece.IsEmpty()) {
        MOCHI_LOG_WARNING(
            "Source from Parts: part %zu (%s) has no geometry%s%s",
            i + 1,
            PartLabel(part).c_str(),
            warnings.empty() ? "." : ": ",
            warnings.empty() ? "" : warnings.front().c_str());
        MOCHI_ERROR_SET(error, "Source from Parts: a part has no geometry (see the log).");
        return {};
      }
      processing::WeldIdenticalVertices(piece);
      int const offset = out.GetNumNodes();
      for (mochi::real const value : piece.coordinates) {
        out.coordinates.push_back(value);
      }
      for (int const index : piece.connectivity) {
        out.connectivity.push_back(index + offset);
      }
    }
    mochi::CoordinateSpaceConverter const toRender(
        mochi::CoordinateSpace::Default(), mochi_renderer::RenderSpace());
    toRender.TranslationsToOutput(mochi::MakeSpan(out.coordinates), error);
    return out;
  }
};

} // namespace

ModifierRegistryEntry MakePartsSourceEntry() {
  return ModifierRegistryEntry{"Source from Parts", ModifierKind::Source, []() {
                                 std::vector<std::unique_ptr<MeshProcessingMethod>> methods;
                                 methods.push_back(std::make_unique<PartsSourceMethod>());
                                 return methods;
                               }};
}

std::unique_ptr<MeshProcessingModifier> MakePartsSourceModifier(std::vector<SourcePart> parts) {
  std::vector<std::unique_ptr<MeshProcessingMethod>> methods;
  methods.push_back(std::make_unique<PartsSourceMethod>(std::move(parts)));
  return std::make_unique<MeshProcessingModifier>(
      "Source from Parts", ModifierKind::Source, std::move(methods));
}

} // namespace superdex::studio
