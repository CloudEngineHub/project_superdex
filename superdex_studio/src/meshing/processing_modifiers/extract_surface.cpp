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

#include "meshing/processing_modifiers/extract_surface.h"

#include "meshing/processing_modifiers/processing_mesh_utils.h" // DisplaySurface

#include <imgui.h>

#include <memory>
#include <vector>

namespace superdex::studio {

namespace {

class TetBoundaryMethod : public ReflectedMethod<EmptyMethodProps> {
 public:
  [[nodiscard]] char const* Name() const override {
    return "Boundary";
  }
  [[nodiscard]] char const* Description() const override {
    return "Replace the tetrahedral input with the triangle mesh of its boundary surface.";
  }
  [[nodiscard]] bool AcceptsTetrahedralInput() const override {
    return true;
  }
  void ShowParams(ModifierGuiContext const& /*gui*/) override {
    ImGui::TextUnformatted("The input must be a tetrahedral mesh (e.g. from Tetrahedralize).");
    ImGui::TextUnformatted("The output is its boundary triangles, without interior nodes.");
  }
  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& error) const override {
    MOCHI_ERROR_IF(
        input.nodesPerElement != 4,
        error,
        "Extract Surface: the input is already a triangle surface. Place it after Tetrahedralize.");
    MOCHI_ERROR_RETURN(error, {});
    return processing::DisplaySurface(input);
  }
};

} // namespace

ModifierRegistryEntry MakeExtractSurfaceEntry() {
  return ModifierRegistryEntry{"Extract Surface", ModifierKind::Transform, []() {
                                 std::vector<std::unique_ptr<MeshProcessingMethod>> methods;
                                 methods.push_back(std::make_unique<TetBoundaryMethod>());
                                 return methods;
                               }};
}

} // namespace superdex::studio
