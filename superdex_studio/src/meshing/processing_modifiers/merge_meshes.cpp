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

#include "meshing/processing_modifiers/merge_meshes.h"

#include "meshing/processing_modifiers/wrap_mesh.h" // MakeShrinkWrapMethod

#include <imgui.h>

#include <mochi_core/utils/log.h>
#include <mochi_core/utils/span.h>
#include <mochi_mesh/mochi_mesh_cli_types.h> // MeshUnionParams
#include <mochi_mesh/step_mesh_stages.h> // UnionMeshes

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace superdex::studio {

namespace {

// The Wrap Detail field's range: finer wraps take minutes, coarser ones lose the pieces' shape.
constexpr double kMinWrapDetail = 0.0005;
constexpr double kMaxWrapDetail = 0.5;

struct MergeUnionProps {
  bool wrapFallback = mochi::mesh::MeshUnionParams{}.wrapFallback;
  double wrapDetail = mochi::mesh::MeshUnionParams{}.wrapDetail;

  MOCHI_STRUCT_BEGIN(superdex::studio::MergeUnionProps)
  MOCHI_FIELD(wrapFallback)
  MOCHI_FIELD(wrapDetail)
  MOCHI_STRUCT_END()
};

class UnionMethod : public ReflectedMethod<MergeUnionProps> {
 public:
  [[nodiscard]] char const* Name() const override {
    return kMergeUnionMethod;
  }
  [[nodiscard]] char const* Description() const override {
    return "One watertight surface, the exact boolean union of the input's pieces. Pieces are "
           "repaired first; one that cannot be is wrapped.";
  }
  void ShowParams(ModifierGuiContext const& gui) override {
    ModifierTooltip const& tooltip = gui.tooltip;
    ImGui::Checkbox("Wrap If Union Fails", &_props.wrapFallback);
    tooltip(
        "If the exact union fails, wrap all the pieces in one closed surface instead of failing. "
        "The union can take many minutes when pieces touch exactly; Shrink Wrap is faster.");
    if (ImGui::InputDouble("Wrap Detail", &_props.wrapDetail, 0.0, 0.0, "%.4f")) {
      _props.wrapDetail = std::clamp(_props.wrapDetail, kMinWrapDetail, kMaxWrapDetail);
    }
    tooltip(
        "The narrowest concavity a wrap keeps, as a fraction of the wrapped mesh's bounding-box "
        "diagonal. Applies to pieces that cannot be repaired and to the fallback wrap. Smaller "
        "keeps finer detail but takes longer.");
  }
  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& error) const override {
    std::vector<mochi::MeshData> const pieces = SplitConnectedPieces(input);
    std::vector<mochi::MeshDataView> const views(pieces.begin(), pieces.end());
    mochi::mesh::MeshUnionParams params;
    params.wrapFallback = _props.wrapFallback;
    params.wrapDetail = _props.wrapDetail;
    mochi::mesh::MeshUnionResult result =
        mochi::mesh::UnionMeshes(mochi::MakeConstSpan(views), params, error);
    if (result.usedFallback) {
      MOCHI_LOG_WARNING(
          "Merge: part of the union of %zu pieces could not be computed exactly and was wrapped, "
          "so it is approximate.",
          pieces.size());
    }
    return std::move(result.mesh);
  }
};

} // namespace

std::vector<mochi::MeshData> SplitConnectedPieces(mochi::MeshData const& mesh) {
  int const numNodes = mesh.GetNumNodes();
  std::vector<int> parent(static_cast<std::size_t>(numNodes));
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&parent](int v) {
    while (parent[static_cast<std::size_t>(v)] != v) {
      parent[static_cast<std::size_t>(v)] =
          parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(v)])];
      v = parent[static_cast<std::size_t>(v)];
    }
    return v;
  };
  int const perElement = mesh.nodesPerElement;
  std::size_t const numElements =
      perElement > 0 ? mesh.connectivity.size() / static_cast<std::size_t>(perElement) : 0;
  for (std::size_t e = 0; e < numElements; ++e) {
    int const first = find(mesh.connectivity[e * perElement]);
    for (int k = 1; k < perElement; ++k) {
      parent[static_cast<std::size_t>(find(mesh.connectivity[e * perElement + k]))] = first;
    }
  }

  std::vector<int> pieceOfRoot(static_cast<std::size_t>(numNodes), -1);
  std::vector<int> localIndex(static_cast<std::size_t>(numNodes), -1);
  std::vector<mochi::MeshData> pieces;
  for (std::size_t e = 0; e < numElements; ++e) {
    int const root = find(mesh.connectivity[e * perElement]);
    int& piece = pieceOfRoot[static_cast<std::size_t>(root)];
    if (piece < 0) {
      piece = static_cast<int>(pieces.size());
      pieces.emplace_back().nodesPerElement = perElement;
    }
    mochi::MeshData& out = pieces[static_cast<std::size_t>(piece)];
    for (int k = 0; k < perElement; ++k) {
      int const v = mesh.connectivity[e * perElement + k];
      int& local = localIndex[static_cast<std::size_t>(v)];
      if (local < 0) {
        local = out.GetNumNodes();
        for (int axis = 0; axis < 3; ++axis) {
          out.coordinates.push_back(mesh.coordinates[3 * v + axis]);
        }
      }
      out.connectivity.push_back(local);
    }
  }
  return pieces;
}

ModifierRegistryEntry MakeMergeMeshesEntry() {
  return ModifierRegistryEntry{"Merge", ModifierKind::Transform, []() {
                                 std::vector<std::unique_ptr<MeshProcessingMethod>> methods;
                                 methods.push_back(std::make_unique<UnionMethod>());
                                 methods.push_back(MakeShrinkWrapMethod());
                                 return methods;
                               }};
}

} // namespace superdex::studio
