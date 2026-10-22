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

// UnionMeshes runs in the superdex_mesh_cli helper (SUPERDEX_MESH_CLI_PATH).

#include "mesh_test_helpers.h"

#include <mochi_mesh/step_mesh_stages.h>

#include <mochi_core/test/mochi_test_helpers.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

using namespace mochi;
using namespace mochi::mesh;
using namespace mochi::test;

namespace {

MeshData Translated(MeshData mesh, Real3 const& offset) {
  for (size_t i = 0; i < mesh.coordinates.size(); ++i) {
    mesh.coordinates[i] += offset[i % 3];
  }
  return mesh;
}

// Enclosed volume of a closed, outward-oriented triangle mesh (divergence theorem).
real Volume(MeshData const& mesh) {
  real volume = 0_r;
  for (size_t t = 0; t + 2 < mesh.connectivity.size(); t += 3) {
    std::array<Real3, 3> p;
    for (int k = 0; k < 3; ++k) {
      int const v = mesh.connectivity[t + k];
      p[k] =
          Real3{mesh.coordinates[3 * v], mesh.coordinates[3 * v + 1], mesh.coordinates[3 * v + 2]};
    }
    volume += Dot(p[0], Cross(p[1], p[2])) / 6_r;
  }
  return volume;
}

real MaxX(MeshData const& mesh) {
  real maxX = std::numeric_limits<real>::lowest();
  for (size_t i = 0; i < mesh.coordinates.size(); i += 3) {
    maxX = std::max(maxX, mesh.coordinates[i]);
  }
  return maxX;
}

MeshUnionResult Union(std::vector<MeshData> const& meshes, MeshUnionParams const& params) {
  std::vector<MeshDataView> views(meshes.begin(), meshes.end());
  return UnionMeshes(MakeConstSpan(views), params, ExpectOK{});
}

} // namespace

TEST(MeshUnionTest, OverlappingBoxesMergeIntoOneSolid) {
  // Two unit cubes overlapping by half along X enclose 1.5 units of volume.
  MeshUnionResult const result =
      Union({CreateCubeTriMesh(), Translated(CreateCubeTriMesh(), {0.5_r, 0_r, 0_r})}, {});
  EXPECT_FALSE(result.usedFallback);
  EXPECT_NEAR(Volume(result.mesh), 1.5_r, 1e-4_r);
  EXPECT_NEAR(MaxX(result.mesh), 1.5_r, 1e-6_r);
}

TEST(MeshUnionTest, FaceSharingConvexPiecesMerge) {
  // A convex decomposition's pieces share exactly coplanar faces, and partial unions of them can
  // meet only along edges, which the exact union cannot represent. A unit cube split into 27
  // sub-cubes must still merge (exactly, or through the wrap fallback) without hanging.
  std::vector<MeshData> pieces;
  real const third = 1_r / 3_r;
  for (int i = 0; i < 27; ++i) {
    MeshData piece = CreateCubeTriMesh();
    for (size_t c = 0; c < piece.coordinates.size(); ++c) {
      int const cell = (c % 3 == 0) ? i % 3 : (c % 3 == 1) ? (i / 3) % 3 : i / 9;
      piece.coordinates[c] = (piece.coordinates[c] + static_cast<real>(cell)) * third;
    }
    pieces.push_back(std::move(piece));
  }
  MeshUnionResult const result = Union(pieces, {});
  EXPECT_NEAR(Volume(result.mesh), 1_r, 0.01_r);
}

TEST(MeshUnionTest, PiecesTouchingAtACornerFallBackToWrap) {
  // The first two cubes meet only at a corner, so their exact union splits that vertex into
  // coincident copies, which corefinement does not reliably support as input. The three enclose
  // 3 - 0.25 units of volume.
  MeshUnionResult const result = Union(
      {CreateCubeTriMesh(),
       Translated(CreateCubeTriMesh(), {1_r, 1_r, 1_r}),
       Translated(CreateCubeTriMesh(), {0.5_r, 0.5_r, 1_r})},
      {});
  EXPECT_TRUE(result.usedFallback);
  EXPECT_NEAR(Volume(result.mesh), 2.75_r, 0.1_r);
}

TEST(MeshUnionTest, DisjointBoxesKeepBothVolumes) {
  MeshUnionResult const result =
      Union({CreateCubeTriMesh(), Translated(CreateCubeTriMesh(), {2_r, 0_r, 0_r})}, {});
  EXPECT_FALSE(result.usedFallback);
  EXPECT_NEAR(Volume(result.mesh), 2_r, 1e-4_r);
}

TEST(MeshUnionTest, HoleFillingClosesAnOpenInput) {
  MeshUnionResult const result = Union({CreateCubeTriMeshWithHole()}, {});
  EXPECT_FALSE(result.usedFallback);
  EXPECT_NEAR(Volume(result.mesh), 1_r, 1e-4_r);
}

TEST(MeshUnionTest, NonConvexInputKeepsItsConcavity) {
  // Two unit cubes offset diagonally make a non-convex solid of 1.75 units (its hull holds 2).
  MeshUnionResult const notch =
      Union({CreateCubeTriMesh(), Translated(CreateCubeTriMesh(), {0.5_r, 0.5_r, 0_r})}, {});
  MeshUnionResult const result =
      Union({notch.mesh, Translated(CreateCubeTriMesh(), {3_r, 0_r, 0_r})}, {});
  EXPECT_FALSE(result.usedFallback);
  EXPECT_NEAR(Volume(result.mesh), 2.75_r, 1e-4_r);
}

TEST(MeshUnionTest, SelfIntersectingInputIsWrappedKeepingItsConcavity) {
  // One input holding two diagonally offset cubes intersects itself, so it is wrapped on its own.
  // The wrap keeps the notch: about 1.75 units, where the convex hull holds 2.
  MeshData overlapping = CreateCubeTriMesh();
  MeshData const second = Translated(CreateCubeTriMesh(), {0.5_r, 0.5_r, 0_r});
  int const offset = overlapping.GetNumNodes();
  for (real const coordinate : second.coordinates) {
    overlapping.coordinates.push_back(coordinate);
  }
  for (int const index : second.connectivity) {
    overlapping.connectivity.push_back(index + offset);
  }
  MeshUnionResult const result = Union({overlapping}, {});
  EXPECT_TRUE(result.usedFallback);
  EXPECT_NEAR(Volume(result.mesh), 1.75_r, 0.08_r);
}

TEST(MeshUnionTest, InsideOutShellIsReoriented) {
  // One input holds two disjoint cubes, the second wound inside-out: both must count as solids.
  MeshData shells = CreateCubeTriMesh();
  MeshData const inverted = Translated(CreateCubeTriMesh(), {3_r, 0_r, 0_r});
  int const offset = shells.GetNumNodes();
  for (real const coordinate : inverted.coordinates) {
    shells.coordinates.push_back(coordinate);
  }
  for (size_t t = 0; t + 2 < inverted.connectivity.size(); t += 3) {
    shells.connectivity.push_back(inverted.connectivity[t] + offset);
    shells.connectivity.push_back(inverted.connectivity[t + 2] + offset);
    shells.connectivity.push_back(inverted.connectivity[t + 1] + offset);
  }
  MeshUnionResult const result = Union({shells}, {});
  EXPECT_FALSE(result.usedFallback);
  EXPECT_NEAR(Volume(result.mesh), 2_r, 1e-4_r);
}

TEST(MeshUnionTest, FailedUnionWithoutFallbackIsAnError) {
  // Cubes meeting at a corner defeat the exact union (see
  // PiecesTouchingAtACornerFallBackToWrap).
  MeshUnionParams params;
  params.wrapFallback = false;
  std::vector<MeshData> const meshes = {
      CreateCubeTriMesh(),
      Translated(CreateCubeTriMesh(), {1_r, 1_r, 1_r}),
      Translated(CreateCubeTriMesh(), {0.5_r, 0.5_r, 1_r})};
  std::vector<MeshDataView> views(meshes.begin(), meshes.end());
  (void)UnionMeshes(MakeConstSpan(views), params, ExpectNotOK{});
}

TEST(MeshUnionTest, NonPositiveWrapDetailIsAnError) {
  MeshUnionParams params;
  params.wrapDetail = 0.0;
  std::vector<MeshData> const meshes = {CreateCubeTriMesh()};
  std::vector<MeshDataView> views(meshes.begin(), meshes.end());
  (void)UnionMeshes(MakeConstSpan(views), params, ExpectNotOK{});
}
