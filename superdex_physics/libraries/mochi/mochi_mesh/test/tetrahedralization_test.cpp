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

#include "mesh_test_helpers.h"

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_mesh/tetrahedralization.h>

#include <algorithm>
#include <limits>

using namespace mochi;
using namespace mochi::mesh;
using namespace mochi::test;

namespace {

Real3 NodePosition(MeshData const& mesh, int node) {
  return {
      mesh.coordinates[3 * node], mesh.coordinates[3 * node + 1], mesh.coordinates[3 * node + 2]};
}

real TetVolume(MeshData const& mesh, int tet) {
  int const* const nodes = &mesh.connectivity[4 * tet];
  Real3 const origin = NodePosition(mesh, nodes[0]);
  return Dot(NodePosition(mesh, nodes[1]) - origin,
             Cross(NodePosition(mesh, nodes[2]) - origin, NodePosition(mesh, nodes[3]) - origin)) /
      6_r;
}

// Checks every node is used and every tetrahedron is positively oriented; returns the total volume.
real ExpectValidTetMesh(MeshData const& mesh) {
  EXPECT_EQ(mesh.nodesPerElement, 4);
  EXPECT_GT(mesh.GetNumElements(), 0);
  DynamicArray<bool> isReferenced(mesh.GetNumNodes(), false);
  for (int const node : mesh.connectivity) {
    isReferenced[node] = true;
  }
  EXPECT_EQ(std::count(isReferenced.begin(), isReferenced.end(), false), 0);
  real totalVolume = 0_r;
  for (int tet = 0; tet < mesh.GetNumElements(); ++tet) {
    real const volume = TetVolume(mesh, tet);
    EXPECT_GT(volume, 0_r);
    totalVolume += volume;
  }
  return totalVolume;
}

MeshData ScaledCube(real scale) {
  MeshData cube = CreateCubeTriMesh();
  for (real& coordinate : cube.coordinates) {
    coordinate *= scale;
  }
  return cube;
}

// Two overlapping unit cubes: each is closed, but their surfaces intersect.
MeshData OverlappingCubes() {
  MeshData mesh = CreateCubeTriMesh();
  MeshData const shifted = CreateCubeTriMesh();
  int const offset = mesh.GetNumNodes();
  for (int node = 0; node < shifted.GetNumNodes(); ++node) {
    for (real const coordinate : NodePosition(shifted, node) + Real3{0.5_r, 0.5_r, 0.5_r}) {
      mesh.coordinates.push_back(coordinate);
    }
  }
  for (int const node : shifted.connectivity) {
    mesh.connectivity.push_back(node + offset);
  }
  return mesh;
}

MeshData TetrahedralizeCube(real edgeSize) {
  TetrahedralizationParams params;
  params.edgeSize = edgeSize;
  return Tetrahedralize(CreateCubeTriMesh(), params, ExpectOK{});
}

} // namespace

TEST(Tetrahedralization, CubeProducesValidTetMesh) {
  // The unit cube's flat faces and sharp edges are representable exactly.
  EXPECT_NEAR(ExpectValidTetMesh(TetrahedralizeCube(0.5_r)), 1_r, 1e-3_r);
}

TEST(Tetrahedralization, OptimizerAndFeatureOptionsProduceValidMeshes) {
  using Option = void (*)(TetrahedralizationParams&);
  Option const options[] = {
      [](TetrahedralizationParams& p) { p.perturb = false; },
      [](TetrahedralizationParams& p) { p.exude = false; },
      [](TetrahedralizationParams& p) { p.odt = true; },
      [](TetrahedralizationParams& p) { p.detectFeatures = false; },
  };
  for (Option const option : options) {
    TetrahedralizationParams params;
    params.edgeSize = 0.5;
    option(params);
    ExpectValidTetMesh(Tetrahedralize(CreateCubeTriMesh(), params, ExpectOK{}));
  }
}

TEST(Tetrahedralization, SmallerEdgeSizeProducesMoreTets) {
  EXPECT_GT(
      TetrahedralizeCube(0.25_r).GetNumElements(), TetrahedralizeCube(0.5_r).GetNumElements());
}

TEST(Tetrahedralization, ZeroFacetDistanceBoundUsesEdgeSizeFraction) {
  TetrahedralizationParams automatic;
  automatic.edgeSize = 0.5;
  automatic.relativeToMeshSize = false;
  automatic.facetDistanceBound = 0.0;

  TetrahedralizationParams explicitDistanceBound = automatic;
  explicitDistanceBound.facetDistanceBound = automatic.edgeSize / 5.0;

  MeshData const automaticResult = Tetrahedralize(CreateCubeTriMesh(), automatic, ExpectOK{});
  MeshData const explicitResult =
      Tetrahedralize(CreateCubeTriMesh(), explicitDistanceBound, ExpectOK{});
  EXPECT_EQ(automaticResult.coordinates, explicitResult.coordinates);
  EXPECT_EQ(automaticResult.connectivity, explicitResult.connectivity);
}

TEST(Tetrahedralization, RelativeSizesScaleWithTheMesh) {
  int const unitCubeTets = TetrahedralizeCube(0.5_r).GetNumElements();

  TetrahedralizationParams relative;
  relative.edgeSize = 0.5;
  MeshData const relativeResult = Tetrahedralize(ScaledCube(2_r), relative, ExpectOK{});
  EXPECT_NEAR(ExpectValidTetMesh(relativeResult), 8_r, 1e-2_r);
  EXPECT_EQ(relativeResult.GetNumElements(), unitCubeTets);

  TetrahedralizationParams absolute = relative;
  absolute.relativeToMeshSize = false;
  EXPECT_GT(Tetrahedralize(ScaledCube(2_r), absolute, ExpectOK{}).GetNumElements(), unitCubeTets);
}

TEST(Tetrahedralization, RelativeSizesIgnoreUnusedVertices) {
  MeshData const expected = TetrahedralizeCube(0.5_r);

  MeshData cubeWithUnusedVertex = CreateCubeTriMesh();
  cubeWithUnusedVertex.coordinates.push_back(1000_r);
  cubeWithUnusedVertex.coordinates.push_back(0_r);
  cubeWithUnusedVertex.coordinates.push_back(0_r);
  TetrahedralizationParams params;
  params.edgeSize = 0.5;
  MeshData const actual = Tetrahedralize(cubeWithUnusedVertex, params, ExpectOK{});

  EXPECT_EQ(actual.coordinates, expected.coordinates);
  EXPECT_EQ(actual.connectivity, expected.connectivity);
}

TEST(Tetrahedralization, OutputIsDeterministic) {
  MeshData const first = TetrahedralizeCube(0.3_r);
  MeshData const second = TetrahedralizeCube(0.3_r);
  EXPECT_EQ(first.coordinates, second.coordinates);
  EXPECT_EQ(first.connectivity, second.connectivity);
}

TEST(Tetrahedralization, OpenSurfaceIsRejected) {
  (void)Tetrahedralize(CreateCubeTriMeshWithHole(), TetrahedralizationParams{}, ExpectNotOK{});
}

TEST(Tetrahedralization, SelfIntersectingSurfaceIsRejected) {
  (void)Tetrahedralize(OverlappingCubes(), TetrahedralizationParams{}, ExpectNotOK{});
}

// Rejected by the client before it reaches the helper.
TEST(Tetrahedralization, TetInputIsRejected) {
  MeshData const tets = TetrahedralizeCube(0.5_r);
  (void)Tetrahedralize(tets, TetrahedralizationParams{}, ExpectNotOK{});
}

TEST(Tetrahedralization, InvalidParamsAreRejected) {
  using Invalidation = void (*)(TetrahedralizationParams&);
  Invalidation const invalidations[] = {
      [](TetrahedralizationParams& p) { p.edgeSize = 0.0; },
      [](TetrahedralizationParams& p) { p.facetDistanceBound = -1.0; },
      [](TetrahedralizationParams& p) { p.facetAngleBound = 31.0; },
      [](TetrahedralizationParams& p) { p.cellRadiusEdgeRatio = 1.5; },
      [](TetrahedralizationParams& p) { p.sharpFeatureAngle = 0.0; },
  };
  for (Invalidation const invalidate : invalidations) {
    TetrahedralizationParams params;
    invalidate(params);
    (void)Tetrahedralize(CreateCubeTriMesh(), params, ExpectNotOK{});
  }
}

TEST(Tetrahedralization, InvalidScaledLengthsAreRejected) {
  using Invalidation = void (*)(TetrahedralizationParams&);
  struct TestCase {
    real cubeScale;
    Invalidation invalidate;
  };
  TestCase const testCases[] = {
      {.cubeScale = 2_r,
       .invalidate =
           [](TetrahedralizationParams& p) { p.edgeSize = std::numeric_limits<double>::max(); }},
      {.cubeScale = 0.5_r,
       .invalidate =
           [](TetrahedralizationParams& p) {
             p.edgeSize = std::numeric_limits<double>::denorm_min();
           }},
      {.cubeScale = 2_r,
       .invalidate =
           [](TetrahedralizationParams& p) {
             p.facetDistanceBound = std::numeric_limits<double>::max();
           }},
      {.cubeScale = 0.5_r,
       .invalidate =
           [](TetrahedralizationParams& p) {
             p.facetDistanceBound = std::numeric_limits<double>::denorm_min();
           }},
  };
  for (TestCase const& testCase : testCases) {
    TetrahedralizationParams params;
    testCase.invalidate(params);
    Error error;
    (void)Tetrahedralize(ScaledCube(testCase.cubeScale), params, error);
    EXPECT_NOT_OK(error);
    EXPECT_STREQ(
        error.GetDescription(), "Scaled tetrahedralization lengths must be positive and finite.");
  }
}
