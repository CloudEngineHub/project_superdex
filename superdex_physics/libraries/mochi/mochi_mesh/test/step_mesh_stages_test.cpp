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

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_mesh/step_mesh_stages.h>

#include <gtest/gtest.h>

using namespace mochi;
using namespace mochi::mesh;
using namespace mochi::test;

namespace {

// A unit square in three triangles. Its bottom edge has an extra vertex 0.05 from the corner, so
// the corner triangle is a needle: its 0.05 edge is ~20x shorter than its longest.
MeshData SquareWithNeedle() {
  MeshData mesh;
  mesh.nodesPerElement = 3;
  mesh.coordinates = {
      0_r,
      0_r,
      0_r, // 0
      0.05_r,
      0_r,
      0_r, // 1
      1_r,
      0_r,
      0_r, // 2
      1_r,
      1_r,
      0_r, // 3
      0_r,
      1_r,
      0_r, // 4
  };
  mesh.connectivity = {0, 1, 4, 1, 2, 3, 1, 3, 4};
  return mesh;
}

} // namespace

TEST(CleanupMesh, RemovesAlmostDegenerateFacesByDefault) {
  MeshData const cleaned = CleanupMesh(SquareWithNeedle(), MeshCleanupParams{}, ExpectOK{});
  EXPECT_EQ(cleaned.GetNumNodes(), 4);
}

TEST(CleanupMesh, KeepsEveryVertexWithoutAlmostDegenerateRemoval) {
  MeshCleanupParams params;
  params.removeAlmostDegenerateFaces = false;
  MeshData const cleaned = CleanupMesh(SquareWithNeedle(), params, ExpectOK{});
  EXPECT_EQ(cleaned.GetNumNodes(), 5);
  EXPECT_EQ(cleaned.GetNumElements(), 3);
}
