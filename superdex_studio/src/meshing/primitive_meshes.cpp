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

#include "meshing/primitive_meshes.h"

#include <mochi_core/utils/constants.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace superdex::studio {

using superdex::robotics::MjcfGeomType;

namespace {

// A point on a surface of revolution about Z: distance from the axis and height.
struct ProfilePoint {
  mochi::real radius;
  mochi::real z;
};

// Revolves a profile running from a bottom pole (radius 0) to a top pole (radius 0) about Z.
// Interior profile points become rings of `segments` vertices; the poles are single vertices.
mochi::MeshData Revolve(std::vector<ProfilePoint> const& profile, int segments) {
  mochi::MeshData mesh;
  mesh.nodesPerElement = 3;
  int const numRings = static_cast<int>(profile.size()) - 2;
  auto addVertex = [&](mochi::real x, mochi::real y, mochi::real z) {
    mesh.coordinates.push_back(x);
    mesh.coordinates.push_back(y);
    mesh.coordinates.push_back(z);
  };
  addVertex(0, 0, profile.front().z);
  for (int ring = 0; ring < numRings; ++ring) {
    ProfilePoint const& p = profile[ring + 1];
    for (int s = 0; s < segments; ++s) {
      mochi::real const angle = 2 * mochi::kPI * static_cast<mochi::real>(s) / segments;
      addVertex(p.radius * std::cos(angle), p.radius * std::sin(angle), p.z);
    }
  }
  addVertex(0, 0, profile.back().z);
  int const top = 1 + numRings * segments;
  auto ringVertex = [&](int ring, int s) { return 1 + ring * segments + (s % segments); };
  auto addTriangle = [&](int a, int b, int c) {
    mesh.connectivity.push_back(a);
    mesh.connectivity.push_back(b);
    mesh.connectivity.push_back(c);
  };
  // Triangles wind counter-clockwise seen from outside: around +theta, then up the profile.
  for (int s = 0; s < segments; ++s) {
    addTriangle(0, ringVertex(0, s + 1), ringVertex(0, s));
    for (int ring = 0; ring + 1 < numRings; ++ring) {
      addTriangle(ringVertex(ring, s), ringVertex(ring, s + 1), ringVertex(ring + 1, s + 1));
      addTriangle(ringVertex(ring, s), ringVertex(ring + 1, s + 1), ringVertex(ring + 1, s));
    }
    addTriangle(ringVertex(numRings - 1, s), ringVertex(numRings - 1, s + 1), top);
  }
  return mesh;
}

// Appends half a sphere's profile, from its pole to its equator (bottom) or from its equator to its
// pole (top), offset along Z.
void AppendHemisphere(
    std::vector<ProfilePoint>& profile,
    mochi::real radius,
    mochi::real zOffset,
    bool bottom,
    int rings) {
  for (int i = 0; i <= rings; ++i) {
    mochi::real const t = static_cast<mochi::real>(i) / rings;
    mochi::real const phi = bottom ? (-mochi::kPI / 2 + t * mochi::kPI / 2) : (t * mochi::kPI / 2);
    profile.push_back({radius * std::cos(phi), zOffset + radius * std::sin(phi)});
  }
}

mochi::MeshData Box(mochi::Real3 const& h) {
  mochi::MeshData mesh;
  mesh.nodesPerElement = 3;
  for (int i = 0; i < 8; ++i) {
    mesh.coordinates.push_back((i & 1) ? h[0] : -h[0]);
    mesh.coordinates.push_back((i & 2) ? h[1] : -h[1]);
    mesh.coordinates.push_back((i & 4) ? h[2] : -h[2]);
  }
  // Vertex i has x from bit 0, y from bit 1, z from bit 2. Two outward triangles per face.
  mesh.connectivity = {
      0, 2, 3, 0, 3, 1, // -Z
      4, 5, 7, 4, 7, 6, // +Z
      0, 1, 5, 0, 5, 4, // -Y
      2, 6, 7, 2, 7, 3, // +Y
      0, 4, 6, 0, 6, 2, // -X
      1, 3, 7, 1, 7, 5, // +X
  };
  return mesh;
}

} // namespace

mochi::MeshData TessellatePrimitive(MjcfGeomType type, mochi::Real3 const& size, int segments) {
  int const rings = std::max(segments / 4, 2);
  std::vector<ProfilePoint> profile;
  switch (type) {
    case MjcfGeomType::Box:
      return Box(size);
    case MjcfGeomType::Sphere:
    case MjcfGeomType::Ellipsoid: {
      bool const isSphere = type == MjcfGeomType::Sphere;
      AppendHemisphere(profile, 1, 0, /*bottom=*/true, rings);
      profile.pop_back(); // the equator is shared with the top half
      AppendHemisphere(profile, 1, 0, /*bottom=*/false, rings);
      mochi::MeshData mesh = Revolve(profile, segments);
      mochi::Real3 const radii = isSphere ? mochi::Real3{size[0], size[0], size[0]} : size;
      for (size_t i = 0; i < mesh.coordinates.size(); ++i) {
        mesh.coordinates[i] *= radii[i % 3];
      }
      return mesh;
    }
    case MjcfGeomType::Capsule:
      AppendHemisphere(profile, size[0], -size[1], /*bottom=*/true, rings);
      AppendHemisphere(profile, size[0], size[1], /*bottom=*/false, rings);
      return Revolve(profile, segments);
    case MjcfGeomType::Cylinder:
      profile = {{0, -size[1]}, {size[0], -size[1]}, {size[0], size[1]}, {0, size[1]}};
      return Revolve(profile, segments);
    case MjcfGeomType::Plane:
    case MjcfGeomType::Hfield:
    case MjcfGeomType::Mesh:
    case MjcfGeomType::Sdf:
      break;
  }
  return {};
}

} // namespace superdex::studio
