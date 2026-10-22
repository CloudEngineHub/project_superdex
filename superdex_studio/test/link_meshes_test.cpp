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

// The URDF and MJCF importers' mesh building (meshing/primitive_meshes.h,
// meshing/link_meshes.h). The collision cases spawn the mesh helper (SUPERDEX_MESH_CLI_PATH).

#include "meshing/link_meshes.h"
#include "meshing/primitive_meshes.h"

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_physics/utils/mochi_model_utils.h>
#include <mochi_renderer/render_space.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace mochi;
using namespace superdex::studio;
using superdex::robotics::GeomReference;
using superdex::robotics::GeomType;

namespace {

// Enclosed volume of a closed, outward-oriented triangle mesh (divergence theorem). Negative for an
// inward-facing one.
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

Real3 Center(MeshData const& mesh) {
  Real3 lo{
      std::numeric_limits<real>::max(),
      std::numeric_limits<real>::max(),
      std::numeric_limits<real>::max()};
  Real3 hi = -lo;
  for (size_t i = 0; i < mesh.coordinates.size(); ++i) {
    lo[i % 3] = std::min(lo[i % 3], mesh.coordinates[i]);
    hi[i % 3] = std::max(hi[i % 3], mesh.coordinates[i]);
  }
  return 0.5_r * (lo + hi);
}

GeomReference BoxGeom(Real3 const& halfExtents, Real3 const& position) {
  GeomReference geom;
  geom.type = GeomType::Box;
  geom.size = halfExtents;
  geom.linkFromGeom = TransformRT{position};
  return geom;
}

class LinkMeshesTest : public testing::Test {
 protected:
  void SetUp() override {
    // Remeshing and SDF baking log progress and fallbacks; this test checks results instead.
    _prevLogFn = GetLogCallback();
    SetLogCallback([](LogChannel, char const*, char const*, int) {});
  }
  void TearDown() override {
    SetLogCallback(_prevLogFn);
  }

  LogFn _prevLogFn{};
  TempDirCleanup _dir = CreateTempDirectory("link_meshes_test", test::ExpectOK{});
};

} // namespace

TEST_F(LinkMeshesTest, TessellatedPrimitivesAreClosedAndHaveTheirVolume) {
  real const r = 0.1_r;
  real const h = 0.2_r;
  struct Case {
    GeomType type;
    Real3 size;
    real volume;
  };
  std::vector<Case> const cases = {
      {GeomType::Box, {r, 2 * r, 3 * r}, 8 * r * 2 * r * 3 * r},
      {GeomType::Sphere, {r, 0_r, 0_r}, 4_r / 3_r * kPI * r * r * r},
      {GeomType::Ellipsoid, {r, 2 * r, 3 * r}, 4_r / 3_r * kPI * r * 2 * r * 3 * r},
      {GeomType::Cylinder, {r, h, 0_r}, kPI * r * r * 2 * h},
      {GeomType::Capsule, {r, h, 0_r}, kPI * r * r * 2 * h + 4_r / 3_r * kPI * r * r * r},
  };
  for (Case const& c : cases) {
    MeshData const mesh = TessellatePrimitive(c.type, c.size, 64);
    ASSERT_FALSE(mesh.IsEmpty());
    // Tessellation slightly undershoots curved surfaces; a positive volume means outward faces.
    EXPECT_NEAR(Volume(mesh), c.volume, 0.02_r * c.volume) << static_cast<int>(c.type);
  }
  EXPECT_TRUE(TessellatePrimitive(GeomType::Plane, {1_r, 1_r, 1_r}).IsEmpty());
}

TEST_F(LinkMeshesTest, RenderSectionsTakeTheMjcfColorAndPose) {
  GeomReference geom = BoxGeom({0.1_r, 0.1_r, 0.1_r}, {1_r, 0_r, 0_r});
  geom.hasMaterial = true;
  geom.rgba = {1_r, 0_r, 0_r, 1_r};
  std::vector<std::string> warnings;
  auto const sections = GeomRenderSections(geom, warnings);
  ASSERT_EQ(sections.size(), 1u);
  std::array<float, 4> const red = {1.0f, 0.0f, 0.0f, 1.0f};
  EXPECT_EQ(sections[0].baseColor, red);
  float minX = std::numeric_limits<float>::max();
  for (size_t i = 0; i < sections[0].positions.size(); i += 3) {
    minX = std::min(minX, sections[0].positions[i]);
  }
  EXPECT_NEAR(minX, 0.9f, 1e-5f);
  EXPECT_TRUE(warnings.empty());
}

TEST_F(LinkMeshesTest, FlattenedMeshDropsItsNormals) {
  // A zero mesh scale has no inverse for the normals, so they are left for the GLB writer to
  // recompute rather than turned into NaNs.
  std::string const obj = (_dir.Path() / "tri.obj").string();
  WriteFile(obj, "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1//1 2//1 3//1\n", test::ExpectOK{});
  GeomReference geom;
  geom.type = GeomType::Mesh;
  geom.meshFile = obj;
  geom.meshScale = {1_r, 0_r, 1_r};
  std::vector<std::string> warnings;
  auto const sections = GeomRenderSections(geom, warnings);
  ASSERT_EQ(sections.size(), 1u);
  EXPECT_FALSE(sections[0].hasNormals);
  EXPECT_TRUE(sections[0].normals.empty());
}

TEST_F(LinkMeshesTest, TextureWithoutTexcoordsFallsBackToColor) {
  // A primitive has no texture coordinates, so its texture cannot be applied.
  std::string const png = (_dir.Path() / "albedo.png").string();
  WriteFile(png, std::string("\x89PNG\r\n\x1a\n", 8), test::ExpectOK{});
  GeomReference geom = BoxGeom({0.1_r, 0.1_r, 0.1_r}, {});
  geom.hasMaterial = true;
  geom.textureFile = png;
  std::vector<std::string> warnings;
  auto const sections = GeomRenderSections(geom, warnings);
  ASSERT_EQ(sections.size(), 1u);
  EXPECT_EQ(sections[0].baseColorTexture, nullptr);
  EXPECT_EQ(warnings.size(), 1u);
}

TEST_F(LinkMeshesTest, UrdfColorsOnlyMeshesWithoutMaterials) {
  std::string const plain = (_dir.Path() / "plain.obj").string();
  std::string const painted = (_dir.Path() / "painted.obj").string();
  WriteFile(plain, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", test::ExpectOK{});
  WriteFile(
      painted,
      "mtllib paint.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl Green\nf 1 2 3\n",
      test::ExpectOK{});
  WriteFile(_dir.Path() / "paint.mtl", "newmtl Green\nKd 0 1 0\n", test::ExpectOK{});
  auto colorOf = [](std::string const& file, bool keepMeshMaterials) {
    GeomReference geom;
    geom.type = GeomType::Mesh;
    geom.meshFile = file;
    geom.hasMaterial = true;
    geom.rgba = {1_r, 0_r, 0_r, 1_r};
    geom.keepMeshMaterials = keepMeshMaterials;
    std::vector<std::string> warnings;
    auto const sections = GeomRenderSections(geom, warnings);
    return sections.empty() ? std::array<float, 4>{} : sections.front().baseColor;
  };
  std::array<float, 4> const red = {1.0f, 0.0f, 0.0f, 1.0f};
  std::array<float, 4> const green = {0.0f, 1.0f, 0.0f, 1.0f};
  EXPECT_EQ(colorOf(plain, /*keepMeshMaterials=*/true), red);
  EXPECT_EQ(colorOf(painted, /*keepMeshMaterials=*/true), green);
  EXPECT_EQ(colorOf(painted, /*keepMeshMaterials=*/false), red);
}

TEST_F(LinkMeshesTest, GlbMeshIsReadInMochiSpace) {
  // GLB (like DAE) geometry is in the renderer's space and keeps its own materials.
  mochi_renderer::MeshSection triangle;
  triangle.positions = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f};
  triangle.indices = {0, 1, 2};
  triangle.baseColor = {0.0f, 1.0f, 0.0f, 1.0f};
  std::string const glb = (_dir.Path() / "triangle.glb").string();
  ASSERT_TRUE(mochi_renderer::WriteGlbToFile(glb.c_str(), {triangle}));
  GeomReference geom;
  geom.type = GeomType::Mesh;
  geom.meshFile = glb;
  geom.hasMaterial = true;
  geom.rgba = {1_r, 0_r, 0_r, 1_r};
  geom.keepMeshMaterials = true;
  std::vector<std::string> warnings;

  auto sections = GeomRenderSections(geom, warnings);
  ASSERT_EQ(sections.size(), 1u);
  EXPECT_EQ(sections[0].baseColor, triangle.baseColor);
  MeshData const mesh = GeomCollisionMesh(geom, /*inLinkFrame=*/true, warnings);
  ASSERT_EQ(mesh.GetNumNodes(), 3);
  for (size_t i = 0; i < mesh.coordinates.size(); ++i) {
    EXPECT_NEAR(mesh.coordinates[i], sections[0].positions[i], 1e-6_r);
  }
  mochi_renderer::ConvertMeshSectionsSpace(
      sections,
      CoordinateSpaceConverter(CoordinateSpace::Default(), mochi_renderer::RenderSpace()));
  for (size_t i = 0; i < triangle.positions.size(); ++i) {
    EXPECT_NEAR(sections[0].positions[i], triangle.positions[i], 1e-6f);
  }
  EXPECT_TRUE(warnings.empty());
}

TEST_F(LinkMeshesTest, LinkGlbHoldsEveryGeom) {
  Path const out = Path{(_dir.Path() / "link.glb").string()};
  CoordinateSpaceConverter const identity(CoordinateSpace::Default(), CoordinateSpace::Default());
  std::vector<std::string> warnings;
  ASSERT_TRUE(BuildLinkGlb(
      {BoxGeom({0.1_r, 0.1_r, 0.1_r}, {}), BoxGeom({0.1_r, 0.1_r, 0.1_r}, {1_r, 0_r, 0_r})},
      out,
      identity,
      warnings));
  EXPECT_EQ(mochi_renderer::ReadGlbFromFile(out.ToString().c_str()).size(), 2u);
}

TEST_F(LinkMeshesTest, UnionCollisionMergesOverlappingGeoms) {
  // Two 0.2 m cubes overlapping by half enclose 0.012 m^3.
  Path const out = Path{(_dir.Path() / "link.mochi.h5").string()};
  LinkCollisionOptions options;
  options.remesh = false;
  std::vector<std::string> warnings;
  ASSERT_TRUE(BuildLinkCollisionH5(
      {BoxGeom({0.1_r, 0.1_r, 0.1_r}, {}), BoxGeom({0.1_r, 0.1_r, 0.1_r}, {0.1_r, 0_r, 0_r})},
      options,
      out,
      warnings));
  ModelData const model = model_utils::LoadFromFile(out.ToString(), test::ExpectOK{});
  ASSERT_TRUE(model.mesh.has_value());
  EXPECT_NEAR(Volume(*model.mesh), 0.012_r, 1e-5_r);
  EXPECT_TRUE(model.sdf.has_value());
  EXPECT_TRUE(warnings.empty());
}

TEST_F(LinkMeshesTest, GeomsInertiaWeighsEveryGeomWithMass) {
  // 0.2 m cubes: one at MuJoCo's default density (8 kg), one with a 2 kg mass 1 m along X, and a
  // massless visual one that must not count.
  GeomReference heavy = BoxGeom({0.1_r, 0.1_r, 0.1_r}, {1_r, 0_r, 0_r});
  heavy.mass = 2_r;
  GeomReference visual = BoxGeom({0.1_r, 0.1_r, 0.1_r}, {5_r, 0_r, 0_r});
  visual.density = 0_r;
  std::vector<std::string> warnings;
  std::optional<GeomsInertia> const inertia =
      ComputeGeomsInertia({BoxGeom({0.1_r, 0.1_r, 0.1_r}, {}), heavy, visual}, warnings);
  ASSERT_TRUE(inertia.has_value());
  EXPECT_NEAR(inertia->mass, 10_r, 1e-4_r);
  EXPECT_TRUE(NearEqual(inertia->centerOfMass, Real3{0.2_r, 0_r, 0_r}, 1e-5_r));
  // Each cube contributes m (0.2^2 + 0.2^2) / 12 about its center; Y and Z add the parallel-axis
  // terms 8 * 0.2^2 + 2 * 0.8^2.
  real const own = 10_r * 0.08_r / 12_r;
  Real6 const expected{own, 0_r, 0_r, own + 1.6_r, 0_r, own + 1.6_r};
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(inertia->momentOfInertia[i], expected[i], 1e-4_r) << i;
  }
  EXPECT_TRUE(warnings.empty());
}

TEST_F(LinkMeshesTest, PrimitiveInertiaIsAnalytic) {
  // A sphere weighs exactly (4/3) pi r^3 at the default density, unlike its tessellation.
  GeomReference sphere;
  sphere.type = GeomType::Sphere;
  sphere.size = {0.1_r, 0_r, 0_r};
  std::vector<std::string> warnings;
  std::optional<GeomsInertia> const inertia = ComputeGeomsInertia({sphere}, warnings);
  ASSERT_TRUE(inertia.has_value());
  real const mass = 1000_r * 4_r / 3_r * kPI * 0.001_r;
  EXPECT_NEAR(inertia->mass, mass, 1e-5_r * mass);
  EXPECT_NEAR(inertia->momentOfInertia[0], 0.4_r * mass * 0.01_r, 1e-5_r * mass * 0.01_r);
}

TEST_F(LinkMeshesTest, BoxRendersWithFlatFaces) {
  std::vector<std::string> warnings;
  auto const sections = GeomRenderSections(BoxGeom({0.1_r, 0.2_r, 0.3_r}, {}), warnings);
  ASSERT_EQ(sections.size(), 1u);
  ASSERT_TRUE(sections[0].hasNormals);
  ASSERT_EQ(sections[0].normals.size(), sections[0].positions.size());
  // Every corner's normal is its face's axis, not a blend across the box's edges.
  for (size_t i = 0; i < sections[0].normals.size(); i += 3) {
    float const largest = std::max(
        {std::abs(sections[0].normals[i]),
         std::abs(sections[0].normals[i + 1]),
         std::abs(sections[0].normals[i + 2])});
    EXPECT_NEAR(largest, 1.0f, 1e-5f);
  }
}

// The vendored mujoco_menagerie Franka FR3 v2 (assets/test/mjcf/franka_fr3_v2).
static superdex::robotics::MjcfBodyReference Fr3v2Body(std::string_view name) {
  char const* assets = std::getenv("SUPERDEX_ASSETS_PATH");
  EXPECT_NE(assets, nullptr);
  std::string const path = (std::filesystem::path(assets ? assets : "") / "test" / "mjcf" /
                            "franka_fr3_v2" / "fr3v2.xml")
                               .string();
  superdex::robotics::MjcfImportInfo info;
  (void)superdex::robotics::LoadBotPrefabFromMjcfFile(path, info, test::ExpectOK{});
  for (auto const& body : info.bodies) {
    if (std::string_view(body.name) == name) {
      return body;
    }
  }
  ADD_FAILURE() << "No FR3 v2 body " << name;
  return {};
}

static GeomList GeomsOf(superdex::robotics::MjcfBodyReference const& body, bool collision) {
  GeomList geoms;
  for (GeomReference const& geom : body.geoms) {
    if (collision ? geom.isCollision : geom.isVisual) {
      geoms.push_back(geom);
    }
  }
  return geoms;
}

TEST_F(LinkMeshesTest, Fr3v2BaseGeomInertiaMatchesMuJoCo) {
  // The inertia Studio infers from fr3v2_link0's geoms, as for a body without an <inertial>.
  // MuJoCo 3.16 with inertiafromgeom="true" gives 2.78558 kg at (-0.02990, -0.00016, 0.05783) m.
  superdex::robotics::MjcfBodyReference const body = Fr3v2Body("fr3v2_link0");
  std::vector<std::string> warnings;
  GeomList const geoms(body.geoms.begin(), body.geoms.end());
  std::optional<GeomsInertia> const inertia = ComputeGeomsInertia(geoms, warnings);
  ASSERT_TRUE(inertia.has_value());
  EXPECT_NEAR(inertia->mass, 2.78558_r, 0.02_r * 2.78558_r);
  EXPECT_TRUE(
      NearEqual(inertia->centerOfMass, Real3{-0.0298969_r, -0.00016023_r, 0.0578298_r}, 0.002_r));
}

TEST_F(LinkMeshesTest, Fr3v2BaseModelsBuild) {
  superdex::robotics::MjcfBodyReference const body = Fr3v2Body("fr3v2_link0");
  GeomList const visual = GeomsOf(body, /*collision=*/false);
  ASSERT_FALSE(visual.empty());
  CoordinateSpaceConverter const identity(CoordinateSpace::Default(), CoordinateSpace::Default());
  std::vector<std::string> warnings;
  Path const glb = Path{(_dir.Path() / "fr3v2_link0.glb").string()};
  ASSERT_TRUE(BuildLinkGlb(visual, glb, identity, warnings));
  // Each visual OBJ keeps its own MJCF material.
  EXPECT_GE(mochi_renderer::ReadGlbFromFile(glb.ToString().c_str()).size(), visual.size());

  LinkCollisionOptions options;
  options.remesh = false;
  Path const h5 = Path{(_dir.Path() / "fr3v2_link0.mochi.h5").string()};
  ASSERT_TRUE(BuildLinkCollisionH5(GeomsOf(body, /*collision=*/true), options, h5, warnings));
  ModelData const model = model_utils::LoadFromFile(h5.ToString(), test::ExpectOK{});
  ASSERT_TRUE(model.mesh.has_value());
  EXPECT_GT(Volume(*model.mesh), 0_r);
  EXPECT_TRUE(model.sdf.has_value());
}

TEST_F(LinkMeshesTest, GeomsWithoutMassHaveNoInertia) {
  GeomReference visual = BoxGeom({0.1_r, 0.1_r, 0.1_r}, {});
  visual.density = 0_r;
  std::vector<std::string> warnings;
  EXPECT_FALSE(ComputeGeomsInertia({visual}, warnings).has_value());
}

TEST_F(LinkMeshesTest, PrimitivesFitTheirMeshLikeMuJoCo) {
  // A 0.1 x 0.2 x 0.6 m box mesh centered at (1, 2, 3). Its equivalent-inertia box is itself, with
  // the long side along the fitted Z axis.
  MeshData const boxMesh = TessellatePrimitive(GeomType::Box, {0.05_r, 0.1_r, 0.3_r});
  std::string obj;
  for (size_t i = 0; i < boxMesh.coordinates.size(); i += 3) {
    obj += "v " + std::to_string(boxMesh.coordinates[i] + 1) + " " +
        std::to_string(boxMesh.coordinates[i + 1] + 2) + " " +
        std::to_string(boxMesh.coordinates[i + 2] + 3) + "\n";
  }
  for (size_t t = 0; t < boxMesh.connectivity.size(); t += 3) {
    obj += "f " + std::to_string(boxMesh.connectivity[t] + 1) + " " +
        std::to_string(boxMesh.connectivity[t + 1] + 1) + " " +
        std::to_string(boxMesh.connectivity[t + 2] + 1) + "\n";
  }
  std::string const path = (_dir.Path() / "box.obj").string();
  WriteFile(path, obj, test::ExpectOK{});

  GeomReference box;
  box.type = GeomType::Box;
  box.meshFile = path;
  std::vector<std::string> warnings;
  ASSERT_TRUE(FitPrimitiveToMesh(box, warnings));
  EXPECT_TRUE(NearEqual(box.size, Real3{0.05_r, 0.1_r, 0.3_r}, 1e-4_r));
  EXPECT_TRUE(NearEqual(box.linkFromGeom.GetTranslation(), Real3{1_r, 2_r, 3_r}, 1e-4_r));
  EXPECT_TRUE(box.meshFile.empty());

  // A capsule takes the mean of the short half-extents as radius and loses half of it from the
  // half-length; fitscale scales both.
  GeomReference capsule;
  capsule.type = GeomType::Capsule;
  capsule.meshFile = path;
  capsule.fitScale = 2_r;
  ASSERT_TRUE(FitPrimitiveToMesh(capsule, warnings));
  EXPECT_TRUE(NearEqual(capsule.size, Real3{0.15_r, 0.525_r, 0_r}, 1e-4_r));
  EXPECT_TRUE(warnings.empty());
}

TEST_F(LinkMeshesTest, OnlyTheFirstObjObjectIsKept) {
  std::filesystem::path const source = _dir.Path() / "source";
  std::filesystem::create_directories(source);
  std::string const twoObjects =
      "mtllib parts.mtl\no first\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nusemtl Red\n"
      "f 1/1 2/1 3/1\no second\nv 0 0 1\nf 1/1 2/1 4/1\n";
  std::string const path = (source / "parts.obj").string();
  WriteFile(path, twoObjects, test::ExpectOK{});
  WriteFile(source / "parts.mtl", "newmtl Red\nKd 1 0 0\nmap_Kd tex.png\n", test::ExpectOK{});
  WriteFile(source / "tex.png", std::string("\x89PNG\r\n\x1a\n", 8), test::ExpectOK{});
  std::filesystem::path const copies = _dir.Path() / "copies";
  std::filesystem::create_directories(copies);

  // The copy holds one triangle and still finds its material color and texture.
  std::string const copy = FirstObjObjectFile(path, copies);
  ASSERT_NE(copy, path);
  std::vector<mochi_renderer::MeshSection> const sections =
      mochi_renderer::ReadObjFromFile(copy.c_str());
  ASSERT_EQ(sections.size(), 1u);
  EXPECT_EQ(sections[0].positions.size(), 9u);
  std::array<float, 4> const red = {1.0f, 0.0f, 0.0f, 1.0f};
  EXPECT_EQ(sections[0].baseColor, red);
  EXPECT_NE(sections[0].baseColorTexture, nullptr);

  // An mtllib line without a library name is kept as it is.
  std::string const unnamed = (source / "unnamed.obj").string();
  WriteFile(
      unnamed,
      "mtllib \no a\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\no b\nf 1 3 2\n",
      test::ExpectOK{});
  EXPECT_NE(FirstObjObjectFile(unnamed, copies), unnamed);

  // A single object, even after a group line that precedes every face, is used as is.
  std::string const single = (source / "single.obj").string();
  WriteFile(single, "g all\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", test::ExpectOK{});
  EXPECT_EQ(FirstObjObjectFile(single, copies), single);
}

TEST_F(LinkMeshesTest, ShrinkWrapCollisionWrapsTheGeoms) {
  // Two 0.2 m cubes overlapping by half form a convex 0.3 x 0.2 x 0.2 box, which a shrink-wrap
  // fits.
  Path const out = Path{(_dir.Path() / "link.mochi.h5").string()};
  LinkCollisionOptions options;
  options.merge = CollisionMerge::ShrinkWrap;
  options.remesh = false;
  std::vector<std::string> warnings;
  ASSERT_TRUE(BuildLinkCollisionH5(
      {BoxGeom({0.1_r, 0.1_r, 0.1_r}, {}), BoxGeom({0.1_r, 0.1_r, 0.1_r}, {0.1_r, 0_r, 0_r})},
      options,
      out,
      warnings));
  ModelData const model = model_utils::LoadFromFile(out.ToString(), test::ExpectOK{});
  ASSERT_TRUE(model.mesh.has_value());
  EXPECT_NEAR(Volume(*model.mesh), 0.012_r, 0.0015_r);
  EXPECT_TRUE(model.sdf.has_value());
}

TEST_F(LinkMeshesTest, SeparateGeomStaysInItsOwnFrame) {
  // ExpandMjcfGeomLinks places each Separate geom, so its file is not posed in the link frame.
  Path const out = Path{(_dir.Path() / "geom.mochi.h5").string()};
  LinkCollisionOptions options;
  options.remesh = false;
  options.bakeSdf = false;
  std::vector<std::string> warnings;
  ASSERT_TRUE(BuildGeomCollisionH5(
      BoxGeom({0.1_r, 0.1_r, 0.1_r}, {1_r, 2_r, 3_r}), options, out, warnings));
  ModelData const model = model_utils::LoadFromFile(out.ToString(), test::ExpectOK{});
  ASSERT_TRUE(model.mesh.has_value());
  EXPECT_TRUE(NearEqual(Center(*model.mesh), Real3{}, 1e-6_r));
}
