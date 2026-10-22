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

#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/file_utils.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <string_view>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

namespace fs = std::filesystem;

// A link with several visuals and collisions, mixing meshes and primitives, and a second link
// without an <inertial>. "blue" is defined inline by one visual and referenced by another, and
// "decal" is a texture-only material. The capsule and the zero-radius sphere are skipped.
constexpr char const* kUrdf = R"(<?xml version="1.0"?>
<robot name="geoms">
  <material name="orange"><color rgba="1 0.5 0 1"/></material>
  <material name="decal"><texture filename="albedo.png"/></material>
  <link name="base">
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <visual name="shell">
      <origin xyz="0 0 0.1"/>
      <geometry><mesh filename="a.obj"/></geometry>
      <material name="orange"/>
    </visual>
    <visual>
      <geometry><mesh filename="b.obj" scale="2 2 2"/></geometry>
      <material name="blue"><color rgba="0 0 1 1"/></material>
    </visual>
    <collision><geometry><box size="0.2 0.4 0.6"/></geometry></collision>
    <collision><geometry><mesh filename="a.obj"/></geometry></collision>
    <collision>
      <origin xyz="0.1 0 0"/>
      <geometry><mesh filename="b.obj"/></geometry>
    </collision>
  </link>
  <link name="arm">
    <collision><geometry><mesh filename="a.obj"/></geometry></collision>
    <collision><geometry><mesh filename="b.obj"/></geometry></collision>
    <collision><geometry><capsule radius="0.1" length="0.2"/></geometry></collision>
    <visual>
      <geometry><cylinder radius="0.05" length="0.3"/></geometry>
      <material name="blue"/>
    </visual>
    <visual><geometry><sphere radius="0.07"/></geometry></visual>
    <collision><geometry><sphere radius="0"/></geometry></collision>
    <visual>
      <geometry><mesh filename="a.obj"/></geometry>
      <material name="decal"/>
    </visual>
  </link>
  <joint name="j" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1"/>
  </joint>
</robot>
)";

class UrdfGeometryTest : public testing::Test {
 protected:
  void SetUp() override {
    // The loaders only check that mesh files exist.
    for (char const* mesh : {"a.obj", "b.obj"}) {
      mochi::WriteFile(_tempDir / mesh, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", ExpectOK{});
    }
    mochi::WriteFile(_tempDir / "albedo.png", std::string("\x89PNG\r\n\x1a\n", 8), ExpectOK{});
    mochi::WriteFile(_urdfPath, std::string_view(kUrdf), ExpectOK{});
  }

  static BotLinkPrefab const& FindLink(BotPrefab const& prefab, std::string const& name) {
    for (BotLinkPrefab const& link : prefab.links) {
      if (std::string(link.name) == name) {
        return link;
      }
    }
    ADD_FAILURE() << "No link named " << name;
    return prefab.links.front();
  }

  mochi::TempDirCleanup _tempDirCleanup =
      mochi::CreateTempDirectory("urdf_geometry_test", ExpectOK{});
  fs::path _tempDir = _tempDirCleanup.Path();
  fs::path _urdfPath = _tempDir / "robot.urdf";
};

} // namespace

// Extra meshes become Hard-jointed children; primitives are not assigned at runtime.
TEST_F(UrdfGeometryTest, RuntimeSplitsExtraMeshesIntoChildLinks) {
  BotPrefab const prefab = LoadBotPrefabFromUrdfFile(_urdfPath.string(), ExpectOK{});

  ASSERT_EQ(prefab.links.size(), 5u);
  BotLinkPrefab const& base = FindLink(prefab, "base");
  EXPECT_TRUE(std::string(base.shapeFile).ends_with("a.obj"));
  EXPECT_TRUE(std::string(base.renderModelFile).ends_with("a.obj"));
  EXPECT_NEAR_EQ(base.renderModelTranslation, (Real3{0_r, 0_r, 0.1_r}));

  BotLinkPrefab const& baseCollision = FindLink(prefab, "base_collision_1");
  EXPECT_EQ(baseCollision.parentLink, base._index);
  EXPECT_EQ(prefab.joints[baseCollision._index].type, ArticulatedJointType::Hard);
  EXPECT_TRUE(std::string(baseCollision.shapeFile).ends_with("b.obj"));
  EXPECT_NEAR_EQ(baseCollision.shapeTranslation, (Real3{0.1_r, 0_r, 0_r}));
  // The parent's explicit inertia covers the whole link, so the extra piece weighs next to nothing.
  ASSERT_TRUE(baseCollision.mass.has_value());
  EXPECT_LT(*baseCollision.mass, 1e-3_r);

  BotLinkPrefab const& baseVisual = FindLink(prefab, "base_visual_1");
  EXPECT_TRUE(baseVisual.shapeFile.empty());
  EXPECT_TRUE(std::string(baseVisual.renderModelFile).ends_with("b.obj"));
  EXPECT_NEAR_EQ(baseVisual.renderModelScale, (Real3{2_r, 2_r, 2_r}));

  // Without an <inertial>, each piece takes its mass from its own shape.
  BotLinkPrefab const& armCollision = FindLink(prefab, "arm_collision_1");
  EXPECT_FALSE(armCollision.mass.has_value());
  EXPECT_FALSE(armCollision.density.has_value());
  EXPECT_TRUE(std::string(FindLink(prefab, "arm").renderModelFile).ends_with("a.obj"));
}

// The import-info overload reports every supported geometry and assigns none.
TEST_F(UrdfGeometryTest, ImportInfoListsGeometry) {
  UrdfImportInfo info;
  BotPrefab const prefab = LoadBotPrefabFromUrdfFile(_urdfPath.string(), info, ExpectOK{});

  ASSERT_EQ(prefab.links.size(), 2u);
  ASSERT_EQ(info.links.size(), 2u);
  for (size_t i = 0; i < prefab.links.size(); ++i) {
    EXPECT_EQ(std::string(info.links[i].link), std::string(prefab.links[i].name));
    EXPECT_TRUE(prefab.links[i].shapeFile.empty());
    EXPECT_TRUE(prefab.links[i].renderModelFile.empty());
  }

  auto const& base = info.links[0].geoms;
  ASSERT_EQ(base.size(), 5u);
  EXPECT_EQ(std::string(base[0].name), "shell");
  EXPECT_TRUE(base[0].isVisual);
  EXPECT_FALSE(base[0].isCollision);
  EXPECT_TRUE(base[0].hasMaterial);
  EXPECT_TRUE(base[0].keepMeshMaterials);
  EXPECT_NEAR_EQ(base[0].rgba, (Real4{1_r, 0.5_r, 0_r, 1_r}));
  EXPECT_NEAR_EQ(base[0].linkFromGeom.GetTranslation(), (Real3{0_r, 0_r, 0.1_r}));
  EXPECT_FALSE(base[0].meshFile.empty());
  EXPECT_EQ(base[2].type, GeomType::Box);
  EXPECT_TRUE(base[2].isCollision);
  EXPECT_NEAR_EQ(base[2].size, (Real3{0.1_r, 0.2_r, 0.3_r}));

  auto const& arm = info.links[1].geoms;
  ASSERT_EQ(arm.size(), 5u);
  EXPECT_EQ(arm[2].type, GeomType::Cylinder);
  EXPECT_NEAR_EQ(arm[2].size, (Real3{0.05_r, 0.15_r, 0_r}));
  EXPECT_NEAR_EQ(arm[2].rgba, (Real4{0_r, 0_r, 1_r, 1_r}));
  EXPECT_EQ(arm[3].type, GeomType::Sphere);
  EXPECT_NEAR_EQ(arm[3].size, (Real3{0.07_r, 0_r, 0_r}));
  EXPECT_FALSE(arm[3].hasMaterial);
  EXPECT_TRUE(arm[4].hasMaterial);
  EXPECT_NEAR_EQ(arm[4].rgba, (Real4{1_r, 1_r, 1_r, 1_r}));
  EXPECT_TRUE(std::string(arm[4].textureFile).ends_with("albedo.png"));

  ASSERT_EQ(info.warnings.size(), 2u);
  EXPECT_NE(std::string(info.warnings[0]).find("capsule"), std::string::npos);
  EXPECT_NE(std::string(info.warnings[1]).find("no positive size"), std::string::npos);
}
