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

// MJCF geometry layout (ExpandMjcfGeomLinks) and self-contact rules (ApplyMjcfContactFilter).

#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/quaternion_utils.h>
#include <mochi_physics/mochi_physics.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <set>
#include <string>
#include <string_view>
#include <tuple>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

namespace fs = std::filesystem;

constexpr real kTol = 1e-5_r;

// A closed unit cube centered at the origin, outward-facing triangles.
constexpr std::string_view kCubeObj = R"(v -0.5 -0.5 -0.5
v 0.5 -0.5 -0.5
v 0.5 0.5 -0.5
v -0.5 0.5 -0.5
v -0.5 -0.5 0.5
v 0.5 -0.5 0.5
v 0.5 0.5 0.5
v -0.5 0.5 0.5
f 1 3 2
f 1 4 3
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
)";

class MjcfLayoutTest : public testing::Test {
 protected:
  void SetUp() override {
    _prevLogFn = GetLogCallback();
    SetLogCallback([](LogChannel, char const*, char const*, int) {});
    fs::create_directories(_dir.Path() / "meshes");
    mochi::WriteFile(_dir.Path() / "meshes" / "cube.obj", kCubeObj, ExpectOK{});
  }

  void TearDown() override {
    SetLogCallback(_prevLogFn);
  }

  fs::path WriteModel(std::string_view xml) {
    fs::path const path = _dir.Path() / "model.xml";
    mochi::WriteFile(path, xml, ExpectOK{});
    return path;
  }

  static int LinkIndex(BotPrefab const& prefab, std::string_view name) {
    for (int i = 0; i < isize(prefab.links); ++i) {
      if (std::string_view(prefab.links[i].name) == name) {
        return i;
      }
    }
    return kIndexNone;
  }

  // Gives every body's link a placeholder shape so the contact filter treats it as colliding.
  static void GiveEveryBodyAShape(BotPrefab& prefab, MjcfImportInfo const& info) {
    for (auto const& body : info.bodies) {
      prefab.links[LinkIndex(prefab, std::string_view(body.link))].shapeFile = "placeholder.obj";
    }
  }

  using Override = std::tuple<std::string, std::string, bool>;
  static std::set<Override> Overrides(BotPrefab const& prefab) {
    std::set<Override> out;
    for (auto const& o : prefab.contactOverrides) {
      std::string a(o.linkA);
      std::string b(o.linkB);
      if (b < a) {
        std::swap(a, b);
      }
      out.emplace(a, b, o.enable);
    }
    return out;
  }

  LogFn _prevLogFn{};
  mochi::TempDirCleanup _dir = mochi::CreateTempDirectory("mjcf_layout_test", ExpectOK{});
};

constexpr std::string_view kSplitModel = R"(
<mujoco>
  <compiler meshdir="meshes"/>
  <asset>
    <mesh name="cube" file="cube.obj" scale="0.1 0.1 0.1"/>
    <mesh name="missing" file="missing.obj"/>
  </asset>
  <default>
    <default class="visual"><geom type="mesh" contype="0" conaffinity="0" group="2"/></default>
    <default class="collision"><geom group="3"/></default>
  </default>
  <worldbody>
    <body name="base">
      <body name="arm" pos="0 0 0.5">
        <joint name="shoulder"/>
        <inertial pos="0 0 0" mass="2" diaginertia="0.01 0.01 0.01"/>
        <geom class="visual" mesh="cube"/>
        <geom class="visual" mesh="cube" pos="0 0 0.1"/>
        <geom class="collision" type="mesh" mesh="cube"/>
        <geom class="collision" type="mesh" mesh="cube" pos="0 0 0.2"/>
        <geom class="collision" type="box" size="0.05 0.05 0.05"/>
        <geom class="visual" mesh="missing"/>
      </body>
      <body name="cable">
        <joint name="cable_joint"/>
        <inertial pos="0 0 0.1" mass="0.5" diaginertia="0.001 0.001 0.001"/>
      </body>
    </body>
  </worldbody>
</mujoco>)";

TEST_F(MjcfLayoutTest, ExtraGeomsBecomeHardJointedChildLinks) {
  BotPrefab const prefab = LoadBotPrefabFromMjcfFile(WriteModel(kSplitModel).string(), ExpectOK{});
  std::string const cube = (_dir.Path() / "meshes" / "cube.obj").lexically_normal().string();

  int const arm = LinkIndex(prefab, "arm");
  ASSERT_NE(arm, kIndexNone);
  // The body's link keeps its inertia and takes the first collision and visual geoms.
  EXPECT_EQ(std::string_view(prefab.links[arm].shapeFile), cube);
  EXPECT_EQ(std::string_view(prefab.links[arm].renderModelFile), cube);
  EXPECT_NEAR(prefab.links[arm].mass.value_or(0_r), 2_r, kTol);
  EXPECT_EQ(prefab.links[arm].shapeScale, (Real3{0.1_r, 0.1_r, 0.1_r}));

  // The second collision mesh hangs off a Hard joint with a negligible mass.
  int const collision = LinkIndex(prefab, "arm_collision_1");
  ASSERT_NE(collision, kIndexNone);
  EXPECT_EQ(prefab.links[collision].parentLink, arm);
  EXPECT_EQ(prefab.joints[collision].type, ArticulatedJointType::Hard);
  EXPECT_EQ(std::string_view(prefab.links[collision].shapeFile), cube);
  EXPECT_TRUE(NearEqual(prefab.links[collision].shapeTranslation, Real3{0_r, 0_r, 0.2_r}, kTol));
  EXPECT_NEAR(prefab.links[collision].mass.value_or(0_r), 2e-6_r, 1e-9_r);

  // The second visual mesh is a shapeless child; the box and the missing mesh are not assigned.
  int const visual = LinkIndex(prefab, "arm_visual_1");
  ASSERT_NE(visual, kIndexNone);
  EXPECT_TRUE(prefab.links[visual].shapeFile.empty());
  EXPECT_TRUE(NearEqual(prefab.links[visual].renderModelTranslation, Real3{0_r, 0_r, 0.1_r}, kTol));
  EXPECT_EQ(LinkIndex(prefab, "arm_collision_2"), kIndexNone);
  EXPECT_EQ(LinkIndex(prefab, "arm_visual_2"), kIndexNone);

  // A body with inertia but no geometry is a shapeless link that keeps its mass.
  int const cable = LinkIndex(prefab, "cable");
  EXPECT_TRUE(prefab.links[cable].shapeFile.empty());
  EXPECT_NEAR(prefab.links[cable].mass.value_or(0_r), 0.5_r, kTol);
  Validate(prefab, nullptr, ExpectOK{});
}

TEST_F(MjcfLayoutTest, SplitBotInstantiatesWithItsMass) {
  BotPrefab const prefab = LoadBotPrefabFromMjcfFile(WriteModel(kSplitModel).string(), ExpectOK{});
  Context* context = CreateContext(0);
  Scene* scene = context->CreateScene("MjcfLayoutTest");
  RoboticsContext* botsContext = CreateRoboticsContext();
  Bot* bot = CreateBot(scene, prefab, botsContext, ExpectOK{});
  ASSERT_NE(bot, nullptr);
  for (int i = 0; i < 10; ++i) {
    scene->Step(0.01);
  }
  // Arm (2) + its collision child (2e-6) + cable (0.5); base and the visual child are massless.
  EXPECT_NEAR(bot->GetArticulatedActor()->GetMass(ExpectOK{}), 2.5_r, 1e-4_r);
  DestroyBot(scene, bot);
  DestroyRoboticsContext(botsContext);
  DestroyContext(context);
}

TEST_F(MjcfLayoutTest, ImportInfoListsEveryGeom) {
  MjcfImportInfo info;
  BotPrefab const prefab =
      LoadBotPrefabFromMjcfFile(WriteModel(kSplitModel).string(), info, ExpectOK{});
  // The info overload assigns nothing; Studio converts geoms itself.
  for (auto const& link : prefab.links) {
    EXPECT_TRUE(link.shapeFile.empty());
    EXPECT_TRUE(link.renderModelFile.empty());
  }
  MjcfBodyReference const& arm = info.bodies[1];
  ASSERT_EQ(std::string_view(arm.name), "arm");
  ASSERT_EQ(arm.geoms.size(), 6u);
  EXPECT_FALSE(arm.geoms[0].isCollision);
  EXPECT_TRUE(arm.geoms[0].isVisual);
  EXPECT_TRUE(arm.geoms[2].isCollision);
  EXPECT_FALSE(arm.geoms[2].isVisual); // group 3 is hidden by default
  EXPECT_EQ(arm.geoms[4].type, MjcfGeomType::Box);
  EXPECT_EQ(arm.geoms[4].size, (Real3{0.05_r, 0.05_r, 0.05_r}));
  EXPECT_TRUE(arm.geoms[4].shapeFile.empty());
  // A missing mesh keeps its reference but has no resolved file.
  EXPECT_EQ(std::string_view(arm.geoms[5].meshRef), "missing.obj");
  EXPECT_TRUE(arm.geoms[5].meshFile.empty());
}

TEST_F(MjcfLayoutTest, MeshScaleAndReferenceFrameBakeIntoTheShape) {
  // refquat turns the mesh a quarter turn about Z and refpos shifts it; MuJoCo maps vertices to
  // conj(refquat) * (scale * v - refpos) before applying the geom pose.
  BotPrefab const prefab = LoadBotPrefabFromMjcfFile(
      WriteModel(R"(
<mujoco>
  <compiler meshdir="meshes"/>
  <asset><mesh name="cube" file="cube.obj" scale="2 2 2" refpos="0 0 1"
                refquat="0.70710678 0 0 0.70710678"/></asset>
  <worldbody>
    <body name="base"><geom type="mesh" mesh="cube" pos="1 0 0"/></body>
  </worldbody>
</mujoco>)")
          .string(),
      ExpectOK{});
  BotLinkPrefab const& base = prefab.links[0];
  Quaternion const expectedRotation = Conjugate(Quaternion::RotationZ(kPI / 2_r));
  Real3 const expectedTranslation = Real3{1_r, 0_r, 0_r} - expectedRotation * Real3{0_r, 0_r, 1_r};
  EXPECT_EQ(base.shapeScale, (Real3{2_r, 2_r, 2_r}));
  EXPECT_TRUE(NearEqual(base.shapeRotation, expectedRotation, kTol));
  EXPECT_TRUE(NearEqual(base.shapeTranslation, expectedTranslation, kTol));
}

// ---------------------------------------------------------------------------
// Contact rules
// ---------------------------------------------------------------------------

TEST_F(MjcfLayoutTest, ContactFilterTreatsWeldedBodiesAsTheirParent) {
  // B has no joint, so MuJoCo welds it to A, and the jointless base is welded to the world. A weld
  // never collides with its parent weld, except the world: base collides with A, B and C, while
  // A, B and C never collide with each other. Mochi disables the adjacent base-A pair, so the
  // filter enables it.
  MjcfImportInfo info;
  BotPrefab prefab = LoadBotPrefabFromMjcfFile(
      WriteModel(R"(
<mujoco>
  <worldbody>
    <body name="base"><geom size="0.1"/>
      <body name="a"><joint name="ja"/><geom size="0.1"/>
        <body name="b"><geom size="0.1"/>
          <body name="c"><joint name="jc"/><geom size="0.1"/></body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>)")
          .string(),
      info,
      ExpectOK{});
  GiveEveryBodyAShape(prefab, info);
  ApplyMjcfContactFilter(prefab, info);
  std::set<Override> const expected = {{"a", "base", true}};
  EXPECT_EQ(Overrides(prefab), expected);
}

TEST_F(MjcfLayoutTest, ContactFilterKeepsFloatingRootAndChildApart) {
  // A free-floating root is its own weld, so MuJoCo filters it against its jointed child, as
  // Mochi does for adjacent links: no override.
  MjcfImportInfo info;
  BotPrefab prefab = LoadBotPrefabFromMjcfFile(
      WriteModel(R"(
<mujoco>
  <worldbody>
    <body name="pelvis"><freejoint/><geom size="0.1"/>
      <body name="thigh"><joint name="hip"/><geom size="0.1"/></body>
    </body>
  </worldbody>
</mujoco>)")
          .string(),
      info,
      ExpectOK{});
  GiveEveryBodyAShape(prefab, info);
  ApplyMjcfContactFilter(prefab, info);
  EXPECT_TRUE(Overrides(prefab).empty());
}

TEST_F(MjcfLayoutTest, ContactFilterKeepsWorldWeldedBodiesApart) {
  // Both jointless top-level bodies are welded to the world, so MuJoCo never collides them; Mochi
  // already filters them through the shapeless world link, so no override is needed.
  MjcfImportInfo info;
  BotPrefab prefab = LoadBotPrefabFromMjcfFile(
      WriteModel(R"(
<mujoco>
  <worldbody>
    <body name="table"><geom size="0.1"/></body>
    <body name="lamp" pos="0 0 0.15"><geom size="0.1"/></body>
  </worldbody>
</mujoco>)")
          .string(),
      info,
      ExpectOK{});
  GiveEveryBodyAShape(prefab, info);
  ApplyMjcfContactFilter(prefab, info);
  EXPECT_TRUE(Overrides(prefab).empty());
}

TEST_F(MjcfLayoutTest, ContactFilterAppliesExcludePairAndBitmasks) {
  MjcfImportInfo info;
  BotPrefab prefab = LoadBotPrefabFromMjcfFile(
      WriteModel(R"(
<mujoco>
  <worldbody>
    <body name="base"><geom name="g_base" size="0.1"/>
      <body name="a"><joint name="ja"/><geom name="g_a" size="0.1"/>
        <body name="b"><joint name="jb"/><geom size="0.1"/>
          <body name="c"><joint name="jc"/><geom size="0.1" contype="2" conaffinity="2"/>
            <body name="d"><joint name="jd"/><geom size="0.1"/></body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <contact>
    <exclude body1="base" body2="b"/>
    <pair geom1="g_base" geom2="g_a"/>
  </contact>
</mujoco>)")
          .string(),
      info,
      ExpectOK{});
  GiveEveryBodyAShape(prefab, info);
  ApplyMjcfContactFilter(prefab, info);
  std::set<Override> const expected = {
      {"b", "base", false}, // <exclude>
      {"a", "base", true}, // <pair> forces contact between parent and child
      {"a", "c", false}, // bitmasks never match
      {"base", "c", false},
  };
  EXPECT_EQ(Overrides(prefab), expected);
}

} // namespace
