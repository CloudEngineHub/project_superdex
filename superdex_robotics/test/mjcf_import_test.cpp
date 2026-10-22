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

// MJCF parsing and mapping onto BotPrefab: compiler settings, defaults, orientations, joints,
// inertia, root handling, keyframes, names, and equality constraints.

#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/quaternion_utils.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <string_view>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

namespace fs = std::filesystem;

constexpr real kTol = 1e-5_r;

class MjcfImportTest : public testing::Test {
 protected:
  void SetUp() override {
    // The importer logs a warning for every skipped MJCF feature; those are expected here.
    _prevLogFn = GetLogCallback();
    SetLogCallback([](LogChannel, char const*, char const*, int) {});
  }

  void TearDown() override {
    SetLogCallback(_prevLogFn);
  }

  // Writes `xml` (and any extra files) into a temp directory and loads it.
  BotPrefab Load(std::string_view xml, MjcfImportInfo& info) {
    fs::path const path = _dir.Path() / "model.xml";
    mochi::WriteFile(path, xml, ExpectOK{});
    return LoadBotPrefabFromMjcfFile(path.string(), info, ExpectOK{});
  }

  BotPrefab Load(std::string_view xml) {
    MjcfImportInfo info;
    return Load(xml, info);
  }

  void WriteExtraFile(std::string_view relativePath, std::string_view contents) {
    fs::path const path = _dir.Path() / relativePath;
    fs::create_directories(path.parent_path());
    mochi::WriteFile(path, contents, ExpectOK{});
  }

  static int LinkIndex(BotPrefab const& prefab, std::string_view name) {
    for (int i = 0; i < isize(prefab.links); ++i) {
      if (std::string_view(prefab.links[i].name) == name) {
        return i;
      }
    }
    ADD_FAILURE() << "No link named " << name;
    return 0;
  }

  static BotJointPrefab const& Joint(BotPrefab const& prefab, std::string_view name) {
    for (auto const& joint : prefab.joints) {
      if (std::string_view(joint.name) == name) {
        return joint;
      }
    }
    ADD_FAILURE() << "No joint named " << name;
    return prefab.joints[0];
  }

  static TransformRT RootFromLink(BotPrefab const& prefab, std::string_view name) {
    return prefab.links[LinkIndex(prefab, name)]._rootFromLink;
  }

  LogFn _prevLogFn{};
  mochi::TempDirCleanup _dir = mochi::CreateTempDirectory("mjcf_import_test", ExpectOK{});
};

// ---------------------------------------------------------------------------
// Compiler and orientation
// ---------------------------------------------------------------------------

TEST_F(MjcfImportTest, HingeRangeUsesDegreesByDefault) {
  BotPrefab const prefab = Load(R"(
<mujoco>
  <worldbody>
    <body name="base">
      <body name="arm"><joint name="hinge" range="-90 45"/></body>
    </body>
  </worldbody>
</mujoco>)");
  BotJointPrefab const& hinge = Joint(prefab, "hinge");
  ASSERT_TRUE(hinge.minLimit && hinge.maxLimit);
  EXPECT_NEAR(Dot(*hinge.minLimit, hinge.axis), -kPI / 2_r, kTol);
  EXPECT_NEAR(Dot(*hinge.maxLimit, hinge.axis), kPI / 4_r, kTol);
}

TEST_F(MjcfImportTest, HingeRangeUsesRadiansWhenCompilerSaysSo) {
  BotPrefab const prefab = Load(R"(
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <body name="arm"><joint name="hinge" range="-1 0.5"/></body>
    </body>
  </worldbody>
</mujoco>)");
  BotJointPrefab const& hinge = Joint(prefab, "hinge");
  ASSERT_TRUE(hinge.minLimit && hinge.maxLimit);
  EXPECT_NEAR(Dot(*hinge.minLimit, hinge.axis), -1_r, kTol);
  EXPECT_NEAR(Dot(*hinge.maxLimit, hinge.axis), 0.5_r, kTol);
}

TEST_F(MjcfImportTest, EveryOrientationFormDescribesTheSameRotation) {
  // Each body is rotated 90 degrees about Z, written five different ways.
  BotPrefab const prefab = Load(R"(
<mujoco>
  <worldbody>
    <body name="base">
      <body name="quat" quat="0.70710678 0 0 0.70710678"/>
      <body name="axisangle" axisangle="0 0 1 90"/>
      <body name="euler" euler="0 0 90"/>
      <body name="xyaxes" xyaxes="0 1 0 -1 0 0"/>
      <body name="zaxis_x" zaxis="1 0 0"/>
    </body>
  </worldbody>
</mujoco>)");
  Quaternion const expected = Quaternion::RotationZ(kPI / 2_r);
  for (std::string_view name : {"quat", "axisangle", "euler", "xyaxes"}) {
    EXPECT_TRUE(NearEqual(RootFromLink(prefab, name).GetRotation(), expected, kTol)) << name;
  }
  // zaxis rotates +Z onto +X: a quarter turn about +Y.
  EXPECT_TRUE(NearEqual(
      RootFromLink(prefab, "zaxis_x").GetRotation(), Quaternion::RotationY(kPI / 2_r), kTol));
}

TEST_F(MjcfImportTest, EulerSequenceFollowsCompilerSetting) {
  // Intrinsic "xyz": rotate about X, then about the rotated Y.
  BotPrefab const prefab = Load(R"(
<mujoco>
  <compiler angle="radian" eulerseq="xyz"/>
  <worldbody>
    <body name="base"><body name="b" euler="0.3 0.5 0"/></body>
  </worldbody>
</mujoco>)");
  Quaternion const expected = Quaternion::RotationX(0.3_r) * Quaternion::RotationY(0.5_r);
  EXPECT_TRUE(NearEqual(RootFromLink(prefab, "b").GetRotation(), expected, kTol));
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

TEST_F(MjcfImportTest, DefaultClassesInheritAndChildclassApplies) {
  BotPrefab const prefab = Load(R"(
<mujoco>
  <default>
    <joint damping="1"/>
    <default class="arm">
      <joint axis="0 1 0" armature="0.2"/>
      <default class="wrist">
        <joint damping="3"/>
      </default>
    </default>
  </default>
  <worldbody>
    <body name="base" childclass="arm">
      <body name="upper"><joint name="shoulder"/></body>
      <body name="hand"><joint name="wrist" class="wrist" frictionloss="0.5"/></body>
      <body name="plain"><joint name="explicit" damping="7" class="main"/></body>
    </body>
  </worldbody>
</mujoco>)");
  BotJointPrefab const& shoulder = Joint(prefab, "shoulder");
  EXPECT_EQ(shoulder.axis, (Real3{0_r, 1_r, 0_r}));
  EXPECT_NEAR(shoulder.friction.viscous, 1_r, kTol); // inherited from main
  EXPECT_NEAR(shoulder.inertia.value_or(0_r), 0.2_r, kTol);

  BotJointPrefab const& wrist = Joint(prefab, "wrist");
  EXPECT_NEAR(wrist.friction.viscous, 3_r, kTol); // overridden by the nested class
  EXPECT_NEAR(wrist.inertia.value_or(0_r), 0.2_r, kTol); // inherited from arm
  EXPECT_NEAR(wrist.friction.coulomb, 0.5_r, kTol); // set on the element

  BotJointPrefab const& explicitJoint = Joint(prefab, "explicit");
  EXPECT_NEAR(explicitJoint.friction.viscous, 7_r, kTol);
  EXPECT_EQ(explicitJoint.axis, (Real3{0_r, 0_r, 1_r})); // main class: MuJoCo's default axis
}

// ---------------------------------------------------------------------------
// Joints and inertia
// ---------------------------------------------------------------------------

TEST_F(MjcfImportTest, JointTypesAndEffortLimits) {
  BotPrefab const prefab = Load(R"(
<mujoco>
  <worldbody>
    <body name="base">
      <body name="a"><joint name="slider" type="slide" axis="2 0 0" range="-0.1 0.2"/></body>
      <body name="b"><joint name="ball" type="ball"/></body>
      <body name="c"><joint name="motorized" actuatorfrcrange="-10 10"/></body>
      <body name="d"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="motorized" gear="2" forcerange="-3 3"/>
  </actuator>
</mujoco>)");
  BotJointPrefab const& slider = Joint(prefab, "slider");
  EXPECT_EQ(slider.type, ArticulatedJointType::Prismatic);
  EXPECT_EQ(slider.axis, (Real3{1_r, 0_r, 0_r}));
  ASSERT_TRUE(slider.minLimit);
  EXPECT_NEAR(Dot(*slider.minLimit, slider.axis), -0.1_r, kTol); // slide ranges are not angles

  EXPECT_EQ(Joint(prefab, "ball").type, ArticulatedJointType::Spherical);
  // The actuator's torque limit (2 x 3) is tighter than the joint's own (10).
  EXPECT_NEAR(Joint(prefab, "motorized").effortLimit, 6_r, kTol);
  EXPECT_EQ(Joint(prefab, "d_fixed").type, ArticulatedJointType::Hard);
}

TEST_F(MjcfImportTest, FullInertiaIsReordered) {
  BotPrefab const prefab = Load(R"(
<mujoco>
  <worldbody>
    <body name="base">
      <inertial pos="0.1 0.2 0.3" mass="2" fullinertia="1 2 3 0.1 0.2 0.3"/>
    </body>
  </worldbody>
</mujoco>)");
  BotLinkPrefab const& link = prefab.links[LinkIndex(prefab, "base")];
  ASSERT_TRUE(link.mass && link.centerOfMass && link.momentOfInertia);
  EXPECT_NEAR(*link.mass, 2_r, kTol);
  EXPECT_EQ(*link.centerOfMass, (Real3{0.1_r, 0.2_r, 0.3_r}));
  // MJCF (ixx iyy izz ixy ixz iyz) -> Mochi (ixx ixy ixz iyy iyz izz).
  Real6 const expected{1_r, 0.1_r, 0.2_r, 2_r, 0.3_r, 3_r};
  EXPECT_TRUE(NearEqual(*link.momentOfInertia, expected, kTol));
}

TEST_F(MjcfImportTest, DiagonalInertiaIsRotatedIntoTheBodyFrame) {
  // A quarter turn about Z swaps the X and Y principal moments.
  BotPrefab const prefab = Load(R"(
<mujoco>
  <worldbody>
    <body name="base">
      <inertial pos="0 0 0" mass="1" diaginertia="1 2 3" axisangle="0 0 1 90"/>
    </body>
  </worldbody>
</mujoco>)");
  BotLinkPrefab const& link = prefab.links[LinkIndex(prefab, "base")];
  ASSERT_TRUE(link.momentOfInertia);
  Real6 const expected{2_r, 0_r, 0_r, 1_r, 0_r, 3_r};
  EXPECT_TRUE(NearEqual(*link.momentOfInertia, expected, kTol));
}

TEST_F(MjcfImportTest, LinkFrameSitsAtTheJoint) {
  // The link is framed at the joint (0.1 along X), so the body's geometry and center of mass are
  // offset by -0.1 in the link frame.
  MjcfImportInfo info;
  BotPrefab const prefab = Load(
      R"(
<mujoco>
  <worldbody>
    <body name="base">
      <body name="arm" pos="0 0 1">
        <joint name="hinge" pos="0.1 0 0"/>
        <inertial pos="0.5 0 0" mass="1" diaginertia="1 1 1"/>
        <geom name="g" type="box" size="0.1 0.1 0.1" pos="0.3 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>)",
      info);
  int const arm = LinkIndex(prefab, "arm");
  EXPECT_TRUE(
      NearEqual(prefab.links[arm]._rootFromLink.GetTranslation(), Real3{0.1_r, 0_r, 1_r}, kTol));
  EXPECT_TRUE(NearEqual(*prefab.links[arm].centerOfMass, Real3{0.4_r, 0_r, 0_r}, kTol));
  MjcfBodyReference const& body = info.bodies[1];
  ASSERT_EQ(std::string_view(body.name), "arm");
  EXPECT_TRUE(NearEqual(body.linkFromBody.GetTranslation(), Real3{-0.1_r, 0_r, 0_r}, kTol));
  ASSERT_EQ(body.geoms.size(), 1u);
  EXPECT_TRUE(NearEqual(body.geoms[0].linkFromGeom.GetTranslation(), Real3{0.2_r, 0_r, 0_r}, kTol));
}

TEST_F(MjcfImportTest, BodyWithSeveralJointsBecomesAChainOfMasslessLinks) {
  BotPrefab const prefab = Load(R"(
<mujoco>
  <worldbody>
    <body name="base">
      <body name="hip">
        <joint name="hip_x" axis="1 0 0"/>
        <joint name="hip_y" axis="0 1 0" pos="0 0 0.05"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>)");
  int const dummy = LinkIndex(prefab, "hip__hip_x");
  int const hip = LinkIndex(prefab, "hip");
  EXPECT_EQ(prefab.links[hip].parentLink, dummy);
  EXPECT_FALSE(prefab.links[dummy].mass.has_value());
  EXPECT_TRUE(prefab.links[hip].mass.has_value());
  EXPECT_EQ(Joint(prefab, "hip_x").axis, (Real3{1_r, 0_r, 0_r}));
  EXPECT_EQ(Joint(prefab, "hip_y").axis, (Real3{0_r, 1_r, 0_r}));
  // The second joint sits 0.05 above the first.
  EXPECT_TRUE(NearEqual(
      Joint(prefab, "hip_y").parentLinkFromJoint.GetTranslation(), Real3{0_r, 0_r, 0.05_r}, kTol));
}

// ---------------------------------------------------------------------------
// Roots
// ---------------------------------------------------------------------------

TEST_F(MjcfImportTest, FreeJointRootFloats) {
  BotPrefab const prefab = Load(R"(
<mujoco model="floater">
  <worldbody>
    <body name="torso" pos="0 0 1"><freejoint name="root"/></body>
  </worldbody>
</mujoco>)");
  EXPECT_EQ(std::string_view(prefab.name), "floater");
  EXPECT_EQ(std::string_view(prefab.joints[0].name), "root");
  EXPECT_EQ(prefab.joints[0].type, ArticulatedJointType::Free);
  EXPECT_TRUE(NearEqual(prefab.worldFromRoot.GetTranslation(), Real3{0_r, 0_r, 1_r}, kTol));
}

TEST_F(MjcfImportTest, FixedRootIsWelded) {
  BotPrefab const prefab = Load(R"(
<mujoco><worldbody><body name="base" pos="1 0 0"/></worldbody></mujoco>)");
  EXPECT_EQ(prefab.joints[0].type, ArticulatedJointType::Hard);
  EXPECT_EQ(std::string_view(prefab.links[0].name), "base");
  EXPECT_TRUE(NearEqual(prefab.worldFromRoot.GetTranslation(), Real3{1_r, 0_r, 0_r}, kTol));
}

TEST_F(MjcfImportTest, SeveralFixedTopLevelBodiesShareAWorldRoot) {
  BotPrefab const prefab = Load(R"(
<mujoco>
  <worldbody>
    <body name="left" pos="0 1 0"/>
    <body name="right" pos="0 -1 0"><joint name="slide" type="slide"/></body>
  </worldbody>
</mujoco>)");
  EXPECT_EQ(std::string_view(prefab.links[0].name), "world");
  EXPECT_EQ(prefab.joints[0].type, ArticulatedJointType::Hard);
  EXPECT_EQ(prefab.links[LinkIndex(prefab, "left")].parentLink, 0);
  EXPECT_EQ(prefab.links[LinkIndex(prefab, "right")].parentLink, 0);
}

TEST_F(MjcfImportTest, FreeTopLevelBodyIsTheRobot) {
  // A floating robot cannot share its root with scene content, which is skipped.
  MjcfImportInfo info;
  BotPrefab const prefab = Load(
      R"(
<mujoco>
  <worldbody>
    <body name="table"/>
    <body name="torso" pos="0 0 1"><freejoint/><body name="leg"><joint name="knee"/></body></body>
  </worldbody>
</mujoco>)",
      info);
  EXPECT_EQ(std::string_view(prefab.links[0].name), "torso");
  EXPECT_EQ(prefab.joints[0].type, ArticulatedJointType::Free);
  EXPECT_EQ(prefab.links.size(), 2u);
  EXPECT_FALSE(info.warnings.empty());
}

// ---------------------------------------------------------------------------
// Keyframes, names, equality constraints, includes
// ---------------------------------------------------------------------------

TEST_F(MjcfImportTest, HomeKeyframeBecomesTheDefaultPose) {
  BotPrefab const prefab = Load(R"(
<mujoco>
  <worldbody>
    <body name="base">
      <freejoint/>
      <body name="a"><joint name="a" range="-90 90"/></body>
      <body name="b"><joint name="b" type="ball"/></body>
    </body>
  </worldbody>
  <keyframe>
    <key name="other" qpos="0 0 0 1 0 0 0  0.1  1 0 0 0"/>
    <key name="home" qpos="0 0 0 1 0 0 0  0.5  0.70710678 0.70710678 0 0"/>
  </keyframe>
</mujoco>)");
  // The free joint's 7 entries are not bot DOFs. The ball quaternion becomes a rotation vector.
  ASSERT_EQ(prefab.defaultPose.size(), 4u);
  EXPECT_NEAR(prefab.defaultPose[0], 0.5_r, kTol);
  EXPECT_NEAR(prefab.defaultPose[1], kPI / 2_r, kTol);
  EXPECT_NEAR(prefab.defaultPose[2], 0_r, kTol);
  EXPECT_NEAR(prefab.defaultPose[3], 0_r, kTol);
}

TEST_F(MjcfImportTest, NamesAreSanitizedAndJointsKeepTheirNames) {
  BotPrefab const prefab = Load(R"(
<mujoco>
  <worldbody>
    <body name="left/base">
      <body name="elbow"><joint name="elbow"/></body>
    </body>
  </worldbody>
</mujoco>)");
  EXPECT_EQ(std::string_view(prefab.links[0].name), "left_base");
  // A body and joint may share a name in MJCF but not in a bot; the joint keeps it.
  EXPECT_EQ(Joint(prefab, "elbow").type, ArticulatedJointType::Revolute);
  LinkIndex(prefab, "elbow_link");
}

TEST_F(MjcfImportTest, ConnectEqualityBecomesACycle) {
  BotPrefab const prefab = Load(R"(
<mujoco>
  <worldbody>
    <body name="base">
      <body name="a" pos="1 0 0"><joint name="ja"/></body>
      <body name="b" pos="0 1 0"><joint name="jb"/></body>
    </body>
  </worldbody>
  <equality><connect body1="a" body2="b" anchor="0 0.5 0"/></equality>
</mujoco>)");
  ASSERT_EQ(prefab.cycles.size(), 1u);
  ArticulatedCycleJointParams const& cycle = prefab.cycles[0];
  EXPECT_EQ(cycle.childLink, LinkIndex(prefab, "a"));
  EXPECT_EQ(cycle.parentLink, LinkIndex(prefab, "b"));
  // The pivot is the anchor in body a's frame.
  EXPECT_TRUE(
      NearEqual(Invert(cycle.jointFromChildLink).GetTranslation(), Real3{0_r, 0.5_r, 0_r}, kTol));
}

TEST_F(MjcfImportTest, NestedIncludesResolveRelativeToTheIncludingFile) {
  WriteExtraFile("parts/arm.xml", R"(
<mujoco><body name="arm"><joint name="shoulder"/><include file="hand.xml"/></body></mujoco>)");
  WriteExtraFile("parts/hand.xml", R"(
<mujoco><body name="hand"><joint name="wrist"/></body></mujoco>)");
  BotPrefab const prefab = Load(R"(
<mujoco>
  <worldbody>
    <body name="base"><include file="parts/arm.xml"/></body>
  </worldbody>
</mujoco>)");
  EXPECT_EQ(prefab.links[LinkIndex(prefab, "hand")].parentLink, LinkIndex(prefab, "arm"));
  EXPECT_EQ(Joint(prefab, "wrist").type, ArticulatedJointType::Revolute);
}

TEST_F(MjcfImportTest, IncludedAssetsMayBeRelativeToTheirFile) {
  WriteExtraFile("assets/asset/meshes.xml", R"(
<mujoco><asset><mesh name="part" file="../geometry/part.obj"/></asset></mujoco>)");
  WriteExtraFile("assets/geometry/part.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
  MjcfImportInfo info;
  (void)Load(
      R"(
<mujoco>
  <include file="assets/asset/meshes.xml"/>
  <worldbody><body name="base"><geom type="mesh" mesh="part"/></body></worldbody>
</mujoco>)",
      info);
  EXPECT_EQ(
      fs::path(std::string(info.bodies[0].geoms[0].meshFile)),
      (_dir.Path() / "assets/geometry/part.obj").lexically_normal());
}

TEST_F(MjcfImportTest, PrimitiveNamingAMeshKeepsItForFitting) {
  WriteExtraFile("part.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
  MjcfImportInfo info;
  (void)Load(
      R"(
<mujoco>
  <asset><mesh name="part" file="part.obj"/></asset>
  <worldbody><body name="base"><geom type="capsule" mesh="part" fitscale="2"/></body></worldbody>
</mujoco>)",
      info);
  GeomReference const& geom = info.bodies[0].geoms[0];
  EXPECT_EQ(geom.type, GeomType::Capsule);
  EXPECT_FALSE(geom.meshFile.empty());
  EXPECT_TRUE(geom.shapeFile.empty());
  EXPECT_EQ(geom.fitScale, 2_r);
}

TEST_F(MjcfImportTest, TopLevelDefaultsMergeIntoMain) {
  // A second top-level <default> (as an <include> can add) extends the main class.
  BotPrefab const prefab = Load(R"(
<mujoco>
  <default><joint damping="2"/></default>
  <default><joint armature="0.3"/></default>
  <worldbody>
    <body name="base"><body name="arm"><joint name="shoulder"/></body></body>
  </worldbody>
</mujoco>)");
  BotJointPrefab const& shoulder = Joint(prefab, "shoulder");
  EXPECT_NEAR(shoulder.friction.viscous, 2_r, kTol);
  EXPECT_NEAR(shoulder.inertia.value_or(0_r), 0.3_r, kTol);
}

TEST_F(MjcfImportTest, StringLoadHasNoGeometryAndRejectsIncludes) {
  BotPrefab const prefab = LoadBotPrefabFromMjcfString(
      R"(
<mujoco>
  <asset><mesh name="part" file="part.obj"/></asset>
  <worldbody>
    <body name="base"><geom type="mesh" mesh="part"/>
      <body name="arm"><joint name="shoulder"/><geom type="mesh" mesh="part"/></body>
    </body>
  </worldbody>
</mujoco>)",
      ExpectOK{});
  EXPECT_EQ(isize(prefab.links), 2);
  for (BotLinkPrefab const& link : prefab.links) {
    EXPECT_TRUE(link.shapeFile.empty()) << link.name;
  }
  (void)LoadBotPrefabFromMjcfString(
      R"(<mujoco><include file="robot.xml"/><worldbody/></mujoco>)", ExpectNotOK{});
}

TEST_F(MjcfImportTest, MalformedVectorIsAnError) {
  fs::path const path = _dir.Path() / "bad.xml";
  mochi::WriteFile(
      path, R"(<mujoco><worldbody><body name="b" pos="0 x 0"/></worldbody></mujoco>)", ExpectOK{});
  MjcfImportInfo info;
  (void)LoadBotPrefabFromMjcfFile(path.string(), info, ExpectNotOK{});
}

} // namespace
