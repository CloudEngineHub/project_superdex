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

// Loads the vendored mujoco_menagerie Franka FR3 v2 (assets/test/mjcf/franka_fr3_v2) with the
// runtime MJCF loader and compares it with MuJoCo 3.16's compiled model of the same file at qpos0.

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/log.h>
#include <mochi_physics/mochi_physics.h>
#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/file_utils.h>

#include "mochi_bots_test_helpers.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <map>
#include <string>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

namespace fs = std::filesystem;

// MuJoCo's world-frame values: body position and orientation (w, x, y, z), and for bodies with an
// <inertial>, mass, center of mass, and moment of inertia (xx, xy, xz, yy, yz, zz).
struct ReferenceBody {
  char const* name;
  Real3 pos;
  std::array<real, 4> quat;
  bool hasInertial;
  real mass;
  Real3 com;
  Real6 inertia;
};

// clang-format off
std::array<ReferenceBody, 10> const kBodies = {{
    {"base", {0_r, 0_r, 0_r}, {1_r, 0_r, 0_r, 0_r}, false, 0_r, {}, {}},
    {"fr3v2_link0", {0_r, 0_r, 0_r}, {1_r, 0_r, 0_r, 0_r}, true, 3.85_r,
     {-0.0172_r, 0.0004_r, 0.0745_r},
     {0.00900002043_r, 0_r, 0.00200001563_r, 0.0115_r, 0_r, 0.00850001957_r}},
    {"fr3v2_link1", {0_r, 0_r, 0.333_r}, {1_r, 0_r, 0_r, 0_r}, true, 2.4377_r,
     {-0.00458_r, 0.027016_r, 0.2712_r},
     {0.0232400417_r, 0.000333006131_r, -0.000676001878_r, 0.0214600048_r, 0.00373901741_r,
      0.00550000354_r}},
    {"fr3v2_link2", {0_r, 0_r, 0.333_r}, {0.707106781_r, -0.707106781_r, 0_r, 0_r}, true, 2.2375_r,
     {-0.003473_r, -0.010479_r, 0.501509_r},
     {0.0190260383_r, 0.000499013282_r, -0.00124100235_r, 0.0171139649_r, 0.00180399432_r,
      0.00565999681_r}},
    {"fr3v2_link3", {0_r, 0_r, 0.649_r}, {1_r, 0_r, 0_r, 0_r}, true, 2.18579_r,
     {0.092968_r, 0.01801_r, 0.682533_r},
     {0.010225017_r, 0.00114396934_r, 0.00349700252_r, 0.0123980091_r, 0.00215598243_r,
      0.00687699393_r}},
    {"fr3v2_link4", {0.0825_r, 0_r, 0.649_r}, {0.707106781_r, 0.707106781_r, 0_r, 0_r}, true,
     2.17587_r, {0.013755_r, -0.011195_r, 0.741947_r},
     {0.00887802034_r, -0.0011390101_r, -0.00374901864_r, 0.01029501_r, 0.00046001211_r,
      0.00762800967_r}},
    {"fr3v2_link5", {0_r, 0_r, 1.033_r}, {1_r, 0_r, 0_r, 0_r}, true, 2.17482_r,
     {-0.004197_r, 0.010257_r, 0.974897_r},
     {0.0318740209_r, 0.000134005579_r, -0.000265003667_r, 0.0296739758_r, -0.00473399593_r,
      0.00475100336_r}},
    {"fr3v2_link6", {0_r, 0_r, 1.033_r}, {0.707106781_r, 0.707106781_r, 0_r, 0_r}, true,
     1.56762_r, {0.070958_r, 0.035324_r, 1.018863_r},
     {0.00286000049_r, 0.000637999339_r, -0.000553001186_r, 0.00452999816_r, 0.00024000025_r,
      0.00391000135_r}},
    {"fr3v2_link7", {0.088_r, 0_r, 1.033_r}, {0_r, 1_r, 0_r, 0_r}, true, 0.682876_r,
     {0.096204_r, -0.001156_r, 0.971984_r},
     {0.000899000179_r, -5.91031402e-11_r, 1.29998468e-05_r, 0.00109900201_r, -2.20000945e-05_r,
      0.000701999809_r}},
    {"fr3v2_link8", {0.088_r, 0_r, 0.926_r}, {0_r, 1_r, 0_r, 0_r}, false, 0_r, {}, {}},
}};
// clang-format on

// MuJoCo's world-frame hinge axis, range, damping, armature, and friction loss.
struct ReferenceJoint {
  char const* name;
  Real3 axis;
  real lo;
  real hi;
  real damping;
  real armature;
  real frictionLoss;
};

std::array<ReferenceJoint, 7> const kJoints = {{
    {"fr3v2_joint1", {0_r, 0_r, 1_r}, -2.9007_r, 2.9007_r, 0.003_r, 0.195_r, 0.2_r},
    {"fr3v2_joint2", {0_r, 1_r, 0_r}, -1.8361_r, 1.8361_r, 0.003_r, 0.195_r, 0.2_r},
    {"fr3v2_joint3", {0_r, 0_r, 1_r}, -2.9007_r, 2.9007_r, 0.003_r, 0.195_r, 0.2_r},
    {"fr3v2_joint4", {0_r, -1_r, 0_r}, -3.077_r, -0.1169_r, 0.003_r, 0.195_r, 0.2_r},
    {"fr3v2_joint5", {0_r, 0_r, 1_r}, -2.8763_r, 2.8763_r, 0.003_r, 0.195_r, 0.2_r},
    {"fr3v2_joint6", {0_r, -1_r, 0_r}, 0.4398_r, 4.6216_r, 0.003_r, 0.195_r, 0.2_r},
    {"fr3v2_joint7", {0_r, 0_r, -1_r}, -3.0508_r, 3.0508_r, 0.003_r, 0.195_r, 0.2_r},
}};

real RotationAngle(Quaternion const& a, Quaternion const& b) {
  Real4 const qa = a.ToReal4();
  Real4 const qb = b.ToReal4();
  real const dot = Abs(qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3]);
  return 2_r * std::acos(Min(dot, 1_r));
}

// R I R^T for a symmetric tensor stored as (xx, xy, xz, yy, yz, zz).
Real6 RotateInertiaTensor(Real6 const& inertia, Quaternion const& rotation) {
  std::array<std::array<real, 3>, 3> const full = {{
      {inertia[0], inertia[1], inertia[2]},
      {inertia[1], inertia[3], inertia[4]},
      {inertia[2], inertia[4], inertia[5]},
  }};
  std::array<Real3, 3> columns;
  for (int k = 0; k < 3; ++k) {
    Real3 axis = {};
    axis[k] = 1_r;
    columns[k] = rotation * axis;
  }
  auto entry = [&](int r, int c) {
    real sum = 0_r;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        sum += columns[i][r] * full[i][j] * columns[j][c];
      }
    }
    return sum;
  };
  return Real6{entry(0, 0), entry(0, 1), entry(0, 2), entry(1, 1), entry(1, 2), entry(2, 2)};
}

class MjcfFr3v2Test : public testing::Test {
 protected:
  void SetUp() override {
    // The FR3 v2 uses MJCF features the loader skips with a warning (a site, actuator gains), and
    // its raw link7 collision mesh is not closed.
    _warningsWereEnabled = IsLogChannelEnabled(LogChannel::Warning);
    EnableLogChannel(LogChannel::Warning, false);
  }
  void TearDown() override {
    EnableLogChannel(LogChannel::Warning, _warningsWereEnabled);
  }

  static std::string Fr3v2Path() {
    return (fs::path(GetAssetsDir()) / "test" / "mjcf" / "franka_fr3_v2" / "fr3v2.xml").string();
  }

  bool _warningsWereEnabled = true;
};

} // namespace

TEST_F(MjcfFr3v2Test, MatchesMuJoCo) {
  MjcfImportInfo info;
  BotPrefab const prefab = LoadBotPrefabFromMjcfFile(Fr3v2Path(), info, ExpectOK{});

  std::map<std::string, int> linkIndex;
  for (int i = 0; i < isize(prefab.links); ++i) {
    linkIndex[std::string(prefab.links[i].name)] = i;
  }
  std::map<std::string, MjcfBodyReference const*> bodyByName;
  for (MjcfBodyReference const& body : info.bodies) {
    bodyByName[std::string(body.name)] = &body;
  }
  auto worldFromLink = [&](int link) {
    return prefab.worldFromRoot * prefab.links[link]._rootFromLink;
  };

  constexpr real kPositionTol = 1e-4_r;
  constexpr real kAngleTol = 1e-3_r;
  for (ReferenceBody const& ref : kBodies) {
    ASSERT_TRUE(bodyByName.count(ref.name)) << ref.name;
    MjcfBodyReference const& body = *bodyByName.at(ref.name);
    int const link = linkIndex.at(std::string(body.link));
    TransformRT const worldFromBody = worldFromLink(link) * body.linkFromBody;
    Quaternion const refRotation{ref.quat[1], ref.quat[2], ref.quat[3], ref.quat[0]};
    EXPECT_LT(Norm(worldFromBody.GetTranslation() - ref.pos), kPositionTol) << ref.name;
    EXPECT_LT(RotationAngle(worldFromBody.GetRotation(), Normalize(refRotation)), kAngleTol)
        << ref.name;
    EXPECT_EQ(body.hasInertial, ref.hasInertial) << ref.name;
    if (!ref.hasInertial) {
      continue;
    }
    BotLinkPrefab const& linkPrefab = prefab.links[link];
    ASSERT_TRUE(linkPrefab.mass && linkPrefab.centerOfMass && linkPrefab.momentOfInertia);
    EXPECT_NEAR(*linkPrefab.mass, ref.mass, 1e-5_r * ref.mass) << ref.name;
    EXPECT_LT(
        Norm(worldFromLink(link).TransformPoint(*linkPrefab.centerOfMass) - ref.com), kPositionTol)
        << ref.name;
    Real6 const inertia =
        RotateInertiaTensor(*linkPrefab.momentOfInertia, worldFromLink(link).GetRotation());
    real scale = 0_r;
    for (real const value : ref.inertia) {
      scale = Max(scale, Abs(value));
    }
    for (int i = 0; i < 6; ++i) {
      EXPECT_NEAR(inertia[i], ref.inertia[i], 1e-3_r * scale) << ref.name << " inertia " << i;
    }
  }

  std::map<std::string, int> jointIndex;
  for (int i = 0; i < isize(prefab.joints); ++i) {
    jointIndex[std::string(prefab.joints[i].name)] = i;
  }
  for (ReferenceJoint const& ref : kJoints) {
    ASSERT_TRUE(jointIndex.count(ref.name)) << ref.name;
    // A joint and the link it moves share an index.
    int const j = jointIndex.at(ref.name);
    BotJointPrefab const& joint = prefab.joints[j];
    EXPECT_EQ(joint.type, ArticulatedJointType::Revolute) << ref.name;
    EXPECT_LT(Norm(worldFromLink(j).GetRotation() * joint.axis - ref.axis), kAngleTol) << ref.name;
    ASSERT_TRUE(joint.minLimit && joint.maxLimit) << ref.name;
    EXPECT_NEAR(Dot(*joint.minLimit, joint.axis), ref.lo, 1e-5_r) << ref.name;
    EXPECT_NEAR(Dot(*joint.maxLimit, joint.axis), ref.hi, 1e-5_r) << ref.name;
    EXPECT_NEAR(joint.friction.viscous, ref.damping, 1e-6_r) << ref.name;
    EXPECT_NEAR(joint.friction.coulomb, ref.frictionLoss, 1e-6_r) << ref.name;
    EXPECT_NEAR(joint.inertia.value_or(0_r), ref.armature, 1e-6_r) << ref.name;
  }
}

TEST_F(MjcfFr3v2Test, LoadsMeshesAndSimulates) {
  BotPrefab prefab = LoadBotPrefabFromMjcfFile(Fr3v2Path(), ExpectOK{});
  Validate(prefab, nullptr, ExpectOK{});
  // Links 0-7 each have one collision mesh, and most have several visual meshes; all resolve.
  int numShapes = 0;
  for (BotLinkPrefab const& link : prefab.links) {
    if (!link.shapeFile.empty()) {
      ++numShapes;
      EXPECT_TRUE(fs::exists(std::string(link.shapeFile))) << link.name;
    }
    if (!link.renderModelFile.empty()) {
      EXPECT_TRUE(fs::exists(std::string(link.renderModelFile))) << link.name;
    }
  }
  EXPECT_EQ(numShapes, 8);

  Context* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));
  Scene* scene = context->CreateScene("MjcfFr3v2Test");
  RoboticsContext* botsContext = CreateRoboticsContext();
  MOCHI_DEFER(DestroyRoboticsContext(botsContext));
  Bot* bot = CreateBot(scene, prefab, botsContext, ExpectOK{});
  ASSERT_NE(bot, nullptr);
  for (int i = 0; i < 50; ++i) {
    scene->Step(0.002);
  }
  for (ActorHandle const handle : bot->GetArticulatedActor()->GetNestedLinkActors(ExpectOK{})) {
    EXPECT_TRUE(IsFinite(scene->GetActor(handle)->GetRootTransform().GetTranslation()))
        << scene->GetActor(handle)->GetName();
  }
  DestroyBot(scene, bot);
}
