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

#include "mochi_physics_test_fixture.h"

#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_physics/mochi_physics_experimental.h>

// These tests inspect island sleep state, which is not exposed through the public API.
#include <mochi_physics/src/mochi_island.h>
#include <mochi_physics/src/mochi_sleep.h>

#include <limits>
#include <numeric>
#include <string>

using namespace mochi;
using namespace mochi::experimental;

/***************************************************************************************************
  SleepParams
*/

namespace {
class SceneSleepParams : public test::MochiSceneTestBase {};
} // namespace

TEST_F(SceneSleepParams, DefaultsDisableSleep) {
  EXPECT_EQ(SleepParams{}, experimental::GetSleepParams(_scene, test::ExpectOK{}));
  EXPECT_FALSE(experimental::GetSleepParams(_scene, test::ExpectOK{}).canSleep);
}

TEST_F(SceneSleepParams, SetThenGet) {
  for (SleepParams const& params :
       {SleepParams{.canSleep = true, .sleepThreshold = 1_r, .minStepsBeforeSleep = 2},
        SleepParams{.canSleep = false, .sleepThreshold = 0.1_r, .minStepsBeforeSleep = 3}}) {
    experimental::SetSleepParams(_scene, params, test::ExpectOK{});
    EXPECT_EQ(params, experimental::GetSleepParams(_scene, test::ExpectOK{}));
  }
}

TEST_F(SceneSleepParams, RejectsInvalidParams) {
  auto const expectRejected = [&](SleepParams const& params) {
    experimental::SetSleepParams(_scene, params, test::ExpectNotOK{});
    EXPECT_EQ(SleepParams{}, experimental::GetSleepParams(_scene, test::ExpectOK{}))
        << "Invalid params must not be stored.";
  };
  expectRejected({.sleepThreshold = 0_r});
  expectRejected({.sleepThreshold = -0.5_r});
  expectRejected({.sleepThreshold = 1.001_r});
  expectRejected({.sleepThreshold = std::numeric_limits<real>::quiet_NaN()});
  expectRejected({.minStepsBeforeSleep = 0});
  expectRejected({.minStepsBeforeSleep = 1});
  expectRejected({.minStepsBeforeSleep = -1});
}

TEST_F(SceneSleepParams, DifferentiableSceneDisablesSleep) {
  experimental::SetSleepParams(_scene, {.canSleep = true}, test::ExpectOK{});
  test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);
  {
    test::ExpectLoggingInScope expectWarning(_mochiContext, LogChannel::Warning);
    MakeSceneDifferentiableInternal(_scene, test::ExpectOK{});
  }
  EXPECT_FALSE(experimental::GetSleepParams(_scene, test::ExpectOK{}).canSleep);

  experimental::SetSleepParams(_scene, {.canSleep = true}, test::ExpectNotOK{});
  experimental::SetSleepParams(_scene, {.canSleep = false}, test::ExpectOK{});
}

/***************************************************************************************************
  Stage rest values
*/

static NewtonSolverStatus<real>
MakeStatus(ConvergenceStatus convergence, int numIterDone, real resNorm0, real resNorm) {
  NewtonSolverStatus<real> status;
  status.convergence = convergence;
  status.numIterDone = numIterDone;
  status.resNorm0 = resNorm0;
  status.resNorm = resNorm;
  return status;
}

TEST(SceneSleepRestValue, ZeroIterationsIsFullyAtRest) {
  EXPECT_EQ(1_r, sleep::StageRestValue(MakeStatus(ConvergenceStatus::Converged, 0, 0_r, 0_r)));
  EXPECT_EQ(
      1_r, sleep::StageRestValue(MakeStatus(ConvergenceStatus::Converged, 0, 1e-4_r, 1e-4_r)));
}

TEST(SceneSleepRestValue, NotConvergedIsNotAtRest) {
  for (auto convergence :
       {ConvergenceStatus::None, ConvergenceStatus::Stopped, ConvergenceStatus::Diverged}) {
    EXPECT_EQ(0_r, sleep::StageRestValue(MakeStatus(convergence, 0, 1_r, 1_r)));
    EXPECT_EQ(0_r, sleep::StageRestValue(MakeStatus(convergence, 3, 1_r, 1_r)));
  }
}

TEST(SceneSleepRestValue, IterationsGiveResidualRatio) {
  EXPECT_NEAR_EQ(
      0.25_r, sleep::StageRestValue(MakeStatus(ConvergenceStatus::Converged, 2, 4_r, 1_r)));
}

/***************************************************************************************************
  Test fixture
*/

static DynamicArray<ActorType> GetAllActorTypes() {
  DynamicArray<ActorType> types;
  for (int i = static_cast<int>(ActorType::None) + 1; i < static_cast<int>(ActorType::Count); ++i) {
    types.push_back(static_cast<ActorType>(i));
  }
  return types;
}

static std::string ActorTypeName(testing::TestParamInfo<ActorType> const& info) {
  return SReflect::EnumToString(info.param);
}

static DynamicArray<real> GetAllDofValues(Actor const* actor) {
  DynamicArray<real> values(actor->GetNumDofs());
  actor->GetDofValues({}, MakeSpan(values), test::ExpectOK{});
  return values;
}

namespace {

class SceneSleep : public test::MochiSceneTestBase {
 public:
  static constexpr double kDt = 0.01;
  static constexpr int kMaxStepsToSleep = 1000;

  void SetUp() override {
    test::MochiSceneTestBase::SetUp();
    // Backward Euler's numerical damping lets actors settle quickly.
    test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);
    experimental::SetSleepParams(_scene, {.canSleep = true}, test::ExpectOK{});
    _ground = CreateGround();
  }

  Actor* CreateGround(Real3 normal = Real3{0_r, 1_r, 0_r}) {
    RigidActorParams params;
    params.name = "Ground";
    params.isStatic = true;
    params.shape = _mochiContext->CreatePlaneShape(Normalize(normal), 0_r, test::ExpectOK{});
    params.colliderType = ColliderType::Plane;
    return _scene->CreateRigidActor(params, test::ExpectOK{});
  }

  ShapeHandle CreateCubeShape(real size) {
    auto [coords, conn] = test::CreateMinimalTetMeshUnitCube(Real3{size, size, size});
    return _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(coords)), Flatten(MakeSpan(conn)), test::ExpectOK{});
  }

  // Create an actor whose bottom is at the given height, centered on x = z = 0.
  Actor* CreateTestActor(ActorType type, real bottomHeight = 0.02_r) {
    constexpr real kCubeSize = 0.2_r; // Edge length of rigid and soft cubes [m]
    constexpr real kThinActorSize = 0.8_r * kCubeSize; // Shells and rods fit on top of a cube [m]
    // The cube and patch meshes have a corner at their local origin.
    auto const cornerAt = [bottomHeight](real size) {
      return Real3{-0.5_r * size, bottomHeight, -0.5_r * size};
    };
    switch (type) {
      case ActorType::Rigid: {
        RigidActorParams params;
        params.shape = CreateCubeShape(kCubeSize);
        params.colliderType = ColliderType::Box;
        params.worldFromLocal.SetTranslation(cornerAt(kCubeSize));
        return _scene->CreateRigidActor(params, test::ExpectOK{});
      }
      case ActorType::Soft: {
        SoftActorParams params;
        params.shape = CreateCubeShape(kCubeSize);
        params.worldFromLocal.SetTranslation(cornerAt(kCubeSize));
        return _scene->CreateSoftActor(params, test::ExpectOK{});
      }
      case ActorType::Shell: {
        DynamicArray<Real3> const coords = {
            Real3{0_r, 0_r, 0_r},
            Real3{kThinActorSize, 0_r, 0_r},
            Real3{kThinActorSize, 0_r, kThinActorSize},
            Real3{0_r, 0_r, kThinActorSize}};
        DynamicArray<Int3> const conn = {Int3{0, 1, 2}, Int3{0, 2, 3}};
        ShellActorParams params;
        params.shape = _mochiContext->CreateTriMeshShape(
            Flatten(MakeConstSpan(coords)), Flatten(MakeConstSpan(conn)), test::ExpectOK{});
        params.worldFromLocal.SetTranslation(cornerAt(kThinActorSize));
        return CreateShellActor(_scene, params, test::ExpectOK{});
      }
      case ActorType::Rod: {
        // Along x, centered on the rod's local origin.
        constexpr int kNumNodes = 5;
        DynamicArray<Real3> nodes;
        for (int i = 0; i < kNumNodes; ++i) {
          real const t = StaticCast<real>(i) / (kNumNodes - 1) - 0.5_r;
          nodes.push_back(Real3{t * kThinActorSize, 0_r, 0_r});
        }
        DynamicArray<Real3> const elementFrameAxes(kNumNodes - 1, Real3{0_r, 1_r, 0_r});
        RodActorParams params;
        params.shape = CreatePolylineShape(
            _mochiContext, nodes, elementFrameAxes, /*isClosedLoop=*/false, test::ExpectOK{});
        params.material.linearDensity = 10_r;
        params.material.linearRotationalInertia = 1_r;
        params.material.axialStiffness = 1e5_r;
        params.material.torsionalStiffness = 1e3_r;
        params.material.flexuralStiffness = {1e3_r, 1e3_r};
        params.worldFromLocal.SetTranslation(Real3{0_r, bottomHeight, 0_r});
        return CreateRodActor(_scene, params, test::ExpectOK{});
      }
      case ActorType::Articulated: {
        real const linkSize = 0.5_r * kCubeSize;
        auto const linkShape = CreateCubeShape(linkSize);
        ArticulatedActorParams params;
        params.joints = {
            {.type = ArticulatedJointType::Free},
            {.type = ArticulatedJointType::Spherical,
             .parentLinkFromJoint = TransformRT{Real3{0.5_r * linkSize, 0_r, 0_r}}}};
        params.links = {
            {.parentLink = -1, .shape = linkShape, .colliderType = ColliderType::Box},
            {.parentLink = 0,
             .parentJointFromLink = TransformRT{Real3{0.5_r * linkSize, 0_r, 0_r}},
             .shape = linkShape,
             .colliderType = ColliderType::Box}};
        params.worldFromRoot.SetTranslation(Real3{-linkSize, bottomHeight, -0.5_r * linkSize});
        return _scene->CreateArticulatedActor(params, test::ExpectOK{});
      }
      case ActorType::None:
      case ActorType::Count:
        break;
    }
    static_assert(
        static_cast<int>(ActorType::Count) == 6,
        "If you add an actor type, then please update the code above.");
    ADD_FAILURE() << "Unsupported actor type.";
    return nullptr;
  }

  entt::entity GetIsland(Actor const* actor) const {
    auto const* memberInfo =
        GetRegistry().try_get<CIslandMemberInfo const>(GetEntity(actor->GetHandle()));
    EXPECT_NE(nullptr, memberInfo);
    return memberInfo ? memberInfo->island : entt::entity{entt::null};
  }

  bool IsAwake(Actor const* actor) const {
    return sleep::IsIslandAwake(GetRegistry(), GetIsland(actor));
  }

  void Step(int numSteps = 1) {
    for (int i = 0; i < numSteps; ++i) {
      _scene->Step(kDt);
    }
  }

  // Step until the actor's island is asleep. Call this right after the island was created or woken.
  // Fails if the island falls asleep sooner than SleepParams::minStepsBeforeSleep allows, or if it
  // is still awake after maxSteps.
  void StepUntilAsleep(Actor const* actor, int maxSteps = kMaxStepsToSleep) {
    // An island solved N times at rest goes to sleep during the next step.
    int const minSteps =
        experimental::GetSleepParams(_scene, test::ExpectOK{}).minStepsBeforeSleep + 1;
    int numSteps = 0;
    do {
      ASSERT_LT(numSteps, maxSteps);
      Step();
      ++numSteps;
    } while (IsAwake(actor));
    ASSERT_GE(numSteps, minSteps);
  }

  // Step and fail if the actor's island is asleep after any of the steps.
  void StepExpectAwake(Actor const* actor, int numSteps) {
    for (int i = 0; i < numSteps; ++i) {
      Step();
      ASSERT_TRUE(IsAwake(actor));
    }
  }

  // Position of the actor, for measuring how far it moved. Articulated actors have no world AABB,
  // so use their root joint's translation DoFs, which are relative to the initial root transform.
  static Real3 GetPosition(Actor const* actor) {
    if (actor->GetType() == ActorType::Articulated) {
      auto const dofs = GetAllDofValues(actor);
      return Real3{dofs[0], dofs[1], dofs[2]};
    }
    return actor->GetAabbWorld(test::ExpectOK{}).GetCenter();
  }

  // Node positions of a deformable actor before any simulation, in the world frame.
  static DynamicArray<Real3> GetRestNodePositionsWorld(Actor const* actor) {
    TransformRT const worldFromLocal = actor->GetRootTransform();
    DynamicArray<Real3> positions;
    for (Real3 const& p : Unflatten<Real3 const>(actor->GetMesh().coordinates)) {
      positions.push_back(worldFromLocal.TransformPoint(p));
    }
    return positions;
  }

  // Hold every position DoF of the actor in place with boundary conditions.
  static void ConstrainInPlace(Actor* actor) {
    ActorType const type = actor->GetType();
    if (type == ActorType::Rigid || type == ActorType::Articulated) {
      auto const values = GetAllDofValues(actor);
      DynamicArray<int> dofs(isize(values));
      std::iota(dofs.begin(), dofs.end(), 0);
      actor->AddBoundaryConditionDofsWorld(
          MakeConstSpan(dofs), MakeConstSpan(values), test::ExpectOK{});
      return;
    }
    auto const positions = GetRestNodePositionsWorld(actor);
    if (type == ActorType::Rod) {
      // Rods do not support node boundary conditions. Their DoFs are 4 per node: the position, then
      // the twist.
      DynamicArray<int> dofs;
      for (int i = 0; i < isize(positions); ++i) {
        for (int k = 0; k < 3; ++k) {
          dofs.push_back(4 * i + k);
        }
      }
      actor->AddBoundaryConditionDofsWorld(
          MakeConstSpan(dofs), Flatten(MakeConstSpan(positions)), test::ExpectOK{});
      return;
    }
    DynamicArray<int> nodes(isize(positions));
    std::iota(nodes.begin(), nodes.end(), 0);
    actor->AddBoundaryConditionNodesWorld(
        MakeConstSpan(nodes), Flatten(MakeConstSpan(positions)), test::ExpectOK{});
  }

  // Give the actor a rigid-body velocity (world frame). Deformables rotate about their local
  // origin.
  static void SetVelocity(Actor* actor, Real3 const& linear, Real3 const& angular = Real3{}) {
    ActorType const type = actor->GetType();
    if (type == ActorType::Rigid) {
      actor->SetVelocity(linear, angular, test::ExpectOK{});
    } else if (type == ActorType::Articulated) {
      // The root free joint has translation DoFs first, then rotation DoFs.
      DynamicArray<real> velocities(actor->GetNumDofs(), 0_r);
      for (int k = 0; k < 3; ++k) {
        velocities[k] = linear[k];
        velocities[3 + k] = angular[k];
      }
      actor->SetArticulatedJointVelocities(MakeConstSpan(velocities), test::ExpectOK{});
    } else {
      // Node velocities in the local frame, which has no rotation. Rod nodes also have a twist.
      int const valuesPerNode = (type == ActorType::Rod) ? 4 : 3;
      auto const positions = Unflatten<Real3 const>(actor->GetMesh().coordinates);
      DynamicArray<real> velocities(isize(positions) * valuesPerNode, 0_r);
      for (int i = 0; i < isize(positions); ++i) {
        Real3 const v = linear + Cross(angular, positions[i]);
        for (int k = 0; k < 3; ++k) {
          velocities[valuesPerNode * i + k] = v[k];
        }
      }
      actor->SetNodeVelocitiesLocal(MakeConstSpan(velocities), test::ExpectOK{});
    }
  }

 protected:
  Actor* _ground = nullptr;
};

class SceneSleepPerType : public SceneSleep, public testing::WithParamInterface<ActorType> {};

} // namespace

/***************************************************************************************************
  Going to sleep
*/

TEST_P(SceneSleepPerType, SettlesOnGroundThenSleeps) {
  Actor const* actor = CreateTestActor(GetParam());
  StepUntilAsleep(actor);
}

TEST_P(SceneSleepPerType, FloatsWithoutGravityThenSleeps) {
  _scene->SetGravity(Real3{});
  Actor const* actor = CreateTestActor(GetParam(), /*bottomHeight=*/1_r);
  StepUntilAsleep(actor);
}

TEST_P(SceneSleepPerType, StateIsFrozenWhileAsleep) {
  Actor const* actor = CreateTestActor(GetParam());
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(actor));
  auto const dofValues = GetAllDofValues(actor);
  auto const rootTransform = actor->GetRootTransform();
  Step(10);
  EXPECT_FALSE(IsAwake(actor));
  EXPECT_EQ(dofValues, GetAllDofValues(actor));
  EXPECT_EQ(rootTransform.GetTranslation(), actor->GetRootTransform().GetTranslation());
}

TEST_P(SceneSleepPerType, SleepingActorHasNoConvergenceStatus) {
  Actor const* actor = CreateTestActor(GetParam());
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(actor));
  EXPECT_EQ(ConvergenceStatus::None, actor->GetConvergenceStatus());

  // Changing the sleep parameters wakes all islands.
  experimental::SetSleepParams(
      _scene, {.canSleep = true, .sleepThreshold = 0.25_r}, test::ExpectOK{});
  Step();
  ASSERT_TRUE(IsAwake(actor));
  EXPECT_NE(ConvergenceStatus::None, actor->GetConvergenceStatus());
}

TEST_P(SceneSleepPerType, CanSleepFalsePreventsSleep) {
  experimental::SetSleepParams(_scene, {.canSleep = false}, test::ExpectOK{});
  Actor const* actor = CreateTestActor(GetParam());
  StepExpectAwake(actor, kMaxStepsToSleep);
}

/***************************************************************************************************
  Staying awake while moving
*/

namespace {
class SceneSleepMoving : public SceneSleepPerType {
 public:
  static constexpr int kNumSteps = 200;

  void SetUp() override {
    SceneSleep::SetUp();
    _scene->SetGravity(Real3{});
    // Pin the tolerance below which the solver skips iterating. The speeds in these tests keep the
    // initial residual well above it.
    auto params = _scene->GetSolverParams();
    params.nonLinearSolver.convergenceMode = NonLinearSolverConvergenceMode::PerActorWeighted;
    params.nonLinearSolver.absTol = 1e-3_r;
    _scene->SetSolverParams(params, test::ExpectOK{});
  }

  // Like StepExpectAwake, but also fail if the solver skips iterating, since the island could then
  // rightly go to sleep.
  void StepExpectIteratingAndAwake(Actor const* actor, int numSteps) {
    for (int i = 0; i < numSteps; ++i) {
      Step();
      ASSERT_GT(_scene->GetSolverStats().maxNonLinearIters, 0);
      ASSERT_TRUE(IsAwake(actor));
    }
  }
};
} // namespace

TEST_P(SceneSleepMoving, SlowConstantVelocityStaysAwake) {
  // Slow, but well above the smallest speed the solver resolves at kDt (~1e-5 m/s without
  // gravity). Below that, the solver itself stops the motion, and then sleeping is correct.
  constexpr real kSpeed = 1e-3_r; // [m/s]
  Actor* actor = CreateTestActor(GetParam(), /*bottomHeight=*/1_r);
  SetVelocity(actor, Real3{kSpeed, 0_r, 0_r});
  real const startX = GetPosition(actor)[0];
  StepExpectIteratingAndAwake(actor, kNumSteps);
  real const expectedDistance = kSpeed * kNumSteps * kDt;
  EXPECT_NEAR(expectedDistance, GetPosition(actor)[0] - startX, 0.1_r * expectedDistance);
}

TEST_P(SceneSleepMoving, SlowConstantRotationStaysAwake) {
  constexpr real kAngularSpeed = 0.1_r; // [rad/s], ~1 cm/s at the ends of an actor
  Actor* actor = CreateTestActor(GetParam(), /*bottomHeight=*/1_r);
  SetVelocity(actor, Real3{}, Real3{0_r, kAngularSpeed, 0_r});
  auto const start = GetAllDofValues(actor);
  StepExpectIteratingAndAwake(actor, kNumSteps);
  EXPECT_NE(start, GetAllDofValues(actor));
}

INSTANTIATE_TEST_SUITE_P(
    AllActorTypes,
    SceneSleepMoving,
    testing::ValuesIn(GetAllActorTypes()),
    ActorTypeName);

TEST_P(SceneSleepPerType, ThrownUpwardStaysAwakeThroughApexAndFall) {
  Actor* actor = CreateTestActor(GetParam(), /*bottomHeight=*/0.5_r);
  SetVelocity(actor, Real3{0_r, 2_r, 0_r});
  // Rises for ~0.2 s, then falls. Stay awake through the apex and part of the fall.
  StepExpectAwake(actor, 40);
  real const height = GetPosition(actor)[1];
  StepExpectAwake(actor, 5);
  EXPECT_LT(GetPosition(actor)[1], height) << "Should be falling by now.";
}

TEST_P(SceneSleepPerType, FullyConstrainedSleeps) {
  Actor* actor = CreateTestActor(GetParam(), /*bottomHeight=*/1_r);
  ConstrainInPlace(actor);
  StepUntilAsleep(actor);
}

namespace {
// Swinging from a pivot 0.5 m away (rigid), or from one of its nodes (deformables).
class SceneSleepPendulum : public SceneSleepPerType {};
} // namespace

TEST_P(SceneSleepPendulum, SwingingStaysAwake) {
  Actor* actor = CreateTestActor(GetParam(), /*bottomHeight=*/1_r);
  if (GetParam() == ActorType::Rigid) {
    RigidPivotPositionConstraintParams params;
    params.actor = actor->GetHandle();
    params.localPosition = Real3{-0.5_r, 0_r, 0_r};
    params.targetPosition = actor->GetRootTransform().TransformPoint(params.localPosition);
    _scene->CreateRigidPivotPositionConstraint(params, test::ExpectOK{});
  } else {
    DeformableNodePositionConstraintParams params;
    params.actor = actor->GetHandle();
    params.nodeIndex = 0;
    params.position = GetRestNodePositionsWorld(actor)[0];
    _scene->CreateDeformableNodePositionConstraint(params, test::ExpectOK{});
  }
  // A few swings. Backward Euler damping takes much longer to stop them.
  StepExpectAwake(actor, 300);
}

INSTANTIATE_TEST_SUITE_P(
    ConstrainableActorTypes,
    SceneSleepPendulum,
    testing::Values(ActorType::Rigid, ActorType::Soft, ActorType::Shell, ActorType::Rod),
    ActorTypeName);

INSTANTIATE_TEST_SUITE_P(
    AllActorTypes,
    SceneSleepPerType,
    testing::ValuesIn(GetAllActorTypes()),
    ActorTypeName);

/***************************************************************************************************
  Stacks
*/

namespace {
struct StackParam {
  ActorType bottom = {};
  ActorType top = {};
};

class SceneSleepStack : public SceneSleep, public testing::WithParamInterface<StackParam> {};
} // namespace

TEST_P(SceneSleepStack, SettlesThenSleepsTogether) {
  // Let the bottom actor settle first, then drop the top one just above it.
  Actor const* bottom = CreateTestActor(GetParam().bottom);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(bottom));
  real const bottomTop = bottom->GetAabbWorld(test::ExpectOK{}).GetMax()[1];
  Actor const* top = CreateTestActor(GetParam().top, /*bottomHeight=*/bottomTop + 0.005_r);
  StepUntilAsleep(top);
  EXPECT_EQ(GetIsland(top), GetIsland(bottom));
  EXPECT_GT(GetPosition(top)[1], GetPosition(bottom)[1])
      << "The top actor should rest on the bottom one.";
}

INSTANTIATE_TEST_SUITE_P(
    Pairs,
    SceneSleepStack,
    testing::Values(
        StackParam{ActorType::Rigid, ActorType::Rigid},
        StackParam{ActorType::Rigid, ActorType::Soft},
        StackParam{ActorType::Soft, ActorType::Rigid},
        StackParam{ActorType::Rigid, ActorType::Shell}),
    [](testing::TestParamInfo<StackParam> const& info) {
      return std::string(SReflect::EnumToString(info.param.top)) + "On" +
          SReflect::EnumToString(info.param.bottom);
    });

TEST_F(SceneSleep, ManyIslandsSleepInTheSameStep) {
  _scene->SetGravity(Real3{});
  RigidActorParams params;
  params.shape = CreateCubeShape(0.2_r);
  params.colliderType = ColliderType::Box;
  DynamicArray<Actor const*> actors;
  for (int i = 0; i < 100; ++i) {
    // 1 m apart, so each actor is its own island.
    params.worldFromLocal.SetTranslation(Real3{StaticCast<real>(i), 1_r, 0_r});
    actors.push_back(_scene->CreateRigidActor(params, test::ExpectOK{}));
  }
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(actors[0]));
  for (Actor const* actor : actors) {
    EXPECT_FALSE(IsAwake(actor));
  }
}

/***************************************************************************************************
  Sleep parameters
*/

TEST_F(SceneSleep, SleepsAfterMinStepsAtMaxThreshold) {
  experimental::SetSleepParams(
      _scene,
      {.canSleep = true, .sleepThreshold = 1_r, .minStepsBeforeSleep = 5},
      test::ExpectOK{});
  _scene->SetGravity(Real3{});
  Actor const* actor = CreateTestActor(ActorType::Rigid, /*bottomHeight=*/1_r);
  // At rest from the start, and steps that take zero solver iterations record exactly 1, so it
  // sleeps as soon as the minimum step count allows.
  StepUntilAsleep(actor, /*maxSteps=*/6);
}

TEST_F(SceneSleep, ChangingSleepParamsWakesAllIslands) {
  Actor const* actor = CreateTestActor(ActorType::Rigid);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(actor));
  auto params = experimental::GetSleepParams(_scene, test::ExpectOK{});
  params.sleepThreshold = 0.25_r;
  experimental::SetSleepParams(_scene, params, test::ExpectOK{});
  StepUntilAsleep(actor);
}

TEST_F(SceneSleep, DisablingSleepWakesAllIslands) {
  experimental::SetSleepParams(
      _scene, {.canSleep = true, .minStepsBeforeSleep = 2}, test::ExpectOK{});
  _scene->SetGravity(Real3{});
  Actor const* actor = CreateTestActor(ActorType::Rigid);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(actor, /*maxSteps=*/3));
  experimental::SetSleepParams(
      _scene, {.canSleep = false, .minStepsBeforeSleep = 2}, test::ExpectOK{});
  StepExpectAwake(actor, 3);
}

/***************************************************************************************************
  Waking up
*/

TEST_F(SceneSleep, FallingActorWakesSleepingActor) {
  Actor const* bottom = CreateTestActor(ActorType::Rigid);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(bottom));
  Actor const* top = CreateTestActor(ActorType::Rigid, /*bottomHeight=*/0.5_r);
  // Islands merge when the falling actor gets close.
  StepExpectAwake(bottom, 1);
  EXPECT_EQ(GetIsland(top), GetIsland(bottom));
  StepUntilAsleep(bottom);
}

TEST_F(SceneSleep, DestroyingSupportWakesStack) {
  Actor* bottom = CreateTestActor(ActorType::Rigid);
  Aabb const support = bottom->GetAabbWorld(test::ExpectOK{});
  Actor const* top =
      CreateTestActor(ActorType::Rigid, /*bottomHeight=*/support.GetMax()[1] + 0.01_r);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(top));
  ASSERT_EQ(GetIsland(top), GetIsland(bottom));
  real const heightBefore = GetPosition(top)[1];
  _scene->DestroyActor(bottom);
  StepUntilAsleep(top);
  EXPECT_LT(GetPosition(top)[1], heightBefore - 0.5_r * support.GetSize()[1])
      << "The top actor should fall onto the ground.";
}

/***************************************************************************************************
  Scene statistics
*/

TEST_F(SceneSleep, SolverStatsIgnoreSleepingIslands) {
  Actor const* actor = CreateTestActor(ActorType::Rigid);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(actor));
  Step();
  auto const stats = _scene->GetSolverStats();
  EXPECT_EQ(0, stats.maxNonLinearIters);
  EXPECT_EQ(0.0, stats.residualNorm);
  EXPECT_EQ(ConvergenceStatus::None, stats.convergenceStatus);
}

TEST_F(SceneSleep, SolverStatsAggregateAwakeIslands) {
  Actor const* sleeping = CreateTestActor(ActorType::Rigid);
  ASSERT_NO_FATAL_FAILURE(StepUntilAsleep(sleeping));
  Actor const* falling = CreateTestActor(ActorType::Rigid, /*bottomHeight=*/2_r);
  Step();
  ASSERT_FALSE(IsAwake(sleeping));
  ASSERT_NE(GetIsland(sleeping), GetIsland(falling));
  EXPECT_NE(ConvergenceStatus::None, falling->GetConvergenceStatus());
  EXPECT_EQ(falling->GetConvergenceStatus(), _scene->GetSolverStats().convergenceStatus);
}
