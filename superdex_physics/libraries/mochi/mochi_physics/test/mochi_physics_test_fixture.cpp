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

#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/test/batched_work_test_utils.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_physics/src/mochi_shell.h>
#include <mochi_physics/src/mochi_soft.h>
#include <mochi_physics/src/mochi_soft_rom_components.h>

#include <functional>
#include <variant>

namespace mochi::test {

MochiContextTestBase::~MochiContextTestBase() {
  MOCHI_ASSERT(
      _mochiContext == nullptr,
      "Class inheriting from MochiContextTestBase has not correctly called TearDown()!"
      " This can result in memory leaks/corruption! Please clean up after yourself!");
}

/*******************************************************************************************
  MochiPhysicsTest (test fixture)
*/

void MochiContextTestBase::SetUp() {
  // Create mochi::Context
  _mochiContext = mochi::CreateContext(_numWorkerThreads);
  EXPECT_NE((Context*)nullptr, _mochiContext);

  // Take the logging callback that has been registered for this test project
  // and make sure it also applies with the mochi_physics DLL (does nothing
  // if mochi_physics is statically linked with this test).
  Context::SetLogCallback(mochi::GetLogCallback());
}

void mochi::test::MochiContextTestBase::TearDown() {
  mochi::DestroyContext(_mochiContext);
  _mochiContext = nullptr;
}

/*******************************************************************************************
  MochiContextTestWithParam (parameterized test fixture)
*/

void MochiContextTestWithParam::SetUp() {
  // Create mochi::Context with some numer of worker threads
  _numWorkerThreads = GetParam().numWorkerThreads;
  Class::SetUp();

  // Apply other parameters
  _mochiContext->SetIsSingleThreaded(GetParam().isSingleThreadedMode);
}

/*******************************************************************************************
  MochiSceneTest
*/

void MochiSceneTestBase::SetUp() {
  MochiContextTestBase::SetUp();

  _scene = _mochiContext->CreateScene("MochiPhysicsTest");
  EXPECT_NE((mochi::Scene*)nullptr, _scene);
}

void MochiSceneTestBase::TearDown() {
  EXPECT_NE((mochi::Scene*)nullptr, _scene);
  _mochiContext->DestroyScene(_scene);
  _scene = nullptr;

  MochiContextTestBase::TearDown();
}

/*******************************************************************************************
  MochiAsyncSceneTest
*/

MochiAsyncSceneTestBase::MochiAsyncSceneTestBase() {
  // Must have at least one worker thread to create an AsyncScene
  _numWorkerThreads = 1;
}

void MochiAsyncSceneTestBase::SetUp() {
  MochiContextTestBase::SetUp();

  _asyncScene = _mochiContext->CreateAsyncScene("MochiPhysicsTest", ExpectOK{});
  EXPECT_NE((mochi::AsyncScene*)nullptr, _asyncScene);
}

void MochiAsyncSceneTestBase::TearDown() {
  EXPECT_NE((mochi::AsyncScene*)nullptr, _asyncScene);
  _mochiContext->DestroyAsyncScene(_asyncScene);
  _asyncScene = nullptr;

  MochiContextTestBase::TearDown();
}

/*******************************************************************************************
  ExpectLoggingInScope
*/

ExpectLoggingInScope::ExpectLoggingInScope(Context* mochiContext, LogChannel channel)
    : _mochiContext(mochiContext),
      _channel(channel),
      _prevCallbackForThisModule(mochi::GetLogCallback()),
      _prevCallbackForMochiPhysics(mochiContext ? mochiContext->GetLogCallback() : LogFn{}) {
  LogFn ourCallback =
      [this](LogChannel channel, char const* /*message*/, char const* /*file*/, int /*line*/) {
        if (channel == _channel) {
          ++_messagesReceived;
        }
      };
  mochi::Context::SetLogCallback(ourCallback);
}

ExpectLoggingInScope::~ExpectLoggingInScope() {
  mochi::Context::SetLogCallback(_prevCallbackForMochiPhysics);
  if (_messagesReceived == 0) {
    ADD_FAILURE() << "Expected a log message, but it never happened.";
  }
}

// Checks the body assembly of a soft or shell actor at a uniform translation, where the
// displacement u, stage-start displacement u₀ and velocity v₀ take the same value at every node.
// There the consistent mass matrix acts as the lumped mass m, stress and stiffness damping exert no
// force, and the stress energy is a constant E₀. For each DoF i, of component c, with α the actor's
// mass-damping coefficient and g the gravity in the actor frame:
//   residual_i = m_i·((u − u₀ − dt·v₀)_c/dt² + α·(u − u₀)_c/dt − g_c),
//   (dresidual·1)_i = (1/dt² + α/dt)·m_i,
//   objective = E₀ + Σᵢ m_i·((u − u₀ − dt·v₀)_c²/(2·dt²) + α·(u − u₀)_c²/(2·dt) − g_c·(u − u₀)_c).
// Every output combination is assembled, into outputs that hold stale values.
void ExpectUniformTranslationAssembly(Scene* scene, Actor* actor, real relTol) {
  constexpr Real3 kDispl = {0.3_r, -0.2_r, 0.1_r};
  constexpr Real3 kStageStartDispl = {-0.1_r, 0.25_r, 0.05_r};
  constexpr Real3 kStageStartVel = {2_r, -1_r, 3_r};
  auto& reg = GetRegistry(scene);
  entt::entity const e = GetEntity(reg, actor->GetHandle(), ExpectOK{});
  auto& displ = reg.get<CDisplacementSlice<real, TimeStep::Current>>(e).value;
  auto const setUniform = [](ColumnVector<real>& values, Real3 const& value) {
    for (int i = 0; i < values.Rows(); ++i) {
      values[i] = value[i % kSpaceDim3];
    }
  };
  setUniform(reg.get<CDisplacementSlice<real, TimeStep::StageStart>>(e).value, kStageStartDispl);
  setUniform(reg.get<CVelocitySlice<real, TimeStep::StageStart>>(e).value, kStageStartVel);

  real const dt = reg.get<CTimeIntegratorState const>(e).dtStage;
  real const alpha = reg.all_of<TagShellActor>(e)
      ? reg.get<CShellMaterialParams const>(e).massDampingCoefficient
      : reg.get<CSoftMaterialParams const>(e).massDampingCoefficient;
  Real3 const gravity = reg.all_of<TagUseGravity>(e)
      ? actor->GetRootTransform().TransformDirectionInverse(scene->GetGravity())
      : Real3{};
  Real3 const predTarget = kStageStartDispl + kStageStartVel * dt;
  auto const& lumpedMass = reg.get<CLumpedMassMatrix const>(e).values;
  int const numDofs = isize(lumpedMass);
  // Objective at the uniform displacement u, but for the stress energy.
  auto const objectiveBesidesStress = [&](Real3 const& u) {
    double objective = 0.0;
    for (int i = 0; i < numDofs; ++i) {
      int const c = i % kSpaceDim3;
      objective += static_cast<double>(
          lumpedMass[i] *
          (Sqr(u[c] - predTarget[c]) / (2_r * Sqr(dt)) +
           alpha * Sqr(u[c] - kStageStartDispl[c]) / (2_r * dt) -
           gravity[c] * (u[c] - kStageStartDispl[c])));
    }
    return objective;
  };
  ColumnVector<real> expectedRes(numDofs);
  ColumnVector<real> expectedDResRowSums(numDofs);
  for (int i = 0; i < numDofs; ++i) {
    int const c = i % kSpaceDim3;
    expectedRes[i] = lumpedMass[i] *
        ((kDispl[c] - predTarget[c]) / Sqr(dt) + alpha * (kDispl[c] - kStageStartDispl[c]) / dt -
         gravity[c]);
    expectedDResRowSums[i] = (1_r / Sqr(dt) + alpha / dt) * lumpedMass[i];
  }

  auto& snle = reg.get<CActorSnle>(e);
  auto const assemble = [&](AssemblyParams const& params) {
    if (reg.all_of<TagShellActor>(e)) {
      ecs::InvokeOnEntity(shell::AssembleBody, reg, e, std::cref(params));
    } else {
      ecs::InvokeOnEntity(soft::AssembleBody, reg, e, std::cref(params));
    }
  };
  // The stress energy is the same at every translation. Take it at the predicted target.
  setUniform(displ, predTarget);
  assemble({.assemObj = true, .assemRes = false, .assemDRes = false});
  double const stressEnergy = snle.objective - objectiveBesidesStress(predTarget);

  setUniform(displ, kDispl);
  for (auto const cfg : fem::kAllOutputConfigs) {
    snle.objective = 1e3;
    snle.fullResidual.SetConstant(1e3_r);
    Fill(GetValues(snle.fullDResidual), 1e3_r);
    assemble({.assemObj = cfg.energy, .assemRes = cfg.residual, .assemDRes = cfg.dresidual});
    if (cfg.energy) {
      EXPECT_NEAR_RTOL(
          stressEnergy + objectiveBesidesStress(kDispl),
          snle.objective,
          static_cast<double>(relTol));
    }
    if (cfg.residual) {
      ColumnVector<real> const resError = snle.fullResidual - expectedRes;
      EXPECT_LE(resError.Norm(), relTol * expectedRes.Norm());
    }
    if (cfg.dresidual) {
      ColumnVector<real> const dresRowSumsError =
          ToMatrix(std::get<BlockSparseMatrix<real, 3>>(snle.fullDResidual)) *
              ColumnVector<real>::Ones(numDofs) -
          expectedDResRowSums;
      EXPECT_LE(dresRowSumsError.Norm(), relTol * expectedDResRowSums.Norm());
    }
  }
}

} // namespace mochi::test
