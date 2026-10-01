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

#include <mochi_physics/mochi_physics_experimental.h>

#include <limits>

using namespace mochi;
using experimental::SleepParams;

namespace {

/***************************************************************************************************
  SleepParams
*/

class SceneSleepParams : public test::MochiSceneTestBase {};

TEST_F(SceneSleepParams, DefaultsDisableSleep) {
  EXPECT_EQ(SleepParams{}, experimental::GetSleepParams(_scene, test::ExpectOK{}));
  EXPECT_FALSE(experimental::GetSleepParams(_scene, test::ExpectOK{}).canSleep);
}

TEST_F(SceneSleepParams, SetThenGet) {
  {
    auto const params =
        SleepParams{.canSleep = true, .sleepThreshold = 1_r, .minStepsBeforeSleep = 2};
    experimental::SetSleepParams(_scene, params, test::ExpectOK{});
    EXPECT_EQ(params, experimental::GetSleepParams(_scene, test::ExpectOK{}));
  }
  {
    auto const params =
        SleepParams{.canSleep = false, .sleepThreshold = 0.1_r, .minStepsBeforeSleep = 3};
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

} // namespace
