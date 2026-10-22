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

#include "core/async_task.h"

#include <mochi_physics/mochi_physics.h>

#include <gtest/gtest.h>
#include <imgui.h>

#include <atomic>
#include <chrono>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace superdex::studio;

namespace {

class AsyncTaskRunnerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(1280.0f, 720.0f);
    io.DeltaTime = 1.0f / 60.0f;
    io.Fonts->Build();
    _context = mochi::CreateContext();
  }

  void TearDown() override {
    mochi::DestroyContext(_context);
    ImGui::DestroyContext();
  }

  // Draws frames until the batch finishes, calling @p onFrame first in each. Returns the
  // allSucceeded that onComplete received.
  template <typename OnFrame>
  bool Run(AsyncTaskRunner::Config config, OnFrame const& onFrame) {
    std::optional<bool> result;
    config.title = "Tasks";
    config.context = _context;
    config.onComplete = [&result](bool allSucceeded) { result = allSucceeded; };
    _runner.Begin(std::move(config));
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!result.has_value() && std::chrono::steady_clock::now() < deadline) {
      ImGui::NewFrame();
      onFrame();
      _runner.ShowModalWindow();
      ImGui::Render();
      // Frames are cheap; pace them like the app does rather than spin.
      // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(result.has_value()) << "the batch did not finish";
    return result.value_or(false);
  }

  // Statuses by batch index, read while the batch runs.
  std::vector<AsyncTaskStatus> Statuses() {
    std::vector<AsyncTaskStatus> statuses;
    for (AsyncTaskRunner::TaskInfo const& info : _runner.GetTaskInfos()) {
      statuses.push_back(info.status);
    }
    return statuses;
  }

  AsyncTaskRunner _runner;
  mochi::Context* _context = nullptr;
};

AsyncTask Task(std::string label, bool result, std::atomic<int>& runs) {
  return {std::move(label), [result, &runs](AsyncCancelToken const&) {
            ++runs;
            return result;
          }};
}

// Runs until it is cancelled or skipped, then returns false, as a task blocked in a helper does
// once the helper is killed.
AsyncTask BlockingTask(std::string label, std::atomic<bool>& started) {
  return {std::move(label), [&started](AsyncCancelToken const& cancel) {
            started = true;
            while (!cancel.IsCancelRequested()) {
              // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
              std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return false;
          }};
}

// The last task of a batch, held open until @p release so the batch's statuses can be read.
AsyncTask GateTask(std::atomic<bool>& reached, std::atomic<bool>& release) {
  return {"Gate", [&reached, &release](AsyncCancelToken const&) {
            reached = true;
            while (!release) {
              // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
              std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return true;
          }};
}

} // namespace

TEST_F(AsyncTaskRunnerTest, DependentsOfAFailedTaskAreSkipped) {
  std::atomic<int> runs{0};
  std::atomic<bool> reached{false};
  std::atomic<bool> release{false};
  AsyncTaskRunner::Config config;
  config.tasks.push_back(Task("Fails", false, runs));
  config.tasks.push_back(Task("Needs Fails", true, runs));
  config.tasks.back().dependsOn = {0};
  config.tasks.push_back(Task("Needs Needs Fails", true, runs));
  config.tasks.back().dependsOn = {1};
  config.tasks.push_back(Task("Independent", true, runs));
  config.tasks.push_back(GateTask(reached, release));
  std::vector<AsyncTaskStatus> statuses;
  EXPECT_FALSE(Run(std::move(config), [&] {
    if (reached && !release) {
      statuses = Statuses();
      release = true;
    }
  }));
  EXPECT_EQ(runs, 2);
  EXPECT_EQ(
      statuses,
      (std::vector<AsyncTaskStatus>{
          AsyncTaskStatus::Failed,
          AsyncTaskStatus::Skipped,
          AsyncTaskStatus::Skipped,
          AsyncTaskStatus::Succeeded,
          AsyncTaskStatus::Running}));
}

TEST_F(AsyncTaskRunnerTest, SkippingAPendingTaskSkipsItsDependents) {
  std::atomic<bool> started{false};
  std::atomic<int> runs{0};
  std::atomic<bool> reached{false};
  std::atomic<bool> release{false};
  AsyncTaskRunner::Config config;
  config.tasks.push_back(BlockingTask("Blocks", started));
  config.tasks.push_back(Task("Skipped", true, runs));
  config.tasks.push_back(Task("Needs Skipped", true, runs));
  config.tasks.back().dependsOn = {1};
  config.tasks.push_back(GateTask(reached, release));
  std::vector<AsyncTaskStatus> statuses;
  bool skippedPending = false;
  EXPECT_FALSE(Run(std::move(config), [&] {
    if (started && !skippedPending) {
      skippedPending = _runner.SkipTask("Skipped");
      EXPECT_TRUE(skippedPending);
      EXPECT_TRUE(_runner.SkipTask("Blocks"));
    }
    if (reached && !release) {
      statuses = Statuses();
      release = true;
    }
  }));
  EXPECT_EQ(runs, 0);
  EXPECT_EQ(
      statuses,
      (std::vector<AsyncTaskStatus>{
          AsyncTaskStatus::Skipped,
          AsyncTaskStatus::Skipped,
          AsyncTaskStatus::Skipped,
          AsyncTaskStatus::Running}));
}

TEST_F(AsyncTaskRunnerTest, SkippingARunningTaskAbortsItsWorker) {
  std::atomic<bool> started{false};
  std::atomic<int> runs{0};
  std::vector<std::thread::id> aborted;
  AsyncTaskRunner::Config config;
  config.tasks.push_back(BlockingTask("Blocks", started));
  config.tasks.push_back(Task("Needs Blocks", true, runs));
  config.tasks.back().dependsOn = {0};
  config.tasks.push_back(Task("Independent", true, runs));
  config.abortTask = [&aborted](std::thread::id worker) { aborted.push_back(worker); };
  bool skipped = false;
  EXPECT_FALSE(Run(std::move(config), [&] {
    if (started && !skipped) {
      skipped = _runner.SkipTask("Blocks");
      EXPECT_TRUE(skipped);
      // A task already being stopped cannot be skipped again.
      EXPECT_FALSE(_runner.SkipTask("Blocks"));
      EXPECT_TRUE(_runner.GetTaskInfos()[0].skipping);
    }
  }));
  EXPECT_EQ(runs, 1);
  ASSERT_EQ(aborted.size(), 1u);
  EXPECT_NE(aborted[0], std::this_thread::get_id());
  EXPECT_NE(aborted[0], std::thread::id{});
}

TEST_F(AsyncTaskRunnerTest, ASkippedTaskThatCompletesKeepsItsResult) {
  std::atomic<bool> started{false};
  std::atomic<bool> finish{false};
  std::atomic<int> runs{0};
  AsyncTaskRunner::Config config;
  config.tasks.push_back({"Ignores Skip", [&](AsyncCancelToken const&) {
                            started = true;
                            while (!finish) {
                              // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
                              std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            }
                            return true;
                          }});
  config.tasks.push_back(Task("Needs Ignores Skip", true, runs));
  config.tasks.back().dependsOn = {0};
  EXPECT_TRUE(Run(std::move(config), [&] {
    if (started && !finish) {
      EXPECT_TRUE(_runner.SkipTask("Ignores Skip"));
      finish = true;
    }
  }));
  EXPECT_EQ(runs, 1);
}

TEST_F(AsyncTaskRunnerTest, ATaskRunningPastTheTimeoutIsTimedOut) {
  std::atomic<bool> started{false};
  std::atomic<int> runs{0};
  std::atomic<bool> reached{false};
  std::atomic<bool> release{false};
  AsyncTaskRunner::Config config;
  config.tasks.push_back(BlockingTask("Blocks", started));
  config.tasks.push_back(Task("Needs Blocks", true, runs));
  config.tasks.back().dependsOn = {0};
  config.tasks.push_back(Task("Independent", true, runs));
  config.tasks.push_back(GateTask(reached, release));
  config.timeoutSeconds = 0.05;
  std::vector<AsyncTaskStatus> statuses;
  EXPECT_FALSE(Run(std::move(config), [&] {
    if (reached && !release) {
      statuses = Statuses();
      release = true;
    }
  }));
  EXPECT_EQ(runs, 1);
  ASSERT_FALSE(statuses.empty());
  EXPECT_EQ(
      statuses,
      (std::vector<AsyncTaskStatus>{
          AsyncTaskStatus::TimedOut,
          AsyncTaskStatus::Skipped,
          AsyncTaskStatus::Succeeded,
          // The gate itself runs past the timeout while the statuses are read.
          statuses.back()}));
}

TEST_F(AsyncTaskRunnerTest, SkipTaskNeedsAPendingOrRunningTask) {
  std::atomic<int> runs{0};
  std::atomic<bool> reached{false};
  std::atomic<bool> release{false};
  AsyncTaskRunner::Config config;
  config.tasks.push_back(Task("Done", true, runs));
  config.tasks.push_back(GateTask(reached, release));
  EXPECT_FALSE(_runner.SkipTask("Done"));
  EXPECT_TRUE(Run(std::move(config), [&] {
    if (reached && !release) {
      EXPECT_FALSE(_runner.SkipTask("Done"));
      EXPECT_FALSE(_runner.SkipTask("Missing"));
      release = true;
    }
  }));
  EXPECT_TRUE(_runner.GetTaskInfos().empty());
}
