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

#pragma once

#include "superdex_asset_tuner/tuner_spec.h"

#include <mochi_physics/mochi_physics.h>

#include <array>
#include <string>

namespace superdex::tuner {

// World-space state of one load during the last step, for visualization.
struct AppliedLoad {
  // Application point (forces), body origin (torques), joint child link origin, or the driven point
  // (PosePath).
  Real3 pointWorld{0.0f, 0.0f, 0.0f};
  // Force [N], torque [N m], the joint's generalized effort (first 3 DoFs), or the force the
  // PosePath drive applies.
  Real3 vectorWorld{0.0f, 0.0f, 0.0f};
  real magnitude = 0;
  // PosePath: where the path wants the point and body orientation at the end of the step, and how
  // far the body is from it (distance [m] and angle [rad]; 0 for what the drive leaves free).
  mochi::TransformRT targetPose;
  real trackingError = 0;
  real angleError = 0;
  // PosePath: the drive mode during the step.
  PathDriveMode driveMode = PathDriveMode::Off;
};

struct Sample {
  // Simulation time at the end of the step [s].
  double time = 0.0;
  // One value per channel, see TunerSession::GetChannelNames.
  DynamicArray<real> values;
  // One entry per load in spec order.
  DynamicArray<AppliedLoad> loads;
  // Index (into TunerSpec::stopConditions) of the stop condition this step triggered, or -1.
  int stopCondition = -1;
  // Why the run stopped, e.g. "'fell' block.pos.z = -0.101 < -0.1". Empty unless it stopped.
  DynamicString stopMessage;
};

// Applies a tuner spec's loads to a live scene and samples its probes. Not thread-safe: all calls
// must happen on the thread that steps the scene.
class TunerSession {
 public:
  // Resolves load and probe targets and registers the queries probes need. Call before the first
  // step. Unresolved targets fail with a message listing the available names.
  bool Bind(mochi::Scene& scene, TunerSpec const& spec, std::string& outError);

  // Clears external forces on targeted actors and cancels registered queries.
  void Unbind(mochi::Scene& scene);

  // Replaces load parameters mid-run. Load count, names, types and targets must be unchanged.
  bool UpdateLoads(Span<LoadSpec const> loads, std::string& outError);

  // Sets each targeted actor's external forces to the sum of its active loads.
  void OnPreStep(mochi::StepInfo const& info);

  // Advances time by the step, samples probes and applied loads, and checks the stop conditions.
  void OnPostStep(mochi::StepInfo const& info, Sample& outSample);

  [[nodiscard]] Span<DynamicString const> GetChannelNames() const {
    return _channelNames;
  }
  [[nodiscard]] Span<LoadSpec const> GetLoads() const {
    return _loads;
  }
  [[nodiscard]] double GetTime() const {
    return _time;
  }

 private:
  struct Accumulator {
    mochi::ActorHandle actor;
    DynamicArray<real> forces;
    DynamicArray<int> dofs;
    DynamicArray<real> values;
  };
  struct ResolvedLoad {
    int accumulator = -1;
    // Rigid actor or link actor whose pose defines pointLocal and Local-frame directions.
    mochi::ActorHandle body;
    // Force / Torque: `body` is an articulated link, loaded through its Jacobian.
    bool isLink = false;
    int dofOffset = 0;
    int dofCount = 0;
    // PosePath: the drive's constraints, the mass and inertia its gains scale with, and the gains
    // last applied (position stiffness, damping, saturation, then rotation's).
    mochi::ConstraintHandle positionDrive;
    mochi::ConstraintHandle rotationDrive;
    real driveMass = 0;
    real driveInertia = 0;
    std::array<real, 6> driveGains{};
  };
  struct ResolvedProbe {
    ProbeKind kind = ProbeKind::BodyPose;
    mochi::ActorHandle actor;
    mochi::QueryHandle query;
    DynamicArray<int> dofIndices;
    // JointState: the joint's DoF values (one per entry of dofIndices).
    DynamicArray<real> dofValues;
    DynamicArray<real> scratch;
    // BodyPose: last unwrapped rotation vector, the reference for the next step's unwrap.
    Real3 unwrappedRotvec{0.0f, 0.0f, 0.0f};
    bool ignoreGround = false;
    // Channels the probe adds to each sample.
    size_t numChannels = 0;
  };
  struct ResolvedStop {
    int specIndex = -1;
    StopCondition condition;
    mochi::ActorHandle body; // LeavesBox / EntersBox
    int channel = -1; // ChannelThreshold: index into the sample values
    // Time the condition started holding continuously, or negative while it does not hold.
    double holdingSince = -1.0;
  };

  bool ResolveLoad(mochi::Scene& scene, LoadSpec const& load, std::string& outError);
  bool ResolvePosePath(
      mochi::Scene& scene,
      LoadSpec const& load,
      ResolvedLoad& resolved,
      std::string& outError);
  // Sets the drive's gains for the load's settings (zero while disabled), then its targets.
  void DrivePosePath(
      mochi::Scene& scene,
      LoadSpec const& load,
      ResolvedLoad& resolved,
      AppliedLoad& applied,
      double timeStep);
  // Fills the post-step drive force and tracking errors into `applied`.
  void SamplePosePath(
      mochi::Scene& scene,
      LoadSpec const& load,
      ResolvedLoad const& resolved,
      AppliedLoad& applied) const;
  bool ResolveProbe(mochi::Scene& scene, ProbeSpec const& probe, std::string& outError);
  bool ResolveStop(mochi::Scene& scene, StopCondition const& stop, std::string& outError);
  void CheckStopConditions(mochi::Scene& scene, Sample& inOutSample);
  int GetOrAddAccumulator(mochi::Actor const& actor);
  void AddChannels(std::string_view prefix, std::initializer_list<std::string_view> suffixes);

  DynamicArray<LoadSpec> _loads;
  DynamicArray<ResolvedLoad> _resolvedLoads;
  DynamicArray<AppliedLoad> _appliedLoads;
  DynamicArray<Accumulator> _accumulators;
  DynamicArray<ResolvedProbe> _probes;
  DynamicArray<ResolvedStop> _stops;
  DynamicArray<DynamicString> _channelNames;
  double _time = 0.0;
};

} // namespace superdex::tuner
