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

#include "superdex_asset_tuner/tuner_session.h"

#include "scene_utils.h"
#include "superdex_asset_tuner/load_math.h"

#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/math_utils.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace superdex::tuner {

namespace {

using mochi::Actor;
using mochi::ActorType;

std::string JoinNames(Span<DynamicString const> names) {
  std::string joined;
  for (auto const& name : names) {
    joined += joined.empty() ? "" : ", ";
    joined += std::string_view(name);
  }
  return joined;
}

Actor*
FindActor(mochi::Scene& scene, std::string_view name, ActorType type, std::string& outError) {
  DynamicArray<DynamicString> available;
  for (Actor* actor : details::GetTopLevelActors(scene)) {
    if (actor->GetType() != type) {
      continue;
    }
    if (name == actor->GetName()) {
      return actor;
    }
    available.emplace_back(actor->GetName());
  }
  outError = "No " + std::string(type == ActorType::Rigid ? "rigid" : "articulated") +
      " actor named '" + std::string(name) + "'. Available: [" + JoinNames(available) + "]";
  return nullptr;
}

// Splits "actor/part" at the last '/' (actor names may contain '/' from nested prefabs).
bool SplitTarget(
    std::string_view target,
    std::string_view& outActor,
    std::string_view& outPart,
    std::string& outError) {
  size_t const slash = target.rfind('/');
  if (slash == std::string_view::npos || slash == 0 || slash + 1 == target.size()) {
    outError = "Target '" + std::string(target) + "' must be of the form 'actor/name'.";
    return false;
  }
  outActor = target.substr(0, slash);
  outPart = target.substr(slash + 1);
  return true;
}

int FindName(Span<DynamicString const> names, std::string_view name) {
  for (size_t i = 0; i < names.size(); ++i) {
    if (std::string_view(names[i]) == name) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

struct ArticulatedPart {
  Actor* articulated = nullptr;
  mochi::ActorHandle link;
  int index = -1;
};

// Resolves "actor/link" (linkNames) or "actor/joint" (jointNames). Joint i drives link i.
bool ResolveArticulatedPart(
    mochi::Scene& scene,
    std::string_view target,
    bool isJoint,
    ArticulatedPart& out,
    std::string& outError) {
  std::string_view actorName;
  std::string_view partName;
  if (!SplitTarget(target, actorName, partName, outError)) {
    return false;
  }
  out.articulated = FindActor(scene, actorName, ActorType::Articulated, outError);
  if (out.articulated == nullptr) {
    return false;
  }
  mochi::ErrorLog error;
  mochi::ArticulatedShapeInfo const info = out.articulated->GetArticulatedShapeInfo(error);
  Span<DynamicString const> const names = isJoint ? info.jointNames : info.linkNames;
  out.index = FindName(names, partName);
  if (out.index < 0) {
    outError = "Articulated actor '" + std::string(actorName) + "' has no " +
        (isJoint ? "joint" : "link") + " named '" + std::string(partName) + "'. Available: [" +
        JoinNames(names) + "]";
    return false;
  }
  out.link = out.articulated->GetNestedLinkActors(error)[static_cast<size_t>(out.index)];
  return true;
}

Real3 NormalizedOrZero(Real3 const& v) {
  real const norm = mochi::Norm(v);
  return norm > 0 ? v * (1 / norm) : Real3{0.0f, 0.0f, 0.0f};
}

// Of the rotation vectors equivalent to `rotvec` (axis * (angle + 2 pi k)), the one nearest
// `previous`, so the trace stays continuous through +-pi instead of flipping sign.
Real3 UnwrapRotationVector(Real3 const& rotvec, Real3 const& previous) {
  constexpr real kTwoPi = 2 * mochi::kPI;
  // Below this angle the axis is numerical noise, so whole turns are added along the previous axis.
  constexpr real kMinAxisAngle = 1e-3f;
  real const angle = mochi::Norm(rotvec);
  if (angle < kMinAxisAngle) {
    real const turns = std::round(mochi::Norm(previous) / kTwoPi);
    return rotvec + NormalizedOrZero(previous) * (turns * kTwoPi);
  }
  Real3 const axis = rotvec * (1 / angle);
  real const turns = std::round((mochi::Dot(axis, previous) - angle) / kTwoPi);
  return axis * (angle + turns * kTwoPi);
}

// Approximate volume [m^3] by which `actor` overlaps other actors: depth times contact area, summed
// over its overlapping contact points.
real PenetrationVolume(mochi::Scene& scene, Actor const& actor, bool ignoreGround) {
  // The overlap with each other actor can be sampled from either side (or both), so each side is
  // summed separately and the larger kept, rather than counting it twice.
  struct Overlap {
    mochi::ActorHandle other;
    real fromSelf = 0;
    real fromOther = 0;
  };
  DynamicArray<Overlap> overlaps;
  mochi::ActorHandle const self = actor.GetHandle();
  mochi::ErrorLog error;
  for (mochi::ContactPoint const& contact : actor.GetContactPointsWorld(error)) {
    bool const selfIsA = contact.actorA == self;
    mochi::ActorHandle const other = selfIsA ? contact.actorB : contact.actorA;
    if (contact.distance >= 0 || other == self) {
      continue;
    }
    auto it = std::find_if(overlaps.begin(), overlaps.end(), [&](Overlap const& overlap) {
      return overlap.other == other;
    });
    if (it == overlaps.end()) {
      overlaps.push_back({other});
      it = overlaps.end() - 1;
    }
    (selfIsA ? it->fromSelf : it->fromOther) += -contact.distance * contact.intWeight;
  }
  real volume = 0;
  for (Overlap const& overlap : overlaps) {
    Actor const* other = scene.GetActor(overlap.other);
    if (ignoreGround && other != nullptr && other->IsStatic() && !details::IsBounded(*other)) {
      continue;
    }
    volume += std::max(overlap.fromSelf, overlap.fromOther);
  }
  return volume;
}

// A rigid actor, or an articulated link as "actor/link".
Actor* FindBody(mochi::Scene& scene, std::string_view target, std::string& outError) {
  std::string rigidError;
  if (Actor* actor = FindActor(scene, target, ActorType::Rigid, rigidError)) {
    return actor;
  }
  ArticulatedPart part;
  std::string linkError;
  if (target.find('/') == std::string_view::npos ||
      !ResolveArticulatedPart(scene, target, false, part, linkError)) {
    outError = rigidError + (linkError.empty() ? "" : " " + linkError);
    return nullptr;
  }
  return scene.GetActor(part.link);
}

// Total mass of the articulation `body` is a link of, or 0 if it is not a link.
real ArticulationMass(mochi::Scene& scene, Actor const& body) {
  mochi::ErrorLog error;
  for (Actor* actor : details::GetTopLevelActors(scene)) {
    if (actor->GetType() != ActorType::Articulated) {
      continue;
    }
    Span<mochi::ActorHandle const> const links = actor->GetNestedLinkActors(error);
    if (std::find(links.begin(), links.end(), body.GetHandle()) == links.end()) {
      continue;
    }
    real total = 0;
    for (mochi::ActorHandle const link : links) {
      if (Actor const* linkActor = scene.GetActor(link)) {
        total += linkActor->GetMass(error);
      }
    }
    return total;
  }
  return 0;
}

// The load.<name>.drive channel: 0 released, 1 position driven, 2 position and orientation driven.
real DriveChannelValue(LoadSpec const& load, PathDriveMode mode) {
  switch (mode) {
    case PathDriveMode::On:
      return load.drive.rotation ? 2 : 1;
    case PathDriveMode::PositionOnly:
      return 1;
    case PathDriveMode::Off:
      break;
  }
  return 0;
}

} // namespace

int TunerSession::GetOrAddAccumulator(Actor const& actor) {
  mochi::ActorHandle const handle = actor.GetHandle();
  for (size_t i = 0; i < _accumulators.size(); ++i) {
    if (_accumulators[i].actor == handle) {
      return static_cast<int>(i);
    }
  }
  _accumulators.emplace_back();
  Accumulator& accumulator = _accumulators.back();
  accumulator.actor = handle;
  accumulator.forces.resize(static_cast<size_t>(actor.GetNumDofs()), 0);
  return static_cast<int>(_accumulators.size() - 1);
}

void TunerSession::AddChannels(
    std::string_view prefix,
    std::initializer_list<std::string_view> suffixes) {
  for (std::string_view const suffix : suffixes) {
    DynamicString name(prefix);
    name += '.';
    name += suffix;
    _channelNames.push_back(std::move(name));
  }
}

bool TunerSession::ResolveLoad(mochi::Scene& scene, LoadSpec const& load, std::string& outError) {
  _resolvedLoads.emplace_back();
  ResolvedLoad& resolved = _resolvedLoads.back();
  switch (load.type) {
    case LoadType::Force:
    case LoadType::Torque: {
      Actor* body = FindBody(scene, load.target, outError);
      if (body == nullptr) {
        return false;
      }
      if (body->IsStatic()) {
        outError = "'" + std::string(load.target) + "' is static.";
        return false;
      }
      resolved.body = body->GetHandle();
      // A link's load becomes joint forces of its articulated actor, through the link Jacobian.
      resolved.isLink = body->IsNestedLinkActor();
      mochi::ErrorLog error;
      Actor const* owner =
          resolved.isLink ? scene.GetActor(body->GetArticulatedActor(error)) : body;
      resolved.accumulator = GetOrAddAccumulator(*owner);
      return true;
    }
    case LoadType::JointEffort: {
      ArticulatedPart part;
      if (!ResolveArticulatedPart(scene, load.target, true, part, outError)) {
        return false;
      }
      resolved.body = part.link;
      resolved.accumulator = GetOrAddAccumulator(*part.articulated);
      mochi::ErrorLog error;
      auto const& dofInfo =
          part.articulated->GetArticulatedShapeInfo(error).dofInfo[static_cast<size_t>(part.index)];
      resolved.dofOffset = dofInfo.offset;
      resolved.dofCount = dofInfo.transSize + dofInfo.rotSize;
      if (resolved.dofCount < 1 || resolved.dofCount > 3) {
        outError = "Joint '" + std::string(load.target) + "' has " +
            std::to_string(resolved.dofCount) +
            " DoFs; JointEffort supports 1-3. Use Force or Torque on its link instead.";
        return false;
      }
      return true;
    }
    case LoadType::PosePath:
      return ResolvePosePath(scene, load, resolved, outError);
  }
  outError = "Unknown load type.";
  return false;
}

bool TunerSession::ResolvePosePath(
    mochi::Scene& scene,
    LoadSpec const& load,
    ResolvedLoad& resolved,
    std::string& outError) {
  Actor* body = FindBody(scene, load.target, outError);
  if (body == nullptr) {
    return false;
  }
  if (body->IsStatic()) {
    outError = "'" + std::string(load.target) + "' is static.";
    return false;
  }
  if (load.waypoints.empty()) {
    outError = "A pose path needs at least one waypoint.";
    return false;
  }
  resolved.body = body->GetHandle();
  mochi::ErrorLog error;
  DriveScale const scale = MakeDriveScale(
      body->GetMass(error),
      MeanDiagonalInertia(body->GetRigidMomentOfInertiaLocal(error)),
      ArticulationMass(scene, *body));
  resolved.driveMass = scale.mass;
  resolved.driveInertia = scale.inertia;

  // Created at the path's start, inert until the first step sets their gains and targets.
  mochi::TransformRT const start = EvaluatePosePath(load.waypoints, 0.0);
  mochi::RigidPivotPositionConstraintParams position;
  position.actor = resolved.body;
  position.localPosition = load.pointLocal;
  position.targetPosition = start.GetTranslation();
  position.stiffness = 0;
  if (mochi::Constraint* constraint = scene.CreateRigidPivotPositionConstraint(position, error)) {
    // The force query lives on the constraint, so destroying it cancels the query.
    constraint->RegisterQuery(mochi::QueryType::ConstraintForce, error);
    resolved.positionDrive = constraint->GetHandle();
  }
  mochi::RigidPivotRotationConstraintParams rotation;
  rotation.actor = resolved.body;
  rotation.targetRotation = start.GetRotation().ToRotationVector();
  rotation.stiffness = 0;
  if (mochi::Constraint* constraint = scene.CreateRigidPivotRotationConstraint(rotation, error)) {
    resolved.rotationDrive = constraint->GetHandle();
  }
  if (!resolved.positionDrive.IsValid() || !resolved.rotationDrive.IsValid()) {
    scene.DestroyConstraint(resolved.positionDrive);
    scene.DestroyConstraint(resolved.rotationDrive);
    resolved.positionDrive = {};
    resolved.rotationDrive = {};
    outError = "Could not attach the drive to '" + std::string(load.target) + "'.";
    return false;
  }
  resolved.driveGains = {0, 0, -1, 0, 0, -1};
  return true;
}

void TunerSession::DrivePosePath(
    mochi::Scene& scene,
    LoadSpec const& load,
    ResolvedLoad& resolved,
    AppliedLoad& applied,
    double timeStep) {
  mochi::Constraint* position = scene.GetConstraint(resolved.positionDrive);
  mochi::Constraint* rotation = scene.GetConstraint(resolved.rotationDrive);
  if (position == nullptr || rotation == nullptr) {
    return;
  }
  bool const active = load.enabled && !load.waypoints.empty();
  // Mid-step, so a waypoint's mode applies from the step starting at its time despite time drift.
  PathDriveMode const mode =
      active ? EvaluatePathDrive(load.waypoints, _time + 0.5 * timeStep) : PathDriveMode::Off;
  applied.driveMode = mode;
  PathDrive const& drive = load.drive;
  std::array<real, 3> const inert{0, 0, -1};
  std::array<real, 3> const positionGains = mode != PathDriveMode::Off
      ? SpringGains(resolved.driveMass, drive.responseHz, drive.dampingRatio, drive.maxAcceleration)
      : inert;
  std::array<real, 3> const rotationGains = mode == PathDriveMode::On && drive.rotation
      ? SpringGains(
            resolved.driveInertia,
            drive.rotResponseHz,
            drive.rotDampingRatio,
            drive.maxAngularAcceleration)
      : inert;
  std::array<real, 6> const gains{
      positionGains[0],
      positionGains[1],
      positionGains[2],
      rotationGains[0],
      rotationGains[1],
      rotationGains[2]};
  mochi::ErrorLog error;
  if (gains != resolved.driveGains) {
    position->SetStiffness(gains[0], error);
    position->SetDamping(gains[1], error);
    position->SetSaturation(gains[2], error);
    rotation->SetStiffness(gains[3], error);
    rotation->SetDamping(gains[4], error);
    rotation->SetSaturation(gains[5], error);
    resolved.driveGains = gains;
  }
  if (!active) {
    return;
  }
  // The implicit solve pulls toward where the path is at the end of this step; the change in
  // target since the last step also gives the drive the path's velocity. Targets keep moving while
  // the drive is off, so it re-engages at the path's velocity rather than with a jump.
  applied.targetPose = EvaluatePosePath(load.waypoints, _time + timeStep);
  position->SetTargetPosition(applied.targetPose.GetTranslation(), error);
  rotation->SetTargetRotation(applied.targetPose.GetRotation(), error);
}

void TunerSession::SamplePosePath(
    mochi::Scene& scene,
    LoadSpec const& load,
    ResolvedLoad const& resolved,
    AppliedLoad& applied) const {
  Actor const* body = scene.GetActor(resolved.body);
  mochi::Constraint const* position = scene.GetConstraint(resolved.positionDrive);
  if (body == nullptr || position == nullptr) {
    return;
  }
  mochi::TransformRT const worldFromBody = body->GetRootTransform();
  applied.pointWorld = worldFromBody.TransformPoint(load.pointLocal);
  if (!load.enabled || load.waypoints.empty()) {
    applied.targetPose = mochi::TransformRT(worldFromBody.GetRotation(), applied.pointWorld);
    return;
  }
  if (applied.driveMode != PathDriveMode::Off) {
    applied.trackingError = mochi::Norm(applied.targetPose.GetTranslation() - applied.pointWorld);
  }
  if (applied.driveMode == PathDriveMode::On) {
    applied.angleError =
        mochi::Norm((applied.targetPose.GetRotation() * worldFromBody.GetRotation().GetConjugate())
                        .ToRotationVector());
  }
  mochi::ErrorLog error;
  Span<real const> const force = position->GetForce(error);
  if (force.size() >= 3) {
    applied.vectorWorld = Real3{force[0], force[1], force[2]};
    applied.magnitude = mochi::Norm(applied.vectorWorld);
  }
}

bool TunerSession::ResolveProbe(
    mochi::Scene& scene,
    ProbeSpec const& probe,
    std::string& outError) {
  _probes.emplace_back();
  ResolvedProbe& resolved = _probes.back();
  resolved.kind = probe.kind;
  std::string_view const target = probe.target;
  mochi::ErrorLog error;
  switch (probe.kind) {
    case ProbeKind::BodyPose:
    case ProbeKind::BodyVelocity:
    case ProbeKind::ContactForce:
    case ProbeKind::Penetration: {
      Actor* actor = FindBody(scene, target, outError);
      if (actor == nullptr) {
        return false;
      }
      resolved.actor = actor->GetHandle();
      if (probe.kind == ProbeKind::BodyPose) {
        AddChannels(
            target,
            {"pos.x",
             "pos.y",
             "pos.z",
             "rotvec.x",
             "rotvec.y",
             "rotvec.z",
             "rotvecUnwrapped.x",
             "rotvecUnwrapped.y",
             "rotvecUnwrapped.z"});
      } else if (probe.kind == ProbeKind::BodyVelocity) {
        AddChannels(target, {"vel.x", "vel.y", "vel.z", "angvel.x", "angvel.y", "angvel.z"});
      } else if (probe.kind == ProbeKind::Penetration) {
        if (!actor->IsQuerySupported(mochi::QueryType::ContactPoints)) {
          outError = "'" + std::string(target) +
              "' does not report contact points, so its penetration cannot be measured.";
          return false;
        }
        resolved.query = actor->RegisterQuery(mochi::QueryType::ContactPoints, error);
        resolved.ignoreGround = probe.ignoreGround;
        AddChannels(target, {"penetration.volume"});
      } else {
        if (!actor->IsQuerySupported(mochi::QueryType::TotalContactForce)) {
          outError = "'" + std::string(target) + "' does not report contact forces.";
          return false;
        }
        resolved.query = actor->RegisterQuery(mochi::QueryType::TotalContactForce, error);
        AddChannels(target, {"contactForce.x", "contactForce.y", "contactForce.z"});
      }
      return true;
    }
    case ProbeKind::JointState: {
      ArticulatedPart part;
      if (!ResolveArticulatedPart(scene, target, true, part, outError)) {
        return false;
      }
      resolved.actor = part.articulated->GetHandle();
      auto const& dofInfo =
          part.articulated->GetArticulatedShapeInfo(error).dofInfo[static_cast<size_t>(part.index)];
      int const count = dofInfo.transSize + dofInfo.rotSize;
      for (int i = 0; i < count; ++i) {
        resolved.dofIndices.push_back(dofInfo.offset + i);
        std::string const index = std::to_string(i);
        AddChannels(target, {"q" + index, "qd" + index});
      }
      resolved.dofValues.resize(static_cast<size_t>(count), 0);
      resolved.scratch.resize(static_cast<size_t>(part.articulated->GetNumDofs()), 0);
      return true;
    }
    case ProbeKind::SolverStats:
      AddChannels("solver", {"iters", "residual", "lineSearchIters", "convergence"});
      return true;
    case ProbeKind::Performance:
      AddChannels("perf", {"solveMs"});
      return true;
  }
  outError = "Unknown probe kind.";
  return false;
}

bool TunerSession::Bind(mochi::Scene& scene, TunerSpec const& spec, std::string& outError) {
  Unbind(scene);
  _loads = spec.loads;
  for (LoadSpec& load : _loads) {
    SortByTime(load);
  }
  for (LoadSpec const& load : _loads) {
    if (!ResolveLoad(scene, load, outError)) {
      if (!load.enabled) {
        // A disabled load may point at something that cannot take loads (e.g. a body made static by
        // an override). Keep its slot so indices line up; it is never applied.
        _resolvedLoads.back() = {};
        outError.clear();
      } else {
        outError = "Load '" + std::string(load.name) + "': " + outError;
        Unbind(scene);
        return false;
      }
    }
    DynamicString const prefix = DynamicString("load.") + load.name;
    AddChannels(prefix, {"mag"});
    if (load.type == LoadType::PosePath) {
      AddChannels(prefix, {"error", "angleError", "drive"});
    }
  }
  _appliedLoads.resize(_loads.size());
  for (ProbeSpec const& probe : spec.probes) {
    size_t const firstChannel = _channelNames.size();
    if (!ResolveProbe(scene, probe, outError)) {
      outError = "Probe '" + std::string(probe.target) + "': " + outError;
      Unbind(scene);
      return false;
    }
    _probes.back().numChannels = _channelNames.size() - firstChannel;
  }
  // After the probes: threshold conditions refer to their channels.
  for (size_t i = 0; i < spec.stopConditions.size(); ++i) {
    StopCondition const& stop = spec.stopConditions[i];
    if (!stop.enabled) {
      continue;
    }
    if (!ResolveStop(scene, stop, outError)) {
      outError = "Stop condition '" + std::string(stop.name) + "': " + outError;
      Unbind(scene);
      return false;
    }
    _stops.back().specIndex = static_cast<int>(i);
  }
  return true;
}

bool TunerSession::ResolveStop(
    mochi::Scene& scene,
    StopCondition const& stop,
    std::string& outError) {
  ResolvedStop resolved;
  resolved.condition = stop;
  switch (stop.kind) {
    case StopKind::LeavesBox:
    case StopKind::EntersBox: {
      Actor* body = FindBody(scene, stop.target, outError);
      if (body == nullptr) {
        return false;
      }
      resolved.body = body->GetHandle();
      break;
    }
    case StopKind::ChannelThreshold: {
      resolved.channel = FindName(_channelNames, stop.channel);
      if (resolved.channel < 0) {
        outError = "No channel named '" + std::string(stop.channel) +
            "'. Add a probe that produces it. Available: [" + JoinNames(_channelNames) + "]";
        return false;
      }
      break;
    }
  }
  _stops.push_back(std::move(resolved));
  return true;
}

void TunerSession::CheckStopConditions(mochi::Scene& scene, Sample& inOutSample) {
  constexpr double kTimeEpsilon = 1e-9;
  char message[256];
  for (ResolvedStop& stop : _stops) {
    StopCondition const& condition = stop.condition;
    if (_time < condition.start) {
      stop.holdingSince = -1.0;
      continue;
    }
    bool holds = false;
    switch (condition.kind) {
      case StopKind::LeavesBox:
      case StopKind::EntersBox: {
        Actor const* body = scene.GetActor(stop.body);
        if (body == nullptr) {
          break; // the body is gone: the condition cannot hold
        }
        Real3 const p = body->GetRootTransform().GetTranslation();
        bool inside = true;
        for (int d = 0; d < 3; ++d) {
          inside = inside && p[d] >= condition.boxMin[d] && p[d] <= condition.boxMax[d];
        }
        bool const enters = condition.kind == StopKind::EntersBox;
        holds = enters ? inside : !inside;
        if (holds) {
          std::snprintf(
              message,
              sizeof(message),
              "'%s': %s %s the box at (%.3g, %.3g, %.3g)",
              condition.name.c_str(),
              condition.target.c_str(),
              enters ? "entered" : "left",
              static_cast<double>(p[0]),
              static_cast<double>(p[1]),
              static_cast<double>(p[2]));
        }
        break;
      }
      case StopKind::ChannelThreshold: {
        real const value = inOutSample.values[static_cast<size_t>(stop.channel)];
        bool const above = condition.comparison == StopComparison::Above;
        holds = above ? value > condition.threshold : value < condition.threshold;
        if (holds) {
          std::snprintf(
              message,
              sizeof(message),
              "'%s': %s = %.4g %s %.4g",
              condition.name.c_str(),
              condition.channel.c_str(),
              static_cast<double>(value),
              above ? ">" : "<",
              static_cast<double>(condition.threshold));
        }
        break;
      }
    }
    if (!holds) {
      stop.holdingSince = -1.0;
      continue;
    }
    if (stop.holdingSince < 0) {
      stop.holdingSince = _time;
    }
    // Every condition keeps its hold timer up to date; the first triggered in spec order wins.
    if (_time - stop.holdingSince + kTimeEpsilon < condition.holdTime ||
        inOutSample.stopCondition >= 0) {
      continue;
    }
    std::string text = message;
    if (condition.holdTime > 0) {
      char held[48];
      std::snprintf(held, sizeof(held), " for %.3g s", condition.holdTime);
      text += held;
    }
    inOutSample.stopCondition = stop.specIndex;
    inOutSample.stopMessage = DynamicString(text);
  }
}

void TunerSession::Unbind(mochi::Scene& scene) {
  for (Accumulator const& accumulator : _accumulators) {
    if (Actor* actor = scene.GetActor(accumulator.actor)) {
      actor->ClearExternalForces();
    }
  }
  for (ResolvedLoad const& resolved : _resolvedLoads) {
    scene.DestroyConstraint(resolved.positionDrive);
    scene.DestroyConstraint(resolved.rotationDrive);
  }
  for (ResolvedProbe const& probe : _probes) {
    if (probe.query.IsValid()) {
      if (Actor* actor = scene.GetActor(probe.actor)) {
        actor->CancelQuery(probe.query);
      }
    }
  }
  _loads.clear();
  _resolvedLoads.clear();
  _appliedLoads.clear();
  _accumulators.clear();
  _probes.clear();
  _stops.clear();
  _channelNames.clear();
  _time = 0.0;
}

bool TunerSession::UpdateLoads(Span<LoadSpec const> loads, std::string& outError) {
  bool compatible = loads.size() == _loads.size();
  for (size_t i = 0; compatible && i < loads.size(); ++i) {
    // A pose path's drive is attached at its point, so moving the point needs a new run.
    compatible = loads[i].name == _loads[i].name && loads[i].type == _loads[i].type &&
        loads[i].target == _loads[i].target &&
        (loads[i].type != LoadType::PosePath || loads[i].pointLocal == _loads[i].pointLocal);
  }
  if (!compatible) {
    outError =
        "Adding, removing, renaming or retargeting loads, or moving a pose path's point, takes "
        "effect on the next run.";
    return false;
  }
  for (size_t i = 0; i < loads.size(); ++i) {
    _loads[i] = loads[i];
    SortByTime(_loads[i]);
  }
  return true;
}

void TunerSession::OnPreStep(mochi::StepInfo const& info) {
  mochi::Scene& scene = *info.scene;
  for (Accumulator& accumulator : _accumulators) {
    std::fill(accumulator.forces.begin(), accumulator.forces.end(), real{0});
  }
  mochi::ErrorLog error;
  for (size_t i = 0; i < _loads.size(); ++i) {
    LoadSpec const& load = _loads[i];
    ResolvedLoad& resolved = _resolvedLoads[i];
    AppliedLoad& applied = _appliedLoads[i];
    applied = {};
    if (!resolved.body.IsValid()) {
      continue;
    }
    if (load.type == LoadType::PosePath) {
      DrivePosePath(scene, load, resolved, applied, info.timeStepSec);
      continue;
    }
    Actor* body = scene.GetActor(resolved.body);
    if (body == nullptr) {
      continue; // the body is gone
    }
    mochi::TransformRT const worldFromBody = body->GetRootTransform();
    applied.pointWorld = worldFromBody.GetTranslation();
    if (!load.enabled) {
      continue;
    }
    applied.magnitude = EvaluateProfile(load.profile, _time);
    Span<real> const forces =
        mochi::MakeSpan(_accumulators[static_cast<size_t>(resolved.accumulator)].forces);
    Real3 const loadDirection = EvaluateLoadDirection(load, _time);
    Real3 const direction = load.frame == LoadFrame::Local
        ? worldFromBody.TransformDirection(loadDirection)
        : loadDirection;
    switch (load.type) {
      case LoadType::Force: {
        Real3 const force = NormalizedOrZero(direction) * applied.magnitude;
        Real3 const com = body->GetCenterOfMassTransform(error).GetTranslation();
        Real3 const point = EvaluateLoadPoint(load, _time);
        applied.pointWorld =
            load.frame == LoadFrame::WorldPoint ? point : worldFromBody.TransformPoint(point);
        applied.vectorWorld = force;
        if (resolved.isLink) {
          AddLinkForceToGeneralized(
              body->GetArticulatedJacobian(error), force, applied.pointWorld, com, forces);
        } else {
          auto const dofs = RigidPointForceToDofs(force, applied.pointWorld, com);
          for (size_t d = 0; d < dofs.size(); ++d) {
            forces[d] += dofs[d];
          }
        }
        break;
      }
      case LoadType::Torque: {
        applied.vectorWorld = NormalizedOrZero(direction) * applied.magnitude;
        if (resolved.isLink) {
          AddLinkTorqueToGeneralized(
              body->GetArticulatedJacobian(error), applied.vectorWorld, forces);
        } else {
          for (size_t d = 0; d < 3; ++d) {
            forces[3 + d] += applied.vectorWorld[d];
          }
        }
        break;
      }
      case LoadType::JointEffort: {
        for (int d = 0; d < resolved.dofCount; ++d) {
          real const effort = loadDirection[d] * applied.magnitude;
          applied.vectorWorld[d] = effort;
          forces[static_cast<size_t>(resolved.dofOffset + d)] += effort;
        }
        break;
      }
      case LoadType::PosePath:
        break; // driven by its constraints above
    }
  }

  for (Accumulator& accumulator : _accumulators) {
    accumulator.dofs.clear();
    accumulator.values.clear();
    for (size_t d = 0; d < accumulator.forces.size(); ++d) {
      if (accumulator.forces[d] != 0) {
        accumulator.dofs.push_back(static_cast<int>(d));
        accumulator.values.push_back(accumulator.forces[d]);
      }
    }
    Actor* actor = scene.GetActor(accumulator.actor);
    if (actor == nullptr) {
      continue;
    }
    if (accumulator.dofs.empty()) {
      actor->ClearExternalForces();
    } else {
      actor->SetExternalForcesOnDofs(
          mochi::MakeConstSpan(accumulator.dofs), mochi::MakeConstSpan(accumulator.values), error);
    }
  }
}

void TunerSession::OnPostStep(mochi::StepInfo const& info, Sample& outSample) {
  _time += info.timeStepSec;
  mochi::Scene& scene = *info.scene;
  mochi::ErrorLog error;
  outSample.time = _time;
  outSample.values.clear();
  outSample.values.reserve(_channelNames.size());
  auto push3 = [&](Real3 const& v) {
    for (int d = 0; d < 3; ++d) {
      outSample.values.push_back(v[d]);
    }
  };
  for (size_t i = 0; i < _appliedLoads.size(); ++i) {
    AppliedLoad& applied = _appliedLoads[i];
    bool const isPath = _loads[i].type == LoadType::PosePath;
    if (isPath && _resolvedLoads[i].body.IsValid()) {
      SamplePosePath(scene, _loads[i], _resolvedLoads[i], applied);
    }
    outSample.values.push_back(applied.magnitude);
    if (isPath) {
      outSample.values.push_back(applied.trackingError);
      outSample.values.push_back(applied.angleError);
      outSample.values.push_back(DriveChannelValue(_loads[i], applied.driveMode));
    }
  }
  for (ResolvedProbe& probe : _probes) {
    Actor* actor = probe.actor.IsValid() ? scene.GetActor(probe.actor) : nullptr;
    if (probe.actor.IsValid() && actor == nullptr) {
      // The target is gone. Fill its channels with NaN to keep every sample one value per channel;
      // a NaN never trips a threshold.
      for (size_t c = 0; c < probe.numChannels; ++c) {
        outSample.values.push_back(std::numeric_limits<real>::quiet_NaN());
      }
      continue;
    }
    switch (probe.kind) {
      case ProbeKind::BodyPose: {
        mochi::TransformRT const pose = actor->GetRootTransform();
        Real3 const rotvec = pose.GetRotation().ToRotationVector();
        probe.unwrappedRotvec = UnwrapRotationVector(rotvec, probe.unwrappedRotvec);
        push3(pose.GetTranslation());
        push3(rotvec);
        push3(probe.unwrappedRotvec);
        break;
      }
      case ProbeKind::BodyVelocity:
        push3(actor->GetLinearVelocity(error));
        push3(actor->GetAngularVelocity(error));
        break;
      case ProbeKind::ContactForce:
        push3(actor->GetContactForceWorld(error));
        break;
      case ProbeKind::Penetration:
        outSample.values.push_back(PenetrationVolume(scene, *actor, probe.ignoreGround));
        break;
      case ProbeKind::JointState: {
        if (probe.dofIndices.empty()) {
          break; // a Hard joint; GetDofValues reads empty indices as all DoFs
        }
        actor->GetDofValues(
            mochi::MakeConstSpan(probe.dofIndices), mochi::MakeSpan(probe.dofValues), error);
        actor->GetArticulatedJointVelocities(mochi::MakeSpan(probe.scratch), error);
        for (size_t i = 0; i < probe.dofIndices.size(); ++i) {
          outSample.values.push_back(probe.dofValues[i]);
          outSample.values.push_back(probe.scratch[static_cast<size_t>(probe.dofIndices[i])]);
        }
        break;
      }
      case ProbeKind::SolverStats: {
        mochi::SolverStats const stats = scene.GetSolverStats();
        outSample.values.push_back(static_cast<real>(stats.maxNonLinearIters));
        outSample.values.push_back(static_cast<real>(stats.residualNorm));
        outSample.values.push_back(static_cast<real>(stats.maxLineSearchIters));
        outSample.values.push_back(static_cast<real>(stats.convergenceStatus));
        break;
      }
      case ProbeKind::Performance:
        // totalStepDurationSec is only set after the post-step callbacks, so it would lag a step.
        outSample.values.push_back(
            static_cast<real>(scene.GetPerformanceStats().solveStepDurationSec * 1000.0));
        break;
    }
  }
  outSample.loads = _appliedLoads;
  outSample.stopCondition = -1;
  outSample.stopMessage.clear();
  CheckStopConditions(scene, outSample);
}

} // namespace superdex::tuner
