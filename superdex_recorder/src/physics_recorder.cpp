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

#include <superdex_recorder/physics_recorder.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace superdex::recorder {
namespace {

constexpr std::string_view kActorArchetype = "superdex.archetypes.PhysicsActor";
constexpr std::string_view kFrameArchetype = "superdex.archetypes.PhysicsFrame";

ComponentField ActorField(std::string component) {
  return {std::string{kActorArchetype}, std::move(component)};
}

// Recorded in this order under superdex.physics.contact_param_names, with the Python field names.
constexpr std::array<std::string_view, 13> kContactParamNames = {
    "penalty_coefficient",
    "penalty_smoothing_half_distance",
    "penalty_threshold_default",
    "penalty_threshold_extra_padding",
    "friction_with_collider_normal",
    "max_alignment_normals",
    "viscous_friction_coefficient",
    "coulomb_friction_coefficient",
    "friction_falloff_vel",
    "normal_viscous_damping_coefficient",
    "distance_error_bound",
    "obj_scale",
    "colliding_penalty_length_scale",
};

// Mirrors mochi::ContactParams so that a new field fails to compile until it is recorded.
struct RecordedContactParamsLayout {
  mochi::real penaltyCoefficient;
  mochi::real penaltySmoothingHalfDistance;
  mochi::real penaltyThresholdDefault;
  mochi::real penaltyThresholdExtraPadding;
  bool frictionWithColliderNormal;
  mochi::real maxAlignmentNormals;
  mochi::real viscousFrictionCoefficient;
  mochi::real coulombFrictionCoefficient;
  mochi::real frictionFalloffVel;
  mochi::real normalViscousDampingCoefficient;
  mochi::real distanceErrorBound;
  mochi::real objScale;
  mochi::real collidingPenaltyLengthScale;
};
static_assert(
    sizeof(mochi::ContactParams) == sizeof(RecordedContactParamsLayout),
    "mochi::ContactParams changed; update kContactParamNames and ContactParamValues.");

std::array<double, kContactParamNames.size()> ContactParamValues(mochi::ContactParams const& p) {
  return {
      p.penaltyCoefficient,
      p.penaltySmoothingHalfDistance,
      p.penaltyThresholdDefault,
      p.penaltyThresholdExtraPadding,
      p.frictionWithColliderNormal ? 1.0 : 0.0,
      p.maxAlignmentNormals,
      p.viscousFrictionCoefficient,
      p.coulombFrictionCoefficient,
      p.frictionFalloffVel,
      p.normalViscousDampingCoefficient,
      p.distanceErrorBound,
      p.objScale,
      p.collidingPenaltyLengthScale,
  };
}

ComponentField FrameField(std::string component) {
  return {std::string{kFrameArchetype}, std::move(component)};
}

std::string_view ActorTypeName(mochi::ActorType type) {
  return type == mochi::ActorType::Rigid ? "rigid" : "articulated";
}

std::string_view JointTypeName(mochi::ArticulatedJointType type) {
  switch (type) {
    case mochi::ArticulatedJointType::Free:
      return "free";
    case mochi::ArticulatedJointType::Prismatic:
      return "prismatic";
    case mochi::ArticulatedJointType::Revolute:
      return "revolute";
    case mochi::ArticulatedJointType::Spherical:
      return "spherical";
    case mochi::ArticulatedJointType::Hard:
      return "hard";
    case mochi::ArticulatedJointType::Cycle:
      return "cycle";
    case mochi::ArticulatedJointType::Count:
      return "invalid";
  }
  return "invalid";
}

template <typename Values>
mochi::Span<mochi::real const> AsSpan(Values const& values) {
  return {values.data(), static_cast<size_t>(values.size())};
}

std::array<uint8_t, 4> ColorForId(std::string_view id) {
  uint32_t hash = 2166136261U;
  for (unsigned char const character : id) {
    hash = (hash ^ character) * 16777619U;
  }
  // Mix related IDs before interpreting the hash bytes as RGB.
  hash ^= hash >> 16;
  hash *= 0x85ebca6bU;
  hash ^= hash >> 13;
  hash *= 0xc2b2ae35U;
  hash ^= hash >> 16;
  return {
      static_cast<uint8_t>(96U + (hash & 0x7fU)),
      static_cast<uint8_t>(96U + ((hash >> 8) & 0x7fU)),
      static_cast<uint8_t>(96U + ((hash >> 16) & 0x7fU)),
      255U};
}

std::string UniqueActorId(mochi::Actor const& actor, std::unordered_set<std::string>& ids) {
  char const* const name = actor.GetName();
  std::string const base = name == nullptr || *name == '\0' ? "actor" : name;
  std::string id = base;
  for (uint32_t suffix = 1; !ids.insert(id).second; ++suffix) {
    id = base + "_" + std::to_string(suffix);
  }
  return id;
}

} // namespace

PhysicsRecorder::Link::Link(std::string name, TransformSeries& transformSeries)
    : name{std::move(name)}, transformSeries{transformSeries} {}

PhysicsRecorder::Actor::Actor(
    std::string id,
    std::string dataPath,
    mochi::Actor& actor,
    ScalarSeries& stateSeries,
    TransformSeries& transformSeries)
    : id{std::move(id)},
      dataPath{std::move(dataPath)},
      actor{actor},
      stateSeries{stateSeries},
      transformSeries{transformSeries} {}

PhysicsRecorder::PhysicsRecorder(Episode& episode, mochi::Scene& scene, mochi::Error& error)
    : _episode{episode}, _scene{scene} {
  _frameSeries = _episode.CreateScalarSeries(
      "data/physics/frames",
      {
          FrameField("superdex.physics.step_dt_sec"),
      },
      error);
  std::vector<std::string> const contactParamNames{
      kContactParamNames.begin(), kContactParamNames.end()};
  _episode.WriteConstantTexts(
      "data/physics/actors",
      ActorField("superdex.physics.contact_param_names"),
      contactParamNames,
      error);
  MOCHI_ERROR_RETURN(error);

  std::unordered_set<std::string> ids;
  scene.ForEachActor([&](mochi::Actor* actor) {
    if (!error.IsOK() || actor == nullptr || actor->IsNestedLinkActor() ||
        actor->IsNestedSoftActor()) {
      return;
    }
    if (actor->GetType() != mochi::ActorType::Rigid &&
        actor->GetType() != mochi::ActorType::Articulated) {
      return;
    }
    RegisterActor(UniqueActorId(*actor, ids), *actor, error);
  });
}

PhysicsRecorder::~PhysicsRecorder() = default;

void PhysicsRecorder::RegisterActor(std::string id, mochi::Actor& actor, mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  std::string const encodedId = Episode::EncodeEntityPathSegment(id);
  std::string const dataPath = "data/physics/actors/" + encodedId;
  std::string const entityPath = "world/" + encodedId;

  bool const hasPoseController =
      actor.GetType() == mochi::ActorType::Articulated && actor.HasArticulatedPoseController(error);
  MOCHI_ERROR_RETURN(error);
  auto* const transformSeries = _episode.CreateTransformSeries(entityPath, error);
  std::vector<ComponentField> fields{
      ActorField("superdex.physics.root_translation_xyz"),
      ActorField("superdex.physics.root_quaternion_xyzw"),
  };
  if (actor.GetType() == mochi::ActorType::Rigid) {
    fields.push_back(ActorField("superdex.physics.linear_velocity_xyz"));
    fields.push_back(ActorField("superdex.physics.angular_velocity_xyz"));
    if (!actor.IsStatic()) {
      fields.push_back(ActorField("superdex.physics.com_translation_xyz"));
    }
  } else {
    fields.push_back(ActorField("superdex.physics.joint_positions"));
    fields.push_back(ActorField("superdex.physics.joint_velocities"));
    if (hasPoseController) {
      fields.push_back(ActorField("superdex.physics.controller_target_positions"));
    }
  }
  auto* const stateSeries = _episode.CreateScalarSeries(dataPath, std::move(fields), error);
  MOCHI_ERROR_RETURN(error);
  if (transformSeries == nullptr || stateSeries == nullptr) {
    MOCHI_ERROR_SET(error, "Failed to create actor recording series.");
    return;
  }

  Actor registered{std::move(id), dataPath, actor, *stateSeries, *transformSeries};

  if (actor.GetType() == mochi::ActorType::Articulated) {
    auto const linkHandles = actor.GetNestedLinkActors(error);
    auto const shapeInfo = actor.GetArticulatedShapeInfo(error);
    MOCHI_ERROR_RETURN(error);
    MOCHI_ERROR_IF_NOT(
        linkHandles.size() == shapeInfo.linkNames.size(),
        error,
        "Articulated actor link handles and names have inconsistent sizes.");
    MOCHI_ERROR_RETURN(error);

    registered.links.reserve(linkHandles.size());
    for (size_t i = 0; i < linkHandles.size(); ++i) {
      mochi::Actor* const linkActor = _scene.GetActor(linkHandles[i]);
      if (linkActor == nullptr) {
        MOCHI_ERROR_SET(error, "Articulated actor has an invalid link.");
        return;
      }
      std::string const linkName = std::string{shapeInfo.linkNames[i]};
      std::string const linkPath =
          entityPath + "/links/" + Episode::EncodeEntityPathSegment(linkName);
      auto* const linkSeries = _episode.CreateTransformSeries(linkPath, error);
      MOCHI_ERROR_RETURN(error);
      if (linkSeries == nullptr) {
        MOCHI_ERROR_SET(error, "Failed to create link recording series.");
        return;
      }
      _episode.WriteConstantText(
          linkPath, ActorField("superdex.physics.link_name"), linkName, error);
      WriteGeometry(linkPath, *linkActor, error);
      WritePhysicalParameters(linkPath, *linkActor, error);
      MOCHI_ERROR_RETURN(error);
      registered.links.emplace_back(linkName, *linkSeries);
    }

    // A skinned articulation keeps its contact skin's parameters on the root actor. The root's
    // mass and density aren't written: they describe the whole articulation, not one body.
    mochi::Error skinContactError;
    mochi::ContactParams const skinContact = actor.GetContactParams(skinContactError);
    if (skinContactError.IsOK()) {
      _episode.WriteConstantScalars(
          dataPath,
          ActorField("superdex.physics.contact_params"),
          ContactParamValues(skinContact),
          error);
    }

    MOCHI_ERROR_IF_NOT(
        shapeInfo.jointNames.size() == shapeInfo.jointTypes.size() &&
            shapeInfo.jointNames.size() == shapeInfo.dofInfo.size(),
        error,
        "Articulated actor joint metadata has inconsistent sizes.");
    MOCHI_ERROR_RETURN(error);

    std::vector<std::string> jointNames;
    std::vector<std::string> jointTypes;
    std::vector<double> jointDofOffsets;
    std::vector<double> jointTranslationDofCounts;
    std::vector<double> jointRotationDofCounts;
    jointNames.reserve(shapeInfo.jointNames.size());
    jointTypes.reserve(shapeInfo.jointTypes.size());
    jointDofOffsets.reserve(shapeInfo.dofInfo.size());
    jointTranslationDofCounts.reserve(shapeInfo.dofInfo.size());
    jointRotationDofCounts.reserve(shapeInfo.dofInfo.size());
    for (size_t i = 0; i < shapeInfo.jointNames.size(); ++i) {
      jointNames.emplace_back(shapeInfo.jointNames[i]);
      jointTypes.emplace_back(JointTypeName(shapeInfo.jointTypes[i]));
      jointDofOffsets.push_back(static_cast<double>(shapeInfo.dofInfo[i].offset));
      jointTranslationDofCounts.push_back(static_cast<double>(shapeInfo.dofInfo[i].transSize));
      jointRotationDofCounts.push_back(static_cast<double>(shapeInfo.dofInfo[i].rotSize));
    }
    if (!jointNames.empty()) {
      std::string const jointLayoutPath = dataPath + "/joint_layout";
      _episode.WriteConstantTexts(
          jointLayoutPath, ActorField("superdex.physics.joint_names"), jointNames, error);
      _episode.WriteConstantTexts(
          jointLayoutPath, ActorField("superdex.physics.joint_types"), jointTypes, error);
      _episode.WriteConstantScalars(
          jointLayoutPath,
          ActorField("superdex.physics.joint_dof_offsets"),
          jointDofOffsets,
          error);
      _episode.WriteConstantScalars(
          jointLayoutPath,
          ActorField("superdex.physics.joint_translation_dof_counts"),
          jointTranslationDofCounts,
          error);
      _episode.WriteConstantScalars(
          jointLayoutPath,
          ActorField("superdex.physics.joint_rotation_dof_counts"),
          jointRotationDofCounts,
          error);
      MOCHI_ERROR_RETURN(error);
    }

    int const numDofs = actor.GetNumDofs();
    registered.positions.resize(numDofs);
    registered.velocities.resize(numDofs);
    registered.hasPoseController = hasPoseController;
    if (hasPoseController) {
      registered.controllerTargets.resize(numDofs);
    }
    registered.worldFromLinks.resize(registered.links.size());
  } else {
    WriteGeometry(entityPath, actor, error);
    WritePhysicalParameters(dataPath, actor, error);
  }
  MOCHI_ERROR_RETURN(error);

  char const* const runtimeName = actor.GetName();
  _episode.WriteConstantText(
      dataPath, ActorField("superdex.physics.actor_id"), registered.id, error);
  _episode.WriteConstantText(
      dataPath,
      ActorField("superdex.physics.runtime_name"),
      runtimeName == nullptr ? "" : runtimeName,
      error);
  _episode.WriteConstantText(
      dataPath, ActorField("superdex.physics.actor_type"), ActorTypeName(actor.GetType()), error);
  std::array<double, 1> const isStatic{actor.IsStatic() ? 1.0 : 0.0};
  std::array<double, 1> const numDofs{static_cast<double>(actor.GetNumDofs())};
  _episode.WriteConstantScalars(
      dataPath, ActorField("superdex.physics.is_static"), isStatic, error);
  _episode.WriteConstantScalars(dataPath, ActorField("superdex.physics.num_dofs"), numDofs, error);
  MOCHI_ERROR_RETURN(error);
  _actors.push_back(std::move(registered));
}

void PhysicsRecorder::WritePhysicalParameters(
    std::string_view path,
    mochi::Actor const& actor,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  // Links without a collider are massless frames with nothing to reapply, and static bodies have
  // no mass.
  mochi::Error contactError;
  mochi::ContactParams const contact = actor.GetContactParams(contactError);
  if (!contactError.IsOK()) {
    return;
  }
  _episode.WriteConstantScalars(
      path, ActorField("superdex.physics.contact_params"), ContactParamValues(contact), error);
  if (actor.IsStatic()) {
    return;
  }
  mochi::Error massError;
  std::array<double, 1> const density{static_cast<double>(actor.GetDensity(massError))};
  std::array<double, 1> const mass{static_cast<double>(actor.GetMass(massError))};
  if (massError.IsOK()) {
    _episode.WriteConstantScalars(path, ActorField("superdex.physics.density"), density, error);
    _episode.WriteConstantScalars(path, ActorField("superdex.physics.mass"), mass, error);
  }
}

void PhysicsRecorder::WriteGeometry(
    std::string_view entityPath,
    mochi::Actor const& actor,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto const mesh = actor.GetSurfaceMesh();
  if (mesh.nodesPerElement != 3 || mesh.coordinates.empty() || mesh.connectivity.empty()) {
    return;
  }

  std::vector<float> vertices;
  vertices.reserve(mesh.coordinates.size());
  std::transform(
      mesh.coordinates.begin(),
      mesh.coordinates.end(),
      std::back_inserter(vertices),
      [](mochi::real value) { return static_cast<float>(value); });

  std::vector<uint32_t> triangles;
  triangles.reserve(mesh.connectivity.size());
  MOCHI_ERROR_IF_NOT(
      std::all_of(
          mesh.connectivity.begin(), mesh.connectivity.end(), [](int index) { return index >= 0; }),
      error,
      "Physics mesh contains a negative vertex index.");
  MOCHI_ERROR_RETURN(error);
  std::transform(
      mesh.connectivity.begin(),
      mesh.connectivity.end(),
      std::back_inserter(triangles),
      [](int index) { return static_cast<uint32_t>(index); });
  _episode.WriteMesh(entityPath, vertices, triangles, ColorForId(entityPath), error);
}

void PhysicsRecorder::Record(SampleTime time, mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (_frameSeries == nullptr) {
    MOCHI_ERROR_SET(error, "Physics frame series is unavailable.");
    return;
  }
  std::array<mochi::real, 1> const stepDuration{static_cast<mochi::real>(_scene.GetLastTimeStep())};
  std::array<mochi::Span<mochi::real const>, 1> const frameFields{AsSpan(stepDuration)};
  _frameSeries->Append(time, frameFields, error);
  MOCHI_ERROR_RETURN(error);

  for (auto& registered : _actors) {
    mochi::Actor& actor = registered.actor;
    if (actor.IsStatic() && registered.stateSeries.GetNumSamples() != 0) {
      continue;
    }

    auto const worldFromActor = actor.GetRootTransform();
    auto const rootTranslation = worldFromActor.GetTranslation();
    auto const& rootRotation = worldFromActor.GetRotation();
    std::array<mochi::real, 3> const translation{
        rootTranslation[0], rootTranslation[1], rootTranslation[2]};
    std::array<mochi::real, 4> const quaternion{
        rootRotation.data[0], rootRotation.data[1], rootRotation.data[2], rootRotation.data[3]};
    registered.transformSeries.Append(time, worldFromActor, error);
    MOCHI_ERROR_RETURN(error);

    if (actor.GetType() == mochi::ActorType::Rigid) {
      auto const linearVelocity = actor.GetLinearVelocity(error);
      auto const angularVelocity = actor.GetAngularVelocity(error);
      // The root transform is rounded from the simulated center-of-mass transform, so restoring
      // the state exactly needs the center of mass itself.
      mochi::Real3 comTranslation;
      if (!actor.IsStatic()) {
        comTranslation = actor.GetCenterOfMassTransform(error).GetTranslation();
      }
      MOCHI_ERROR_RETURN(error);
      std::array<mochi::Span<mochi::real const>, 5> const fields{
          AsSpan(translation),
          AsSpan(quaternion),
          AsSpan(linearVelocity),
          AsSpan(angularVelocity),
          AsSpan(comTranslation),
      };
      size_t const numFields = actor.IsStatic() ? fields.size() - 1 : fields.size();
      registered.stateSeries.Append(
          time, mochi::Span<mochi::Span<mochi::real const> const>(fields.data(), numFields), error);
      MOCHI_ERROR_RETURN(error);
      continue;
    }

    actor.GetArticulatedPose(registered.positions, error);
    actor.GetArticulatedJointVelocities(registered.velocities, error);
    MOCHI_ERROR_RETURN(error);
    if (registered.hasPoseController) {
      actor.GetArticulatedTargetPose(registered.controllerTargets, error);
      MOCHI_ERROR_RETURN(error);
    }
    std::array<mochi::Span<mochi::real const>, 5> const fields{
        AsSpan(translation),
        AsSpan(quaternion),
        AsSpan(registered.positions),
        AsSpan(registered.velocities),
        AsSpan(registered.controllerTargets),
    };
    size_t const numFields = registered.hasPoseController ? fields.size() : fields.size() - 1;
    registered.stateSeries.Append(
        time, mochi::Span<mochi::Span<mochi::real const> const>(fields.data(), numFields), error);
    MOCHI_ERROR_RETURN(error);

    actor.GetArticulatedLinkTransforms(registered.worldFromLinks, error);
    MOCHI_ERROR_RETURN(error);
    mochi::TransformRT const actorFromWorld = mochi::Invert(worldFromActor);
    for (size_t i = 0; i < registered.links.size(); ++i) {
      mochi::TransformRT const actorFromLink = actorFromWorld * registered.worldFromLinks[i];
      registered.links[i].transformSeries.Append(time, actorFromLink, error);
      MOCHI_ERROR_RETURN(error);
    }
  }
}

} // namespace superdex::recorder
