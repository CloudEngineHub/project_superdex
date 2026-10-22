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

#include <mochi_physics/mochi_physics.h>

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/span.h>

#include <limits>

// Builders of articulated actors shared by tests.
namespace mochi::test {

/**************************************************************************************************
  Three-bone skinned actors: a Hard -> Spherical -> Spherical skeleton of gravity-free unit cubes
  without colliders, with links offset by 1 along x and y, and a unit-cube skin or nested soft actor
  skinned to the three bones.
*/

// Physical parameters of three-bone actors. The defaults (massless links, and a soft actor without
// inertia and stiffness) suit residual assembly tests, but make stepping ill-posed.
struct ThreeBoneOptions {
  real linkDensity = 1e-10_r;
  real softYoungsModulus = 1e-10_r;
  bool softHasInertia = false;
};

// Shape of a unit tet cube skinned to three bones. Bone 0 is static.
// weight on bone 2, so the samples of the cube face y = 1 depend on bone 1 only, and the samples of
// other faces depend on bones 1 and 2. If blendedSoftName is non-empty, the nodes with
// x < blendedMaxX are blended with the soft actor of that name.
ShapeHandle CreateThreeBoneShape(
    Context* context,
    DynamicArray<int> constrainedNodes = {},
    DynamicString const& blendedSoftName = {},
    real blendedMaxX = std::numeric_limits<real>::max());

// Linear ROM version of CreateThreeBoneShape, with node 0 constrained and a single mode that moves
// all other nodes along x.
ShapeHandle CreateThreeBoneTetRomShape(Context* context);

ArticulatedActorParams MakeThreeBoneSkeletonParams(
    Context* context,
    ThreeBoneOptions const& options = {});

// An articulated actor with the skeleton of MakeThreeBoneSkeletonParams and the skin of
// CreateThreeBoneShape in the "Colliding" layer.
Actor* CreateThreeBoneSkinnedActor(
    Context* context,
    Scene* scene,
    real coulombCoefficient,
    real viscousCoefficient,
    real dampingCoefficient,
    ActorBoundaryElementType boundaryElementType,
    ThreeBoneOptions const& options = {});

// Variants of a soft-skinned actor with the skeleton of MakeThreeBoneSkeletonParams and a nested
// soft actor named "soft":
// - Fom, Rom: the soft actor (shape of CreateThreeBoneShape or CreateThreeBoneTetRomShape) collides
//   in the "Colliding" layer.
// - BlendedFom, BlendedRom: the soft actor does not collide. Instead, the articulated actor has a
//   colliding skin blended with it.
// - PartiallyBlendedFom: like BlendedFom, but only the skin nodes with x < 0.5 are blended.
// - PosedFom: like Fom, but with posed (world-space) gravity and stress of the soft actor.
enum class ThreeBoneSoftSkinnedMode {
  Fom,
  Rom,
  BlendedFom,
  BlendedRom,
  PartiallyBlendedFom,
  PosedFom
};

[[nodiscard]] inline bool UsesRom(ThreeBoneSoftSkinnedMode mode) {
  return mode == ThreeBoneSoftSkinnedMode::Rom || mode == ThreeBoneSoftSkinnedMode::BlendedRom;
}

[[nodiscard]] inline bool UsesBlendedSurface(ThreeBoneSoftSkinnedMode mode) {
  return mode == ThreeBoneSoftSkinnedMode::BlendedFom ||
      mode == ThreeBoneSoftSkinnedMode::BlendedRom ||
      mode == ThreeBoneSoftSkinnedMode::PartiallyBlendedFom;
}

// Creates the soft-skinned actor of the given mode. The soft actor is constrained at node 0.
// Contact coefficients apply to the colliding surface.
Actor* CreateThreeBoneSoftSkinnedActor(
    Context* context,
    Scene* scene,
    ThreeBoneSoftSkinnedMode mode,
    TransformRT const& worldFromRoot,
    real coulombCoefficient,
    real viscousCoefficient,
    real dampingCoefficient,
    ActorBoundaryElementType boundaryElementType,
    ThreeBoneOptions const& options = {});

// A box-colliding rigid unit cube in the "Collider" layer, overlapping the face x = 1 of the
// three-bone skin, whose samples depend on bones 1 and 2.
Actor* AddThreeBoneSkinCollider(Scene* scene, bool isStatic);

/**************************************************************************************************
  Chains of unit cubes
*/

// A chain of box-colliding unit cubes in the "Articulated" layer, with the given joint types, all
// with axis x. Joint 0 is offset by (1, 1, 1) from the root, and the other links are co-located.
// Non-free joints get the given inertia, viscous damping and Coulomb friction, if positive.
Actor* CreateUnitCubeChain(
    Scene* scene,
    ShapeHandle cubeShape,
    Span<ArticulatedJointType const> joints,
    real dampingCoeff,
    real inertiaCoeff,
    real frictionCoeff);

// A revolute joint, about z by default.
ArticulatedJointParams MakeRevoluteJoint(
    Real3 const& parentLinkFromJoint,
    ArticulatedJointFrictionParams const& friction = {},
    Real3 const& axis = Real3{0_r, 0_r, 1_r});

struct CubeTreeOptions {
  TransformRT worldFromRoot{};
  // Parent of each link. If empty, link i is the child of link i - 1.
  DynamicArray<int> parents{};
  // Transform of each link in its joint frame. If empty, identity.
  DynamicArray<TransformRT> parentJointFromLinks{};
  // Link without a shape, if any.
  int dummyLink = -1;
  ColliderType colliderType = ColliderType::None;
};

// An articulated actor of unit cubes, i.e. [0, 1]³ in their link frames, with one link per joint.
Actor* CreateCubeTree(
    Scene* scene,
    DynamicArray<ArticulatedJointParams> joints,
    CubeTreeOptions const& options = {});

// A Hard-rooted double pendulum of unit cubes without colliders, with revolute joints about z.
Actor* CreateDoublePendulum(Scene* scene, ArticulatedJointFrictionParams const& friction = {});

// A Hard-rooted chain of unit cubes without colliders, whose joints (revolute about the vertical,
// Hard, and prismatic along a horizontal axis) can only move orthogonally to gravity, so it stays
// static at rest. Link 2 is welded to link 1 and holds link 3, so a dummy link 2 transmits a load
// without adding a DoF.
Actor* CreateStaticChain(Scene* scene, int dummyLink = -1);

/**************************************************************************************************
  Articulated actors with link loads. Links are unit cubes, i.e. [0, 1]³ in their local frame, and
  revolute joints get viscous damping `jointDamping`.
*/

// Static ground plane through the origin, with normal +y.
Actor* AddFlatGround(Scene* scene);

// A rigid unit cube, i.e. [0, 1]³ in its local frame.
Actor* AddRigidUnitCube(
    Scene* scene,
    TransformRT const& worldFromLocal,
    ColliderType colliderType,
    bool isStatic = false);

// A free-floating bridge on the ground plane of AddFlatGround. Its end links are box-colliding and
// rest on the ground, and its middle link has no collider and spans them 0.5 above the ground. The
// two joints are revolute about the vertical, so that the bridge is static and its joints carry the
// weight of the middle link.
Actor* CreateBridgeOnGround(Scene* scene, real jointDamping);

// An actor whose links collide with each other. A static (Hard-rooted) box link 0 supports box
// link 2, which hangs from a revolute arm (link 1, no collider). Links 0 and 2 are not adjacent, so
// their contact is not filtered.
Actor* CreateSelfContactArm(Scene* scene, real jointDamping);

// A parallelogram linkage without colliders: a static (Hard-rooted) link 0, and links 1 to 3
// connected by revolute joints at the corners of a 2 x 2 square in the xy plane. At rest, link 3
// meets link 0 at the origin. If useCycleJoint, a cycle joint closes the loop there.
Actor* CreateParallelogramLinkage(Scene* scene, bool useCycleJoint, real jointDamping);

// A spherical joint constraint between two rigid actors (or links) at a world point, from their
// current root transforms.
Constraint* CreateRigidSphericalJointAt(
    Scene* scene,
    Actor const* actorA,
    Actor const* actorB,
    Real3 const& worldPoint,
    real stiffness);

} // namespace mochi::test
