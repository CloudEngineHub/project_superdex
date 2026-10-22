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

#include "geom_utils.h"

#include <superdex_robotics/utils/bot_utils.h>

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/transform_rt.h>

#include <string>
#include <unordered_set>

using namespace mochi;
using namespace superdex::robotics;

namespace {

constexpr real kCollisionChildMassFraction = 1e-6_r;
constexpr real kCollisionChildMinMass = 1e-9_r;

// Pose of a geom's file data in the link frame, in the shape-bake form R * (scale * v) + t.
TransformRT LinkFromMesh(GeomReference const& geom) {
  return geom.linkFromGeom * geom.geomFromMesh;
}

void AssignShape(BotLinkPrefab& link, GeomReference const& geom) {
  TransformRT const linkFromMesh = LinkFromMesh(geom);
  link.shapeFile = geom.shapeFile;
  link.shapeScale = geom.meshScale;
  link.shapeRotation = linkFromMesh.GetRotation();
  link.shapeTranslation = linkFromMesh.GetTranslation();
}

void AssignRenderModel(BotLinkPrefab& link, GeomReference const& geom) {
  TransformRT const linkFromMesh = LinkFromMesh(geom);
  link.renderModelFile = geom.renderModelFile;
  link.renderModelScale = geom.meshScale;
  link.renderModelRotation = linkFromMesh.GetRotation();
  link.renderModelTranslation = linkFromMesh.GetTranslation();
}

// Mass from a geom's own mass or density, for links without an explicit inertia. A geom with
// neither leaves Mochi's default density.
void AssignGeomMass(BotLinkPrefab& link, GeomReference const& geom) {
  if (geom.mass && *geom.mass > 0_r) {
    link.mass = *geom.mass;
  } else if (geom.density && *geom.density > 0_r) {
    link.density = *geom.density;
  }
}

bool IsAssignableCollision(GeomReference const& geom) {
  return geom.isCollision && !geom.shapeFile.empty() && geom.type != GeomType::Plane &&
      geom.type != GeomType::Hfield && geom.type != GeomType::Sdf;
}

} // namespace

std::string superdex::robotics::ClaimUniqueName(
    std::string const& name,
    std::unordered_set<std::string>& taken,
    std::string const& suffix) {
  if (taken.insert(name).second) {
    return name;
  }
  std::string candidate = name + suffix;
  for (int i = 2; !taken.insert(candidate).second; ++i) {
    candidate = name + suffix + "_" + std::to_string(i);
  }
  return candidate;
}

void superdex::robotics::ExpandGeomLinks(
    BotPrefab& botPrefab,
    Span<LinkGeoms const> links,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  // Links and joints share one namespace in a BotPrefab.
  std::unordered_set<std::string> names;
  for (auto const& link : botPrefab.links) {
    names.insert(std::string(link.name));
  }
  for (auto const& joint : botPrefab.joints) {
    names.insert(std::string(joint.name));
  }
  auto findLink = [&](std::string_view name) {
    for (int i = 0; i < isize(botPrefab.links); ++i) {
      if (std::string_view(botPrefab.links[i].name) == name) {
        return i;
      }
    }
    return static_cast<int>(kIndexNone);
  };
  auto addChild = [&](int parent, std::string const& name) -> BotLinkPrefab& {
    BotLinkPrefab child;
    child.name = DynamicString{ClaimUniqueName(name, names, "_link")};
    child.parentLink = parent;
    BotJointPrefab joint;
    joint.name =
        DynamicString{ClaimUniqueName(std::string(child.name) + "_fixed", names, "_joint")};
    joint.type = ArticulatedJointType::Hard;
    botPrefab.links.push_back(std::move(child));
    botPrefab.joints.push_back(std::move(joint));
    return botPrefab.links.back();
  };

  for (LinkGeoms const& entry : links) {
    int const parent = findLink(entry.link);
    MOCHI_ERROR_IF(parent == kIndexNone, error, "Geom link not found in the bot");
    MOCHI_ERROR_RETURN(error);
    bool const hasExplicitInertia = botPrefab.links[parent].mass.has_value() &&
        botPrefab.links[parent].momentOfInertia.has_value();
    real const childMass =
        Max(kCollisionChildMinMass,
            kCollisionChildMassFraction * botPrefab.links[parent].mass.value_or(0_r));
    std::string const linkName(entry.link);

    std::unordered_set<int> renderedGeoms;
    int numCollisions = 0;
    for (int g = 0; g < isize(entry.geoms); ++g) {
      GeomReference const& geom = entry.geoms[g];
      if (!IsAssignableCollision(geom)) {
        continue;
      }
      bool const alsoRender = geom.isVisual && !geom.renderModelFile.empty();
      if (numCollisions == 0) {
        BotLinkPrefab& link = botPrefab.links[parent];
        AssignShape(link, geom);
        if (!hasExplicitInertia) {
          AssignGeomMass(link, geom);
        }
        if (alsoRender && link.renderModelFile.empty()) {
          AssignRenderModel(link, geom);
          renderedGeoms.insert(g);
        }
      } else {
        BotLinkPrefab& child =
            addChild(parent, linkName + "_collision_" + std::to_string(numCollisions));
        AssignShape(child, geom);
        if (hasExplicitInertia) {
          child.mass = childMass;
        } else {
          AssignGeomMass(child, geom);
        }
        if (alsoRender) {
          AssignRenderModel(child, geom);
          renderedGeoms.insert(g);
        }
      }
      ++numCollisions;
    }

    int numVisuals = 0;
    for (int g = 0; g < isize(entry.geoms); ++g) {
      GeomReference const& geom = entry.geoms[g];
      if (!geom.isVisual || geom.renderModelFile.empty() || renderedGeoms.count(g) > 0) {
        continue;
      }
      if (botPrefab.links[parent].renderModelFile.empty()) {
        AssignRenderModel(botPrefab.links[parent], geom);
      } else {
        ++numVisuals;
        AssignRenderModel(
            addChild(parent, linkName + "_visual_" + std::to_string(numVisuals)), geom);
      }
    }
  }
  RebuildBotData(botPrefab, error);
}
