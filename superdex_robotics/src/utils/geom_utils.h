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

#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/utils/span.h>

#include <string>
#include <string_view>
#include <unordered_set>

// Internal helpers shared by the URDF and MJCF loaders for laying out a link's geoms.

namespace superdex::robotics {
using namespace mochi;

/* @brief Returns `name` if free, else `name<suffix>`, `name<suffix>_2`, ..., and records it as
 * taken. */
std::string ClaimUniqueName(
    std::string const& name,
    std::unordered_set<std::string>& taken,
    std::string const& suffix);

/* @brief The geoms of one link, by link name. */
struct LinkGeoms {
  std::string_view link;
  Span<GeomReference const> geoms;
};

/* @brief Assigns each link's geoms to it and to Hard-jointed children, one geom per link, as
 * documented for @ref ExpandMjcfGeomLinks and @ref ExpandUrdfGeomLinks. */
void ExpandGeomLinks(BotPrefab& botPrefab, Span<LinkGeoms const> links, Error& error);

} // namespace superdex::robotics
