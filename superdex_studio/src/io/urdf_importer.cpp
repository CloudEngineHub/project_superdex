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

#include "io/urdf_importer.h"

#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/utils/error.h>

#include <string>

namespace superdex::studio {

UrdfImporter::UrdfImporter(SuperDexStudio* studio)
    : BotImporter(
          studio,
          Format{
              .displayName = "URDF",
              .extensions = {std::string(superdex::robotics::kUrdfExtension)},
              .allowsCopy = true,
              .collisionFromVisuals = true}) {}

bool UrdfImporter::LoadSource(mochi::Path const& path, std::vector<std::string>& notes) {
  mochi::ErrorLog error;
  _info = {};
  _botPrefab = superdex::robotics::LoadBotPrefabFromUrdfFile(path.ToString(), _info, error);
  for (auto const& warning : _info.warnings) {
    notes.emplace_back(warning);
  }
  return error.IsOK();
}

int UrdfImporter::NumSourceLinks() const {
  return static_cast<int>(_info.links.size());
}

std::string UrdfImporter::SourceLinkName(int source) const {
  return std::string(_info.links[source].link);
}

mochi::DynamicArray<superdex::robotics::GeomReference>& UrdfImporter::SourceGeoms(int source) {
  return _info.links[source].geoms;
}

void UrdfImporter::FinishPrefab(bool anySeparate, mochi::Error& error) {
  if (anySeparate) {
    superdex::robotics::ExpandUrdfGeomLinks(_botPrefab, _info, error);
  }
}

} // namespace superdex::studio
