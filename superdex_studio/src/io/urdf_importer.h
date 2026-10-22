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

#include "io/bot_importer.h"

#include <superdex_robotics/utils/file_utils.h>

#include <string>
#include <vector>

namespace superdex::studio {

// Imports URDF files. A link's only mesh may be copied as it is, which the default operations do
// for meshes that are already .glb render models or .mochi.h5 collision models; a link without
// collision geometry gets its collision model from its visuals.
class UrdfImporter : public BotImporter {
 public:
  explicit UrdfImporter(SuperDexStudio* studio);

 protected:
  bool LoadSource(mochi::Path const& path, std::vector<std::string>& notes) override;
  [[nodiscard]] int NumSourceLinks() const override;
  [[nodiscard]] std::string SourceLinkName(int source) const override;
  mochi::DynamicArray<superdex::robotics::GeomReference>& SourceGeoms(int source) override;
  void FinishPrefab(bool anySeparate, mochi::Error& error) override;

 private:
  superdex::robotics::UrdfImportInfo _info;
};

} // namespace superdex::studio
