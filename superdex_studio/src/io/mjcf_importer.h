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

#include <mochi_core/utils/file_utils.h>
#include <superdex_robotics/utils/file_utils.h>

#include <optional>
#include <string>
#include <vector>

namespace superdex::studio {

// Imports MuJoCo MJCF files. Bodies become links. Beyond the shared wizard, it applies MuJoCo's
// mesh quirks, infers the inertia of bodies without an <inertial>, and writes MuJoCo's contact
// rules.
class MjcfImporter : public BotImporter {
 public:
  explicit MjcfImporter(SuperDexStudio* studio);

  // ".xml" is shared with other formats, so only files whose root element is <mujoco> qualify.
  [[nodiscard]] bool CanImportFile(mochi::Path const& path) const override;

 protected:
  bool LoadSource(mochi::Path const& path, std::vector<std::string>& notes) override;
  [[nodiscard]] int NumSourceLinks() const override;
  [[nodiscard]] std::string SourceLinkName(int source) const override;
  mochi::DynamicArray<superdex::robotics::GeomReference>& SourceGeoms(int source) override;
  void PrepareGeoms(std::vector<std::string>& notes) override;
  void AddLinkTasks(
      int source,
      superdex::robotics::BotLinkPrefab* link,
      std::vector<AsyncTask>& tasks) override;
  void FinishPrefab(bool anySeparate, mochi::Error& error) override;

 private:
  superdex::robotics::MjcfImportInfo _info;
  // First-object copies of multi-object OBJ files (see FirstObjObjectFile), kept until the next
  // import begins.
  std::optional<mochi::TempDirCleanup> _objCopies;
};

} // namespace superdex::studio
