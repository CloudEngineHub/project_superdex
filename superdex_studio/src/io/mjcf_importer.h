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

#include "io/importer.h"
#include "meshing/mjcf_link_meshes.h"

#include <mochi_core/utils/file_utils.h>
#include <mochi_renderer/path.h>
#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/file_utils.h>

#include <deque>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace superdex::studio {

// Which of a body's geoms a model is built from.
enum class MjcfGeomSource { Visual, Collision };

// Operation applied to a link's model in either tab. MJCF geoms are always converted (merged into
// one model per link), so unlike URDF there is no Copy.
enum class MjcfModelOperation { Convert, Ignore };

// A geom a model can be built from, as the import tables list it.
struct MjcfGeomListing {
  std::string name; // mesh file name, or primitive type and geom name
  std::string detail; // full mesh path, or what a primitive was fitted to
};

// Per-link import choices, one entry per MJCF body. Both model tabs share this entry.
struct MjcfLinkImportEntry {
  std::string linkName;
  int body = 0; // index into MjcfImportInfo::bodies
  // Buildable geoms by role, and how many of the body's mesh files were not found on disk.
  std::vector<MjcfGeomListing> visualGeoms;
  std::vector<MjcfGeomListing> collisionGeoms;
  int numMissing = 0;
  // Render Models tab selection.
  MjcfGeomSource renderSource = MjcfGeomSource::Visual;
  MjcfModelOperation renderOp = MjcfModelOperation::Ignore;
  // Collision Models tab selection: `merge` applies when `mochiOp` is Convert.
  MjcfGeomSource mochiSource = MjcfGeomSource::Collision;
  MjcfModelOperation mochiOp = MjcfModelOperation::Ignore;
  MjcfCollisionMerge merge = MjcfCollisionMerge::Union;

  [[nodiscard]] std::vector<MjcfGeomListing> const& Geoms(MjcfGeomSource source) const {
    return source == MjcfGeomSource::Visual ? visualGeoms : collisionGeoms;
  }
  [[nodiscard]] int GeomCount(MjcfGeomSource source) const {
    return static_cast<int>(Geoms(source).size());
  }
};

struct MjcfImportOptions {
  std::string name;
  std::vector<MjcfLinkImportEntry> links;
  // Subdirectories (relative to the import destination) for the produced render and mochi model
  // files. Blank means write directly into the destination root.
  std::string renderDir;
  std::string mochiDir;
  // Remesh mochi (collision) models, producing a clean watertight surface.
  bool mochiRemesh = true;
  // Bake a signed distance field into mochi (collision) models.
  bool mochiBakeSdf = true;
};

class MjcfImporter : public Importer {
 public:
  explicit MjcfImporter(SuperDexStudio* studio);

  [[nodiscard]] std::string const& GetDisplayName() const override;
  [[nodiscard]] std::vector<std::string> const& GetExtensions() const override;
  [[nodiscard]] ImU32 GetColor() const override;
  // ".xml" is shared with other formats, so only files whose root element is <mujoco> qualify.
  [[nodiscard]] bool CanImportFile(mochi::Path const& path) const override;
  // name, world_joint (free | hard), collision_operation (union | shrink_wrap | separate | ignore),
  // render_dir, collision_dir, remesh, bake_sdf (true | false).
  bool SetOption(std::string_view key, std::string_view value) override;
  [[nodiscard]] mochi::Path GetOutputAssetPath() const override;

 protected:
  void OnBeginImport(mochi::Path const& path, mochi::Path const& destDir) override;
  void OnShowImportOptions() override;
  bool OnFinishImport(std::vector<AsyncTask>& tasks, FinalizeFn& finalize) override;
  bool CanImport() override;
  [[nodiscard]] ImVec2 GetModalSize() const override;
  [[nodiscard]] std::vector<std::string> SiblingFoldersToEnsure() const override;

 private:
  // Indices (into the body's geoms) of the geoms of `entry`'s body that a model draws from.
  [[nodiscard]] std::vector<int> SelectGeoms(
      MjcfLinkImportEntry const& entry,
      MjcfGeomSource source) const;

  std::string _displayName;
  std::vector<std::string> _extensions;

  MjcfImportOptions _options;
  superdex::robotics::BotPrefab _botPrefab;
  superdex::robotics::MjcfImportInfo _info;
  // First-object copies of multi-object OBJ files (see FirstObjObjectFile), kept until the next
  // import begins.
  std::optional<mochi::TempDirCleanup> _objCopies;

  // Transient state for the blanket controls (-1 = no choice yet).
  int _renderSourceChoice = -1;
  int _renderOpChoice = -1;
  int _mochiSourceChoice = -1;
  int _collisionOpChoice = -1;

  // Notes collected by background tasks (missing meshes, union fallbacks, ...), one buffer per
  // task, logged by finalize. A deque keeps each task's buffer in place as more are added.
  std::deque<std::vector<std::string>> _taskWarnings;

  // False if the MJCF failed to load or produced no links; the dialog then shows a warning and
  // disables Import.
  bool _prefabValid = false;

  // True while the Name field would overwrite an existing bot in the destination.
  bool _nameConflicts = false;
};

} // namespace superdex::studio
