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
#include "meshing/collision_recipe.h"
#include "meshing/link_meshes.h"
#include "meshing/processing_modifiers/source_parts.h"

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_renderer/path.h>
#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/file_utils.h>

#include <deque>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace superdex::studio {

// Which of a link's geoms a model is built from.
enum class GeomSource { Visual, Collision };

// How a link's render model is made: its geoms merged into one GLB, or its single mesh file copied
// as it is.
enum class RenderOperation { Convert, Copy, Ignore };

// How a link's collision model is made: its geoms merged into one mesh (Union, ShrinkWrap), split
// over Hard-jointed child links (Separate), or its single mesh file copied as it is (Copy).
enum class CollisionOperation { Union, ShrinkWrap, Separate, Copy, Ignore };

// A geom a model can be built from, as the import tables list it.
struct GeomListing {
  std::string name; // mesh file name, or primitive type and geom name
  std::string detail; // full mesh path, or what a primitive was fitted to
};

// Per-link import choices. Both model tabs share this entry.
struct LinkImportEntry {
  std::string linkName;
  int sourceLink = 0; // index into the importer's source links (see BotImporter::SourceGeoms)
  // Buildable geoms by role.
  std::vector<GeomListing> visualGeoms;
  std::vector<GeomListing> collisionGeoms;
  // The file of a role's only geom, when it is a mesh that can be copied as it is; else empty.
  std::string visualCopyFile;
  std::string collisionCopyFile;
  // References of the link's meshes that were not found on disk.
  std::vector<std::string> missingMeshes;
  // Render Models tab selection.
  GeomSource renderSource = GeomSource::Visual;
  RenderOperation renderOp = RenderOperation::Ignore;
  // Collision Models tab selection.
  GeomSource collisionSource = GeomSource::Collision;
  CollisionOperation collisionOp = CollisionOperation::Ignore;

  [[nodiscard]] std::vector<GeomListing> const& Geoms(GeomSource source) const {
    return source == GeomSource::Visual ? visualGeoms : collisionGeoms;
  }
  [[nodiscard]] int GeomCount(GeomSource source) const {
    return static_cast<int>(Geoms(source).size());
  }
  [[nodiscard]] std::string const& CopyableFile(GeomSource source) const {
    return source == GeomSource::Visual ? visualCopyFile : collisionCopyFile;
  }
};

struct BotImportOptions {
  std::string name;
  std::vector<LinkImportEntry> links;
  // Subdirectories (relative to the import destination) for the produced render and mochi model
  // files. Blank means write directly into the destination root.
  std::string renderDir;
  std::string mochiDir;
  // Remesh converted mochi (collision) models, producing a clean watertight surface.
  bool mochiRemesh = true;
  // Bake a signed distance field into converted mochi (collision) models.
  bool mochiBakeSdf = true;
  // Keep each converted collision model's processing recipe (in the `intermediates` folder) and the
  // source files it reads (in a `src` folder beside the models), so the model can be reopened,
  // tuned and rebuilt in the Model Editor.
  bool keepRecipes = true;
};

// Import wizard for robot formats whose links carry several visual and collision geoms (URDF,
// MJCF). Each link gets one render model (its geoms merged into a GLB) and one collision model
// (its geoms merged per CollisionOperation), or keeps a single mesh file as it is where the format
// allows Copy. Subclasses load the format and apply its rules.
class BotImporter : public Importer {
 public:
  [[nodiscard]] std::string const& GetDisplayName() const override;
  [[nodiscard]] std::vector<std::string> const& GetExtensions() const override;
  [[nodiscard]] ImU32 GetColor() const override;
  // name, world_joint (free | hard), render_source and collision_source (visual | collision),
  // render_operation (convert | copy | ignore),
  // collision_operation (union | shrink_wrap | separate | copy | ignore), render_dir,
  // collision_dir, remesh, bake_sdf, keep_recipes (true | false). Copy needs a format that allows
  // it.
  bool SetOption(std::string_view key, std::string_view value) override;
  [[nodiscard]] mochi::Path GetOutputAssetPath() const override;

 protected:
  struct Format {
    std::string displayName;
    std::vector<std::string> extensions;
    // A link's only mesh file may be copied as it is instead of converted.
    bool allowsCopy = false;
    // A link without collision geoms gets its collision model from its visual geoms.
    bool collisionFromVisuals = false;
  };

  BotImporter(SuperDexStudio* studio, Format format);

  // Loads `path` into `_botPrefab`, with no geometry assigned, and the per-link geoms. Appends
  // notes on skipped content to `notes`. Returns false if the file could not be loaded.
  virtual bool LoadSource(mochi::Path const& path, std::vector<std::string>& notes) = 0;
  [[nodiscard]] virtual int NumSourceLinks() const = 0;
  [[nodiscard]] virtual std::string SourceLinkName(int source) const = 0;
  virtual mochi::DynamicArray<superdex::robotics::GeomReference>& SourceGeoms(int source) = 0;
  // Adapts geoms to the format once the tables have listed them (e.g. MJCF's mesh quirks).
  virtual void PrepareGeoms(std::vector<std::string>& /*notes*/) {}
  // Adds tasks that complete `link` beside its model builds (e.g. MJCF's inferred inertia).
  virtual void AddLinkTasks(
      int /*source*/,
      superdex::robotics::BotLinkPrefab* /*link*/,
      std::vector<AsyncTask>& /*tasks*/) {}
  // Places the per-geom shapes of Separate Links links (anySeparate) and applies the format's rules
  // to `_botPrefab`, once every task has finished and before it is saved.
  virtual void FinishPrefab(bool anySeparate, mochi::Error& error) = 0;

  // A notes buffer for one background task, logged when the import finishes.
  std::vector<std::string>* AddTaskWarnings();

  void OnBeginImport(mochi::Path const& path, mochi::Path const& destDir) override;
  void OnShowImportOptions() override;
  bool OnFinishImport(std::vector<AsyncTask>& tasks, FinalizeFn& finalize) override;
  bool CanImport() override;
  [[nodiscard]] ImVec2 GetModalSize() const override;
  [[nodiscard]] std::vector<std::string> SiblingFoldersToEnsure() const override;

  superdex::robotics::BotPrefab _botPrefab;

 private:
  // Indices (into the link's geoms) of the geoms that a model of `entry` built from `source` uses.
  [[nodiscard]] std::vector<int> SelectGeoms(LinkImportEntry const& entry, GeomSource source);
  void ApplyBlanketSource(int choice, bool render);
  void ApplyBlanketRenderOp();
  void ApplyBlanketCollisionOp();
  void ShowRenderModelsTab();
  void ShowCollisionModelsTab();
  // The collision operations this format offers, in display order.
  [[nodiscard]] std::vector<CollisionOperation> CollisionOperations() const;
  // The file a collision recipe reads for @p geom's mesh: while recipes are kept, a copy in
  // @p srcDir, made once per source file (@p copies) under a name unique in @p claimed; otherwise
  // the geom's own file.
  std::string RecipeMeshFile(
      superdex::robotics::GeomReference const& geom,
      mochi::Path const& srcDir,
      std::unordered_map<std::string, std::string>& copies,
      std::set<mochi::Path>& claimed);
  // While recipes are kept, copies @p geom's mesh, if it is an STL, to `cad/<model>.stl` for the
  // model @p dst, where the Model Editor finds it as the model's CAD Model, and returns the copy.
  // Empty otherwise, or if that file exists already.
  std::string CadModelCopy(superdex::robotics::GeomReference const& geom, mochi::Path const& dst);
  // Writes the recipe building @p dst from @p parts and returns the task that runs it, which calls
  // @p onBuilt once @p dst is written. The task fails if the recipe could not be written.
  AsyncTask RecipeTask(
      std::vector<SourcePart> parts,
      CollisionRecipeOptions const& options,
      mochi::Path const& dst,
      std::function<void()> onBuilt);
  // Combo over CollisionOperations (plus Custom if `withCustom`), with Copy disabled unless
  // `canCopy`. Returns true on change.
  bool ShowCollisionOpCombo(char const* label, int& choice, bool withCustom, bool canCopy) const;

  Format _format;
  BotImportOptions _options;
  // Notes on the source shown under Import Notes: skipped content and missing meshes.
  std::vector<std::string> _notes;

  // Blanket choices (see the k* constants in the .cpp).
  int _renderSourceChoice = 0;
  int _renderOpChoice = 0;
  int _collisionSourceChoice = 0;
  int _collisionOpChoice = 0;

  // Notes collected by background tasks, one buffer per task, logged by finalize. A deque keeps
  // each task's buffer in place as more are added.
  std::deque<std::vector<std::string>> _taskWarnings;
  // Holds the recipes of an import that does not keep them, until the next import begins.
  std::optional<mochi::TempDirCleanup> _recipeScratch;

  // False if the file failed to load or produced no links; the dialog then shows a warning and
  // disables Import.
  bool _prefabValid = false;

  // True while the Name field would overwrite an existing bot in the destination.
  bool _nameConflicts = false;
};

} // namespace superdex::studio
