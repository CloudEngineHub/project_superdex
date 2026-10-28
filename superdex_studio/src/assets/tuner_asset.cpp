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

#include "assets/tuner_asset.h"

#include "assets/asset_manager.h"
#include "assets/mochi_prefab_asset.h"
#include "editors/tuner_editor.h"

#include <mochi_core/utils/path.h>

namespace superdex::studio {

std::unique_ptr<TunerAsset>
TunerAsset::Create(std::string const& name, mochi::Path const& path, AssetManager* manager) {
  superdex::tuner::TunerSpec spec;
  std::string error;
  if (!superdex::tuner::LoadTunerSpecFromFile(path.AsFilesystemPath(), {}, spec, error)) {
    MOCHI_LOG_ERROR("Failed to load asset tuner: %s", error.c_str());
    return nullptr;
  }
  auto asset = std::unique_ptr<TunerAsset>(new TunerAsset(name, path, AssetType::Tuner, manager));
  asset->_spec = std::move(spec);
  asset->_prefabPath =
      mochi::Path(superdex::tuner::ResolvePrefabPath(asset->_spec, path.AsFilesystemPath()));
  if (manager->LoadMochiPrefabAsset(asset->_prefabPath) == nullptr) {
    MOCHI_LOG_ERROR(
        "Asset Tuner %s: failed to load prefab %s",
        path.ToString().c_str(),
        asset->_prefabPath.ToString().c_str());
  }
  manager->RegisterReferencer(asset.get());
  return asset;
}

char const* TunerAsset::GetTypeLabel() const {
  return "Asset Tuner";
}

bool TunerAsset::RendersThumbnail() const {
  return GetPrefabAsset() != nullptr;
}

void TunerAsset::StageThumbnailScene(mochi_renderer::Scene& scene) {
  if (MochiPrefabAsset* prefab = GetPrefabAsset()) {
    prefab->StageThumbnailScene(scene);
  }
}

bool TunerAsset::IsSavable() const {
  return !IsReadOnly();
}

bool TunerAsset::Save() {
  std::string error;
  if (!superdex::tuner::SaveTunerSpecToFile(_spec, _path.AsFilesystemPath(), error)) {
    MOCHI_LOG_ERROR("Failed to save asset tuner: %s", error.c_str());
    return false;
  }
  return true;
}

std::unique_ptr<AssetEditor> TunerAsset::CreateEditor(SuperDexStudio* studio) {
  return std::make_unique<TunerEditor>(studio, this);
}

MochiPrefabAsset* TunerAsset::GetPrefabAsset() const {
  return _manager->FindAssetByPath<MochiPrefabAsset>(_prefabPath);
}

std::string const& TunerAsset::GetReferencerName() const {
  return GetName();
}

void TunerAsset::ForEachReferencedPath(
    std::function<void(mochi::Path const&)> const& callback) const {
  callback(_prefabPath);
}

bool TunerAsset::RewriteReferencedPath(mochi::Path const& oldPath, mochi::Path const& newPath) {
  if (!(_prefabPath == oldPath)) {
    return false;
  }
  _prefabPath = newPath;
  // Keep the tuner-relative "./" form when the prefab stays next to (or below) the tuner.
  std::filesystem::path const tunerDir = _path.GetParentPath().AsFilesystemPath();
  std::filesystem::path const relative = newPath.AsFilesystemPath().lexically_relative(tunerDir);
  bool const isBelowTuner = !relative.empty() && *relative.begin() != "..";
  _spec.prefab = isBelowTuner ? "./" + relative.generic_string() : newPath.ToString();
  SetDirty(true);
  return true;
}

} // namespace superdex::studio
