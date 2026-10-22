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

#include "io/bot_importer.h"
#include "app/app.h"
#include "meshing/mesh_conversion.h"
#include "ui/imgui_widgets.h"

#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_renderer/render_space.h>

#include <imgui.h>
#include <imguios/fonts/icons_font_awesome5.h>
#include <misc/cpp/imgui_stdlib.h>

#include <array>
#include <cfloat>
#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>

namespace superdex::studio {

using superdex::robotics::GeomReference;
using superdex::robotics::GeomType;

// Blanket choices for the Source rows.
static constexpr int kSourceVisual = 0;
static constexpr int kSourceCollision = 1;
static constexpr int kSourceCustom = 2;

// Blanket choices for the render Operation row.
static constexpr int kRenderConvertAll = 0;
static constexpr int kRenderCopyAll = 1;
static constexpr int kRenderIgnoreAll = 2;
static constexpr int kRenderCustom = 3;

// Blanket collision choices are CollisionOperation values, plus Custom.
static constexpr int kCollisionCustom = static_cast<int>(CollisionOperation::Ignore) + 1;

// X offset of blanket radio column `i`: past the row labels, then columns as wide as the widest
// radio, so the rows line up at any font size.
static float RadioColumn(int i) {
  ImGuiStyle const& style = ImGui::GetStyle();
  float const labelWidth = ImGui::CalcTextSize("Operation:").x + 2.0f * style.ItemSpacing.x;
  float const radioWidth = ImGui::GetFrameHeight() + style.ItemInnerSpacing.x +
      ImGui::CalcTextSize("Convert All").x + 2.0f * style.ItemSpacing.x;
  return labelWidth + radioWidth * static_cast<float>(i);
}

static ImVec4 const kWarningColor(0.95f, 0.85f, 0.45f, 1.0f);
static mochi::Real3 const kUnitScale{mochi::real(1), mochi::real(1), mochi::real(1)};

// A geom a model can be built from: a primitive, or a mesh whose file was found.
static bool IsBuildable(GeomReference const& geom) {
  return superdex::robotics::IsPrimitiveGeom(geom.type) ||
      (geom.type == GeomType::Mesh && !geom.meshFile.empty());
}

static bool HasRole(GeomReference const& geom, GeomSource source) {
  return source == GeomSource::Visual ? geom.isVisual : geom.isCollision;
}

static bool EndsWith(std::string const& file, std::string_view extension) {
  return mochi::Path{file}.AsLowercaseString().ends_with(extension);
}

static GeomListing ListGeom(GeomReference const& geom) {
  if (geom.type == GeomType::Mesh) {
    return {
        std::filesystem::path(std::string(geom.meshRef)).filename().string(),
        std::string(geom.meshFile)};
  }
  static constexpr std::array<char const*, 9> kTypeNames = {
      "plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh", "sdf"};
  std::string name = kTypeNames[static_cast<int>(geom.type)];
  if (!geom.name.empty()) {
    name += " \"" + std::string(geom.name) + "\"";
  }
  std::string detail = name;
  if (!geom.meshFile.empty()) {
    detail += ", fitted to " + std::string(geom.meshFile);
  }
  return {name, detail};
}

static char const* CollisionOperationLabel(CollisionOperation op) {
  switch (op) {
    case CollisionOperation::Union:
      return "Union";
    case CollisionOperation::ShrinkWrap:
      return "Shrink-Wrap";
    case CollisionOperation::Separate:
      return "Separate Links";
    case CollisionOperation::Copy:
      return "Copy";
    case CollisionOperation::Ignore:
      break;
  }
  return "Ignore";
}

// The render operation a blanket choice gives a link. Convert All copies a model that is already a
// GLB; Copy All converts what cannot be copied.
static RenderOperation BlanketRenderOp(LinkImportEntry const& entry, int choice) {
  std::string const& copyFile = entry.CopyableFile(entry.renderSource);
  if (entry.GeomCount(entry.renderSource) == 0 || choice == kRenderIgnoreAll) {
    return RenderOperation::Ignore;
  }
  bool const copy =
      !copyFile.empty() && (choice == kRenderCopyAll || EndsWith(copyFile, kGlbExtension));
  return copy ? RenderOperation::Copy : RenderOperation::Convert;
}

// The collision operation a blanket choice gives a link. Merging operations copy a model that is
// already a .mochi.h5; Copy unions what cannot be copied.
static CollisionOperation BlanketCollisionOp(LinkImportEntry const& entry, CollisionOperation op) {
  std::string const& copyFile = entry.CopyableFile(entry.collisionSource);
  if (entry.GeomCount(entry.collisionSource) == 0 || op == CollisionOperation::Ignore) {
    return CollisionOperation::Ignore;
  }
  if (copyFile.empty()) {
    return op == CollisionOperation::Copy ? CollisionOperation::Union : op;
  }
  return op == CollisionOperation::Copy || EndsWith(copyFile, kMochiH5Extension)
      ? CollisionOperation::Copy
      : op;
}

// A unique output path in `dir` for `stem` + `extension`, also unique against `claimed`.
static mochi::Path ClaimOutputPath(
    mochi::Path const& dir,
    std::string const& stem,
    std::string const& extension,
    std::set<mochi::Path>& claimed) {
  std::string const unique = MakeUniqueFileName(
      stem, dir, extension, [&claimed](mochi::Path const& p) { return claimed.count(p) > 0; });
  mochi::Path result = dir / (unique + extension);
  claimed.insert(result);
  return result;
}

// Output path for a copy of `file`, keeping its name (and a .mochi.h5 double extension).
static mochi::Path
ClaimCopyPath(mochi::Path const& dir, std::string const& file, std::set<mochi::Path>& claimed) {
  mochi::Path const source{file};
  std::string const filename = source.GetFilename();
  std::string_view const h5 = kMochiH5Extension;
  if (EndsWith(file, h5)) {
    return ClaimOutputPath(
        dir, filename.substr(0, filename.size() - h5.size()), std::string(h5), claimed);
  }
  return ClaimOutputPath(dir, source.GetStem(), source.GetExtension(), claimed);
}

static ImVec2 RemainingTableSize() {
  return {0.0f, ImGui::GetContentRegionAvail().y};
}

static void ShowMissingWarning(LinkImportEntry const& entry) {
  if (entry.missingMeshes.empty()) {
    return;
  }
  ImGui::SameLine();
  ImGui::TextColored(kWarningColor, "%s", ICON_FA_EXCLAMATION_TRIANGLE);
  if (ImGui::IsItemHovered()) {
    std::string text = "Mesh files not found on disk:";
    for (std::string const& ref : entry.missingMeshes) {
      text += "\n" + ref;
    }
    ImGui::SetTooltip("%s", text.c_str());
  }
}

static void ShowSourceRadios(int& sourceChoice) {
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Source:");
  ImGui::SameLine(RadioColumn(0));
  ImGui::RadioButton("Visual", &sourceChoice, kSourceVisual);
  ImGui::SameLine(RadioColumn(1));
  ImGui::RadioButton("Collision", &sourceChoice, kSourceCollision);
  ImGui::SameLine(RadioColumn(2));
  ImGui::RadioButton("Custom##source", &sourceChoice, kSourceCustom);
}

static void ShowDestinationField(std::string& destSubdir) {
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Destination:");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(18.0f * ImGui::GetFontSize());
  ImGui::InputTextWithHint("##Destination", "(root)", &destSubdir);
}

static void ShowCollisionOptions(BotImportOptions& options) {
  ImGui::Checkbox("Remesh", &options.mochiRemesh);
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Remesh each converted collision model into a clean, watertight surface.");
  }
  ImGui::SameLine();
  ImGui::Checkbox("Bake SDF", &options.mochiBakeSdf);
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Bake a signed distance field into each converted collision model.");
  }
}

// Lists the geoms a model is built from, one per line. The first line is a combo choosing between
// the link's visual and collision geoms, enabled when `editable`. Hovering a line shows its file.
static void ShowSourceCell(LinkImportEntry const& entry, GeomSource& source, bool editable) {
  std::vector<GeomListing> const& geoms = entry.Geoms(source);
  ImGui::BeginDisabled(!editable);
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (ImGui::BeginCombo("##Source", geoms.empty() ? "<none>" : geoms.front().name.c_str())) {
    for (GeomSource const s : {GeomSource::Visual, GeomSource::Collision}) {
      std::string const label = std::string(s == GeomSource::Visual ? "Visual" : "Collision") +
          " (" + std::to_string(entry.GeomCount(s)) + ")";
      if (ImGui::Selectable(label.c_str(), source == s)) {
        source = s;
      }
      if (ImGui::IsItemHovered() && entry.GeomCount(s) > 0) {
        std::string names;
        for (GeomListing const& geom : entry.Geoms(s)) {
          names += (names.empty() ? "" : "\n") + geom.name;
        }
        ImGui::SetTooltip("%s", names.c_str());
      }
    }
    ImGui::EndCombo();
  }
  for (size_t i = 0; i < geoms.size(); ++i) {
    if (i > 0) {
      // Indented like the combo's text above.
      ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetStyle().FramePadding.x);
      ImGui::AlignTextToFramePadding();
      ImGui::TextUnformatted(geoms[i].name.c_str());
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
      ImGui::SetTooltip("%s", geoms[i].detail.c_str());
    }
  }
  ImGui::EndDisabled();
}

BotImporter::BotImporter(SuperDexStudio* studio, Format format)
    : Importer(studio), _format(std::move(format)) {}

std::string const& BotImporter::GetDisplayName() const {
  return _format.displayName;
}

std::vector<std::string> const& BotImporter::GetExtensions() const {
  return _format.extensions;
}

ImU32 BotImporter::GetColor() const {
  return GetAssetTypeColor(AssetType::Bot);
}

bool BotImporter::SetOption(std::string_view key, std::string_view value) {
  if (key == "name") {
    _options.name = value;
    return !value.empty();
  }
  if (key == "render_dir") {
    _options.renderDir = value;
    return true;
  }
  if (key == "collision_dir") {
    _options.mochiDir = value;
    return true;
  }
  if (key == "world_joint") {
    if (_botPrefab.joints.empty() || (value != "free" && value != "hard")) {
      return false;
    }
    _botPrefab.joints[0].type =
        value == "free" ? mochi::ArticulatedJointType::Free : mochi::ArticulatedJointType::Hard;
    return true;
  }
  if (key == "render_operation") {
    static std::unordered_map<std::string_view, int> const kOperations = {
        {"convert", kRenderConvertAll}, {"copy", kRenderCopyAll}, {"ignore", kRenderIgnoreAll}};
    auto const it = kOperations.find(value);
    if (it == kOperations.end() || (it->second == kRenderCopyAll && !_format.allowsCopy)) {
      return false;
    }
    _renderOpChoice = it->second;
    ApplyBlanketRenderOp();
    return true;
  }
  if (key == "collision_operation") {
    static std::unordered_map<std::string_view, CollisionOperation> const kOperations = {
        {"union", CollisionOperation::Union},
        {"shrink_wrap", CollisionOperation::ShrinkWrap},
        {"separate", CollisionOperation::Separate},
        {"copy", CollisionOperation::Copy},
        {"ignore", CollisionOperation::Ignore}};
    auto const it = kOperations.find(value);
    if (it == kOperations.end() ||
        (it->second == CollisionOperation::Copy && !_format.allowsCopy)) {
      return false;
    }
    _collisionOpChoice = static_cast<int>(it->second);
    ApplyBlanketCollisionOp();
    return true;
  }
  std::optional<bool> const flag = Importer::ParseBoolOption(value);
  if (!flag) {
    return false;
  }
  if (key == "remesh") {
    _options.mochiRemesh = *flag;
  } else if (key == "bake_sdf") {
    _options.mochiBakeSdf = *flag;
  } else {
    return false;
  }
  return true;
}

mochi::Path BotImporter::GetOutputAssetPath() const {
  return GetDestDir() / (_options.name + std::string(superdex::robotics::kBotExtension));
}

std::vector<std::string>* BotImporter::AddTaskWarnings() {
  return &_taskWarnings.emplace_back();
}

std::vector<int> BotImporter::SelectGeoms(LinkImportEntry const& entry, GeomSource source) {
  std::vector<int> indices;
  auto const& geoms = SourceGeoms(entry.sourceLink);
  for (int g = 0; g < static_cast<int>(geoms.size()); ++g) {
    if (HasRole(geoms[g], source) && IsBuildable(geoms[g])) {
      indices.push_back(g);
    }
  }
  return indices;
}

std::vector<CollisionOperation> BotImporter::CollisionOperations() const {
  std::vector<CollisionOperation> ops = {
      CollisionOperation::Union, CollisionOperation::ShrinkWrap, CollisionOperation::Separate};
  if (_format.allowsCopy) {
    ops.push_back(CollisionOperation::Copy);
  }
  ops.push_back(CollisionOperation::Ignore);
  return ops;
}

void BotImporter::ApplyBlanketSource(int choice, bool render) {
  if (choice == kSourceCustom) {
    return;
  }
  GeomSource const source = choice == kSourceVisual ? GeomSource::Visual : GeomSource::Collision;
  // A link without geoms of that role keeps its source (a link with no visual geoms renders from
  // its collision geoms).
  for (auto& entry : _options.links) {
    if (entry.GeomCount(source) > 0) {
      (render ? entry.renderSource : entry.collisionSource) = source;
    }
  }
}

void BotImporter::ApplyBlanketRenderOp() {
  if (_renderOpChoice == kRenderCustom) {
    return;
  }
  for (auto& entry : _options.links) {
    entry.renderOp = BlanketRenderOp(entry, _renderOpChoice);
  }
}

void BotImporter::ApplyBlanketCollisionOp() {
  if (_collisionOpChoice == kCollisionCustom) {
    return;
  }
  for (auto& entry : _options.links) {
    entry.collisionOp =
        BlanketCollisionOp(entry, static_cast<CollisionOperation>(_collisionOpChoice));
  }
}

bool BotImporter::ShowCollisionOpCombo(
    char const* label,
    int& choice,
    bool withCustom,
    bool canCopy) const {
  bool changed = false;
  char const* const preview = choice == kCollisionCustom
      ? "Custom"
      : CollisionOperationLabel(static_cast<CollisionOperation>(choice));
  if (ImGui::BeginCombo(label, preview)) {
    for (CollisionOperation const op : CollisionOperations()) {
      int const value = static_cast<int>(op);
      ImGuiSelectableFlags const flags = op == CollisionOperation::Copy && !canCopy
          ? ImGuiSelectableFlags_Disabled
          : ImGuiSelectableFlags_None;
      if (ImGui::Selectable(CollisionOperationLabel(op), choice == value, flags)) {
        changed = choice != value;
        choice = value;
      }
    }
    if (withCustom && ImGui::Selectable("Custom", choice == kCollisionCustom)) {
      changed = choice != kCollisionCustom;
      choice = kCollisionCustom;
    }
    ImGui::EndCombo();
  }
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
    std::string tooltip =
        "How each link's collision geoms become a Mochi shape.\n"
        "Union: one watertight mesh, the boolean union of the geoms (exact; can be slow when\n"
        "  pieces touch exactly; choose Shrink-Wrap for a link whose build stalls).\n"
        "Shrink-Wrap: one mesh wrapped tightly around the geoms (fast; fills deep concavities).\n"
        "Separate Links: one Hard-jointed child link per geom.\n";
    if (_format.allowsCopy) {
      tooltip += "Copy: keep a link's only mesh file as it is.\n";
    }
    tooltip += "Ignore: no collision model.";
    ImGui::SetTooltip("%s", tooltip.c_str());
  }
  return changed;
}

void BotImporter::OnBeginImport(mochi::Path const& path, mochi::Path const& destDir) {
  _notes.clear();
  _botPrefab = {};
  _prefabValid = LoadSource(path, _notes) && !_botPrefab.links.empty();
  if (!_prefabValid) {
    MOCHI_LOG_ERROR("Failed to load %s: %s", _format.displayName.c_str(), path.ToString().c_str());
  }
  _options = {};
  _options.name = MakeUniqueFileName(
      std::string(_botPrefab.name), destDir, std::string(superdex::robotics::kBotExtension));
  _options.renderDir = std::string(superdex::robotics::kRenderSubdir);
  _options.mochiDir = std::string(superdex::robotics::kCollisionSubdir);

  _renderSourceChoice = kSourceVisual;
  _renderOpChoice = kRenderConvertAll;
  _collisionSourceChoice = kSourceCollision;
  _collisionOpChoice = static_cast<int>(CollisionOperation::Union);

  std::set<std::string> missing;
  for (int s = 0; _prefabValid && s < NumSourceLinks(); ++s) {
    LinkImportEntry entry;
    entry.linkName = SourceLinkName(s);
    entry.sourceLink = s;
    for (GeomReference const& geom : SourceGeoms(s)) {
      if (geom.type == GeomType::Mesh && geom.meshFile.empty()) {
        entry.missingMeshes.emplace_back(geom.meshRef);
        missing.emplace(geom.meshRef);
      }
      if (!IsBuildable(geom)) {
        continue;
      }
      GeomListing const listing = ListGeom(geom);
      if (geom.isVisual) {
        entry.visualGeoms.push_back(listing);
      }
      if (geom.isCollision) {
        entry.collisionGeoms.push_back(listing);
      }
    }
    if (_format.allowsCopy) {
      for (GeomSource const source : {GeomSource::Visual, GeomSource::Collision}) {
        std::vector<int> const selected = SelectGeoms(entry, source);
        if (selected.size() != 1) {
          continue;
        }
        GeomReference const& only = SourceGeoms(s)[selected.front()];
        if (only.type == GeomType::Mesh) {
          (source == GeomSource::Visual ? entry.visualCopyFile : entry.collisionCopyFile) =
              std::string(only.meshFile);
        }
      }
    }
    entry.renderSource = entry.visualGeoms.empty() ? GeomSource::Collision : GeomSource::Visual;
    entry.collisionSource = entry.collisionGeoms.empty() && _format.collisionFromVisuals
        ? GeomSource::Visual
        : GeomSource::Collision;
    _options.links.push_back(std::move(entry));
  }
  for (std::string const& ref : missing) {
    _notes.push_back("Mesh not found on disk: " + ref);
  }
  ApplyBlanketRenderOp();
  ApplyBlanketCollisionOp();
  // The tables above list the original files; geoms may now change to suit the format.
  if (_prefabValid) {
    PrepareGeoms(_notes);
  }
}

void BotImporter::ShowRenderModelsTab() {
  ShowSourceRadios(_renderSourceChoice);
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Operation:");
  int column = 0;
  ImGui::SameLine(RadioColumn(column++));
  ImGui::RadioButton("Convert All", &_renderOpChoice, kRenderConvertAll);
  if (_format.allowsCopy) {
    ImGui::SameLine(RadioColumn(column++));
    ImGui::RadioButton("Copy All", &_renderOpChoice, kRenderCopyAll);
  }
  ImGui::SameLine(RadioColumn(column++));
  ImGui::RadioButton("Ignore All", &_renderOpChoice, kRenderIgnoreAll);
  ImGui::SameLine(RadioColumn(column));
  ImGui::RadioButton("Custom##operation", &_renderOpChoice, kRenderCustom);
  ShowDestinationField(_options.renderDir);
  ApplyBlanketSource(_renderSourceChoice, /*render=*/true);
  ApplyBlanketRenderOp();

  auto const tableFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
      ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_ScrollY;
  if (!ImGui::BeginTable("##RenderModels", 3, tableFlags, RemainingTableSize())) {
    return;
  }
  ImGui::TableSetupColumn("Link", ImGuiTableColumnFlags_WidthStretch, 2.0f);
  ImGui::TableSetupColumn("Source", ImGuiTableColumnFlags_WidthStretch, 2.0f);
  ImGui::TableSetupColumn("Operation", ImGuiTableColumnFlags_WidthStretch, 2.0f);
  ImGui::TableSetupScrollFreeze(0, 1);
  ImGui::TableHeadersRow();
  ImGui::PushFramelessWidgetStyle();
  for (size_t iRow = 0; iRow < _options.links.size(); ++iRow) {
    auto& entry = _options.links[iRow];
    ImGui::PushID(static_cast<int>(iRow));
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(entry.linkName.c_str());
    ShowMissingWarning(entry);

    ImGui::TableNextColumn();
    ShowSourceCell(entry, entry.renderSource, _renderSourceChoice == kSourceCustom);

    ImGui::TableNextColumn();
    ImGui::SetNextItemWidth(-FLT_MIN);
    bool const hasSource = entry.GeomCount(entry.renderSource) > 0;
    bool const canCopy = !entry.CopyableFile(entry.renderSource).empty();
    if (!hasSource || (entry.renderOp == RenderOperation::Copy && !canCopy)) {
      entry.renderOp = hasSource ? RenderOperation::Convert : RenderOperation::Ignore;
    }
    ImGui::BeginDisabled(_renderOpChoice != kRenderCustom || !hasSource);
    std::array<char const*, 3> const labels = {"Convert to GLB", "Copy", "Ignore"};
    if (ImGui::BeginCombo("##Op", labels[static_cast<int>(entry.renderOp)])) {
      for (RenderOperation const op :
           {RenderOperation::Convert, RenderOperation::Copy, RenderOperation::Ignore}) {
        if (op == RenderOperation::Copy && !_format.allowsCopy) {
          continue;
        }
        ImGuiSelectableFlags const flags = op == RenderOperation::Copy && !canCopy
            ? ImGuiSelectableFlags_Disabled
            : ImGuiSelectableFlags_None;
        if (ImGui::Selectable(labels[static_cast<int>(op)], entry.renderOp == op, flags)) {
          entry.renderOp = op;
        }
      }
      ImGui::EndCombo();
    }
    ImGui::EndDisabled();
    ImGui::PopID();
  }
  ImGui::PopFramelessWidgetStyle();
  ImGui::EndTable();
}

void BotImporter::ShowCollisionModelsTab() {
  ShowSourceRadios(_collisionSourceChoice);
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Operation:");
  ImGui::SameLine(RadioColumn(0));
  ImGui::SetNextItemWidth(12.0f * ImGui::GetFontSize());
  ShowCollisionOpCombo("##Operation", _collisionOpChoice, /*withCustom=*/true, /*canCopy=*/true);
  ShowDestinationField(_options.mochiDir);
  ShowCollisionOptions(_options);
  ApplyBlanketSource(_collisionSourceChoice, /*render=*/false);
  ApplyBlanketCollisionOp();

  auto const tableFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
      ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_ScrollY;
  if (!ImGui::BeginTable("##CollisionModels", 3, tableFlags, RemainingTableSize())) {
    return;
  }
  ImGui::TableSetupColumn("Link", ImGuiTableColumnFlags_WidthStretch, 2.0f);
  ImGui::TableSetupColumn("Source", ImGuiTableColumnFlags_WidthStretch, 2.0f);
  ImGui::TableSetupColumn("Operation", ImGuiTableColumnFlags_WidthStretch, 2.0f);
  ImGui::TableSetupScrollFreeze(0, 1);
  ImGui::TableHeadersRow();
  ImGui::PushFramelessWidgetStyle();
  for (size_t iRow = 0; iRow < _options.links.size(); ++iRow) {
    auto& entry = _options.links[iRow];
    ImGui::PushID(static_cast<int>(iRow));
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(entry.linkName.c_str());
    ShowMissingWarning(entry);

    ImGui::TableNextColumn();
    ShowSourceCell(entry, entry.collisionSource, _collisionSourceChoice == kSourceCustom);

    ImGui::TableNextColumn();
    ImGui::SetNextItemWidth(-FLT_MIN);
    bool const hasSource = entry.GeomCount(entry.collisionSource) > 0;
    std::string const& copyFile = entry.CopyableFile(entry.collisionSource);
    if (!hasSource || (entry.collisionOp == CollisionOperation::Copy && copyFile.empty())) {
      entry.collisionOp = hasSource ? CollisionOperation::Union : CollisionOperation::Ignore;
    }
    ImGui::BeginDisabled(_collisionOpChoice != kCollisionCustom || !hasSource);
    int op = static_cast<int>(entry.collisionOp);
    if (ShowCollisionOpCombo("##Op", op, /*withCustom=*/false, !copyFile.empty())) {
      entry.collisionOp = static_cast<CollisionOperation>(op);
    }
    ImGui::EndDisabled();
    if (entry.collisionOp == CollisionOperation::Copy && !EndsWith(copyFile, kMochiH5Extension)) {
      ImGui::SameLine();
      ImGui::TextColored(kWarningColor, "%s", ICON_FA_EXCLAMATION_TRIANGLE);
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Copy preserves source bytes. This source is not a .mochi.h5 model, so the imported "
            "collision reference may fail to load.");
      }
    }
    ImGui::PopID();
  }
  ImGui::PopFramelessWidgetStyle();
  ImGui::EndTable();
}

void BotImporter::OnShowImportOptions() {
  if (!_prefabValid) {
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(230, 100, 100, 255));
    ImGui::TextWrapped(
        "Failed to load the %s file. It could not be parsed or contains no links. See the log.",
        _format.displayName.c_str());
    ImGui::PopStyleColor();
    return;
  }

  if (!ImGui::BeginTabBar("##BotImportTabs")) {
    return;
  }
  if (ImGui::BeginTabItem("General")) {
    if (_nameConflicts) {
      ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.5f, 0.0f, 0.0f, 1.0f));
    }
    ImGui::InputText("Name", &_options.name);
    if (_nameConflicts) {
      ImGui::PopStyleColor();
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("A bot named \"%s\" already exists here.", _options.name.c_str());
      }
    }
    // The root joint attaches the bot to the world: Free for a floating base, Hard to weld it.
    if (!_botPrefab.joints.empty()) {
      ImGui::ComboArticulatedJointType(
          "World Joint",
          _botPrefab.joints[0].type,
          ImGui::ArticulatedJointTypeFilter::HardFreeOnly);
    }

    std::array<char const*, 3> const sources = {"Visual", "Collision", "Custom"};
    int const numSources = static_cast<int>(sources.size());
    ImGui::SeparatorText("Render (.glb)");
    ImGui::PushID("Render");
    if (ImGui::Combo("Source", &_renderSourceChoice, sources.data(), numSources)) {
      ApplyBlanketSource(_renderSourceChoice, /*render=*/true);
      ApplyBlanketRenderOp();
    }
    ImGui::InputTextWithHint("Destination", "(root)", &_options.renderDir);
    ImGui::PopID();

    ImGui::SeparatorText("Collision (.mochi.h5)");
    ImGui::PushID("Mochi");
    if (ImGui::Combo("Source", &_collisionSourceChoice, sources.data(), numSources)) {
      ApplyBlanketSource(_collisionSourceChoice, /*render=*/false);
      ApplyBlanketCollisionOp();
    }
    ImGui::InputTextWithHint("Destination", "(root)", &_options.mochiDir);
    if (ShowCollisionOpCombo(
            "Operation", _collisionOpChoice, /*withCustom=*/true, /*canCopy=*/true)) {
      ApplyBlanketCollisionOp();
    }
    ShowCollisionOptions(_options);
    ImGui::PopID();

    if (!_notes.empty()) {
      ImGui::Spacing();
      std::string const header = "Import Notes (" + std::to_string(_notes.size()) + ")###Notes";
      if (ImGui::CollapsingHeader(header.c_str())) {
        for (std::string const& note : _notes) {
          ImGui::TextWrapped("%s", note.c_str());
        }
      }
    }
    ImGui::EndTabItem();
  }
  if (ImGui::BeginTabItem("Render Models")) {
    ShowRenderModelsTab();
    ImGui::EndTabItem();
  }
  if (ImGui::BeginTabItem("Collision Models")) {
    ShowCollisionModelsTab();
    ImGui::EndTabItem();
  }
  ImGui::EndTabBar();
}

// A task copying `source` to `dst`. On success it gives `*file` the copy and the geom's placement.
static AsyncTask CopyModelTask(
    mochi::DynamicString* file,
    mochi::Real3* scale,
    mochi::Quaternion* rotation,
    mochi::Real3* translation,
    GeomReference const& geom,
    mochi::Path const& dst) {
  mochi::Path const source{std::string(geom.meshFile)};
  mochi::TransformRT const linkFromMesh = geom.linkFromGeom * geom.geomFromMesh;
  return AsyncTask{
      "Copy " + source.GetFilename(),
      [=, meshScale = geom.meshScale](AsyncCancelToken const& cancel) {
        if (cancel.IsCancelRequested()) {
          return false;
        }
        std::error_code error;
        std::filesystem::copy_file(
            source.AsFilesystemPath(),
            dst.AsFilesystemPath(),
            std::filesystem::copy_options::overwrite_existing,
            error);
        if (error) {
          MOCHI_LOG_ERROR("Failed to copy import asset: %s", source.ToString().c_str());
          return false;
        }
        *file = dst.ToString();
        *scale = meshScale;
        *rotation = linkFromMesh.GetRotation();
        *translation = linkFromMesh.GetTranslation();
        return true;
      }};
}

bool BotImporter::OnFinishImport(std::vector<AsyncTask>& tasks, FinalizeFn& finalize) {
  mochi::Path const destDir = GetDestDir();
  mochi::Path const mochiDir = _options.mochiDir.empty() ? destDir : destDir / _options.mochiDir;
  mochi::Path const renderDir = _options.renderDir.empty() ? destDir : destDir / _options.renderDir;
  mochi::CoordinateSpaceConverter const renderConverter(
      mochi::CoordinateSpace::Default(), mochi_renderer::RenderSpace());
  LinkCollisionOptions collisionOptions;
  collisionOptions.remesh = _options.mochiRemesh;
  collisionOptions.bakeSdf = _options.mochiBakeSdf;

  // Output names are de-duplicated against files on disk and against each other.
  std::set<mochi::Path> claimed;
  std::unordered_map<std::string, int> linkIndex;
  for (int i = 0; i < static_cast<int>(_botPrefab.links.size()); ++i) {
    linkIndex[std::string(_botPrefab.links[i].name)] = i;
  }
  // The importer assigns every model itself; the Expand*GeomLinks functions only place the
  // per-geom shapes of Separate Links links, so clear the loader's raw mesh assignments.
  for (int s = 0; s < NumSourceLinks(); ++s) {
    for (GeomReference& geom : SourceGeoms(s)) {
      geom.shapeFile.clear();
      geom.renderModelFile.clear();
    }
  }

  _taskWarnings.clear();
  bool anySeparate = false;
  bool needRenderDir = false;
  bool needMochiDir = false;
  for (LinkImportEntry const& entry : _options.links) {
    auto& geoms = SourceGeoms(entry.sourceLink);
    auto const linkIt = linkIndex.find(entry.linkName);
    if (linkIt == linkIndex.end()) {
      MOCHI_LOG_WARNING(
          "%s import: no link named \"%s\"; skipping it.",
          _format.displayName.c_str(),
          entry.linkName.c_str());
      continue;
    }
    superdex::robotics::BotLinkPrefab* link = &_botPrefab.links[linkIt->second];
    AddLinkTasks(entry.sourceLink, link, tasks);

    std::vector<int> const renderGeoms = SelectGeoms(entry, entry.renderSource);
    // A Copy left over from an earlier Source is converted instead (Copy needs one mesh file).
    if (entry.renderOp == RenderOperation::Copy &&
        !entry.CopyableFile(entry.renderSource).empty()) {
      GeomReference const& geom = geoms[renderGeoms.front()];
      needRenderDir = true;
      tasks.push_back(CopyModelTask(
          &link->renderModelFile,
          &link->renderModelScale,
          &link->renderModelRotation,
          &link->renderModelTranslation,
          geom,
          ClaimCopyPath(renderDir, std::string(geom.meshFile), claimed)));
    } else if (entry.renderOp != RenderOperation::Ignore && !renderGeoms.empty()) {
      GeomList selected;
      for (int g : renderGeoms) {
        selected.push_back(geoms[g]);
      }
      mochi::Path const dst =
          ClaimOutputPath(renderDir, entry.linkName, std::string(kGlbExtension), claimed);
      std::vector<std::string>* warnings = AddTaskWarnings();
      needRenderDir = true;
      tasks.push_back(
          AsyncTask{
              "Build " + dst.GetFilename(),
              [link, selected = std::move(selected), dst, renderConverter, warnings](
                  AsyncCancelToken const& cancel) {
                if (cancel.IsCancelRequested() ||
                    !BuildLinkGlb(selected, dst, renderConverter, *warnings)) {
                  return false;
                }
                link->renderModelFile = dst.ToString();
                link->renderModelScale = kUnitScale;
                link->renderModelRotation = mochi::Quaternion::Identity();
                link->renderModelTranslation = {};
                return true;
              }});
    }

    std::vector<int> const collisionGeoms = SelectGeoms(entry, entry.collisionSource);
    if (entry.collisionOp == CollisionOperation::Ignore || collisionGeoms.empty()) {
      continue;
    }
    needMochiDir = true;
    if (entry.collisionOp == CollisionOperation::Copy &&
        !entry.CopyableFile(entry.collisionSource).empty()) {
      GeomReference const& geom = geoms[collisionGeoms.front()];
      tasks.push_back(CopyModelTask(
          &link->shapeFile,
          &link->shapeScale,
          &link->shapeRotation,
          &link->shapeTranslation,
          geom,
          ClaimCopyPath(mochiDir, std::string(geom.meshFile), claimed)));
      continue;
    }
    // Geoms chosen as collision geometry collide, whatever their source said.
    for (int g : collisionGeoms) {
      GeomReference& geom = geoms[g];
      if (!geom.isCollision) {
        geom.isCollision = true;
        geom.contype = 1;
        geom.conaffinity = 1;
      }
    }
    std::string const h5(kMochiH5Extension);
    if (entry.collisionOp == CollisionOperation::Separate) {
      anySeparate = true;
      int k = 0;
      for (int g : collisionGeoms) {
        GeomReference* geom = &geoms[g];
        mochi::Path const dst =
            ClaimOutputPath(mochiDir, entry.linkName + "_" + std::to_string(k++), h5, claimed);
        std::vector<std::string>* warnings = AddTaskWarnings();
        tasks.push_back(
            AsyncTask{
                "Build " + dst.GetFilename(),
                [geom, dst, collisionOptions, warnings](AsyncCancelToken const& cancel) {
                  if (cancel.IsCancelRequested() ||
                      !BuildGeomCollisionH5(*geom, collisionOptions, dst, *warnings)) {
                    return false;
                  }
                  // The file holds the scaled mesh; the layout applies only the pose.
                  geom->shapeFile = dst.ToString();
                  geom->meshScale = kUnitScale;
                  return true;
                }});
      }
      continue;
    }

    GeomList selected;
    for (int g : collisionGeoms) {
      selected.push_back(geoms[g]);
    }
    LinkCollisionOptions options = collisionOptions;
    options.merge = entry.collisionOp == CollisionOperation::ShrinkWrap ? CollisionMerge::ShrinkWrap
                                                                        : CollisionMerge::Union;
    mochi::Path const dst = ClaimOutputPath(mochiDir, entry.linkName, h5, claimed);
    std::vector<std::string>* warnings = AddTaskWarnings();
    tasks.push_back(
        AsyncTask{
            "Build " + dst.GetFilename(),
            [link, selected = std::move(selected), dst, options, warnings](
                AsyncCancelToken const& cancel) {
              if (cancel.IsCancelRequested() ||
                  !BuildLinkCollisionH5(selected, options, dst, *warnings)) {
                return false;
              }
              link->shapeFile = dst.ToString();
              link->shapeScale = kUnitScale;
              link->shapeRotation = mochi::Quaternion::Identity();
              link->shapeTranslation = {};
              return true;
            }});
  }

  auto ensureDir = [](mochi::Path const& dir) {
    std::error_code dirError;
    std::filesystem::create_directories(dir.AsFilesystemPath(), dirError);
    if (dirError) {
      MOCHI_LOG_WARNING(
          "Could not create %s: %s", dir.ToString().c_str(), dirError.message().c_str());
    }
  };
  if (needRenderDir) {
    ensureDir(renderDir);
  }
  if (needMochiDir) {
    ensureDir(mochiDir);
  }

  mochi::Path const outPath = GetOutputAssetPath();
  finalize = [this, outPath, destDir, anySeparate](mochi::Path& outAssetPath) -> bool {
    _botPrefab.name = _options.name;
    mochi::ErrorLog error;
    FinishPrefab(anySeparate, error);
    if (!error.IsOK()) {
      MOCHI_LOG_ERROR(
          "Failed to lay out the Separate Links collision geoms; the bot was not saved.");
      return false;
    }
    std::string notes;
    for (auto const& warnings : _taskWarnings) {
      for (auto const& warning : warnings) {
        notes += "\n  " + warning;
      }
    }
    if (!notes.empty()) {
      MOCHI_LOG_WARNING(
          "%s import of \"%s\":%s",
          _format.displayName.c_str(),
          _options.name.c_str(),
          notes.c_str());
    }
    superdex::robotics::SaveToFile(_botPrefab, outPath.ToString(), error);
    if (!error.IsOK()) {
      MOCHI_LOG_ERROR("Failed to save bot: %s", outPath.ToString().c_str());
      return false;
    }
    // Anchor the imported bot under a bots root so its root-relative (`//`) and tag (`@`)
    // references resolve. If nothing up the ancestry is marked, mark the destination folder.
    if (!superdex::robotics::FindBotsRoot(outPath.AsFilesystemPath()).has_value()) {
      SuperDexStudio::CreateRootMarker(destDir, mochi::ErrorLog{});
    }
    outAssetPath = outPath;
    return true;
  };
  return true;
}

bool BotImporter::CanImport() {
  std::error_code ec;
  _nameConflicts = !_options.name.empty() &&
      std::filesystem::exists(GetOutputAssetPath().AsFilesystemPath(), ec);
  return _prefabValid && !_options.name.empty() && !_nameConflicts;
}

std::vector<std::string> BotImporter::SiblingFoldersToEnsure() const {
  // Standard layout is all-or-nothing: if either destination was redirected, the user has chosen a
  // custom layout, so only the folders the import writes to are created.
  bool const standardLayout =
      std::string_view(_options.renderDir) == superdex::robotics::kRenderSubdir &&
      std::string_view(_options.mochiDir) == superdex::robotics::kCollisionSubdir;
  if (!standardLayout) {
    return {};
  }
  return {
      std::string(superdex::robotics::kCadSubdir),
      std::string(superdex::robotics::kIntermediatesSubdir),
      std::string(superdex::robotics::kRenderSubdir),
      std::string(superdex::robotics::kCollisionSubdir)};
}

ImVec2 BotImporter::GetModalSize() const {
  return {960.0f, 1080.0f};
}

} // namespace superdex::studio
