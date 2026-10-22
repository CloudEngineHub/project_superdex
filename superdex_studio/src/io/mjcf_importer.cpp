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

#include "io/mjcf_importer.h"
#include "app/app.h"
#include "meshing/mesh_conversion.h"
#include "ui/imgui_widgets.h"

#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/log.h>
#include <mochi_renderer/render_space.h>

#include <imgui.h>
#include <imguios/fonts/icons_font_awesome5.h>
#include <misc/cpp/imgui_stdlib.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cfloat>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>

namespace superdex::studio {

using superdex::robotics::MjcfBodyReference;
using superdex::robotics::MjcfGeomReference;
using superdex::robotics::MjcfGeomType;

// Blanket choices for the Source row.
static constexpr int kSourceVisual = 0;
static constexpr int kSourceCollision = 1;
static constexpr int kSourceCustom = 2;

// Blanket choices for the Operation row.
static constexpr int kOpConvertAll = 0;
static constexpr int kOpIgnoreAll = 1;
static constexpr int kOpCustom = 2;

// Collision Operation choices: the MjcfCollisionMerge values, each converting the link's geoms,
// then Ignore. The blanket controls add Custom.
static constexpr int kCollisionOpIgnore = 3;
static constexpr int kCollisionOpCustom = 4;
static constexpr std::array<char const*, 5> kCollisionOpLabels =
    {"Union", "Shrink-Wrap", "Separate Links", "Ignore", "Custom"};

// Aligned x offsets (px) for the blanket radio columns so both rows line up.
static constexpr float kRadioCol0 = 80.0f;
static constexpr float kRadioCol1 = 200.0f;
static constexpr float kRadioCol2 = 320.0f;

// How far into a .xml file to look for the <mujoco> root element.
static constexpr std::streamsize kSniffBytes = 64 * 1024;

static ImVec4 const kWarningColor(0.95f, 0.85f, 0.45f, 1.0f);
static mochi::Real3 const kUnitScale{mochi::real(1), mochi::real(1), mochi::real(1)};

// A geom a model can be built from: a primitive, or a mesh whose file was found.
static bool IsBuildable(MjcfGeomReference const& geom) {
  switch (geom.type) {
    case MjcfGeomType::Sphere:
    case MjcfGeomType::Capsule:
    case MjcfGeomType::Ellipsoid:
    case MjcfGeomType::Cylinder:
    case MjcfGeomType::Box:
      return true;
    case MjcfGeomType::Mesh:
      return !geom.meshFile.empty();
    case MjcfGeomType::Plane:
    case MjcfGeomType::Hfield:
    case MjcfGeomType::Sdf:
      break;
  }
  return false;
}

static bool HasRole(MjcfGeomReference const& geom, MjcfGeomSource source) {
  return source == MjcfGeomSource::Visual ? geom.isVisual : geom.isCollision;
}

static MjcfModelOperation DefaultOp(MjcfLinkImportEntry const& entry, MjcfGeomSource source) {
  return entry.GeomCount(source) > 0 ? MjcfModelOperation::Convert : MjcfModelOperation::Ignore;
}

static MjcfGeomListing ListGeom(MjcfGeomReference const& geom) {
  if (geom.type == MjcfGeomType::Mesh) {
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

static ImVec2 RemainingTableSize() {
  return {0.0f, ImGui::GetContentRegionAvail().y};
}

static void ShowMissingWarning(MjcfLinkImportEntry const& entry) {
  if (entry.numMissing == 0) {
    return;
  }
  ImGui::SameLine();
  ImGui::TextColored(kWarningColor, "%s", ICON_FA_EXCLAMATION_TRIANGLE);
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("%d mesh file(s) of this body were not found on disk.", entry.numMissing);
  }
}

static void ApplyBlanketSource(std::vector<MjcfLinkImportEntry>& links, int choice, bool render) {
  if (choice == kSourceCustom) {
    return;
  }
  MjcfGeomSource const source =
      choice == kSourceVisual ? MjcfGeomSource::Visual : MjcfGeomSource::Collision;
  // A link without geoms of that role keeps its source (a body with no visual geoms renders from
  // its collision geoms).
  for (auto& entry : links) {
    if (entry.GeomCount(source) > 0) {
      (render ? entry.renderSource : entry.mochiSource) = source;
    }
  }
}

static void
ApplyBlanketOperation(std::vector<MjcfLinkImportEntry>& links, int choice, bool render) {
  if (choice == kOpCustom) {
    return;
  }
  for (auto& entry : links) {
    MjcfGeomSource const source = render ? entry.renderSource : entry.mochiSource;
    (render ? entry.renderOp : entry.mochiOp) =
        choice == kOpConvertAll ? DefaultOp(entry, source) : MjcfModelOperation::Ignore;
  }
}

static void SetCollisionOp(MjcfLinkImportEntry& entry, int choice) {
  if (choice == kCollisionOpIgnore) {
    entry.mochiOp = MjcfModelOperation::Ignore;
    return;
  }
  entry.mochiOp = DefaultOp(entry, entry.mochiSource);
  entry.merge = static_cast<MjcfCollisionMerge>(choice);
}

static void ApplyBlanketCollisionOp(std::vector<MjcfLinkImportEntry>& links, int choice) {
  if (choice == kCollisionOpCustom) {
    return;
  }
  for (auto& entry : links) {
    SetCollisionOp(entry, choice);
  }
}

static int CollisionOpChoice(MjcfLinkImportEntry const& entry) {
  return entry.mochiOp == MjcfModelOperation::Ignore ? kCollisionOpIgnore
                                                     : static_cast<int>(entry.merge);
}

static void ShowSourceRadios(int& sourceChoice) {
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Source:");
  ImGui::SameLine(kRadioCol0);
  ImGui::RadioButton("Visual", &sourceChoice, kSourceVisual);
  ImGui::SameLine(kRadioCol1);
  ImGui::RadioButton("Collision", &sourceChoice, kSourceCollision);
  ImGui::SameLine(kRadioCol2);
  ImGui::RadioButton("Custom##source", &sourceChoice, kSourceCustom);
}

static void ShowOperationRadios(int& opChoice) {
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Operation:");
  ImGui::SameLine(kRadioCol0);
  ImGui::RadioButton("Convert All", &opChoice, kOpConvertAll);
  ImGui::SameLine(kRadioCol1);
  ImGui::RadioButton("Ignore All", &opChoice, kOpIgnoreAll);
  ImGui::SameLine(kRadioCol2);
  ImGui::RadioButton("Custom##operation", &opChoice, kOpCustom);
}

static void ShowDestinationField(std::string& destSubdir) {
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Destination:");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(300.0f);
  ImGui::InputTextWithHint("##Destination", "(root)", &destSubdir);
}

// Combo over kCollisionOpLabels, without Custom unless `withCustom`. Returns true on change.
static bool ShowCollisionOpCombo(char const* label, int& choice, bool withCustom) {
  bool const changed = ImGui::Combo(
      label,
      &choice,
      kCollisionOpLabels.data(),
      withCustom ? kCollisionOpCustom + 1 : kCollisionOpCustom);
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
    ImGui::SetTooltip(
        "How each body's collision geoms become a Mochi shape.\n"
        "Union: one watertight mesh, the boolean union of the geoms (exact; can be slow on\n"
        "  Windows and CMake builds; choose Shrink-Wrap for a link whose build stalls).\n"
        "Shrink-Wrap: one mesh wrapped tightly around the geoms (fast; fills deep concavities).\n"
        "Separate Links: one Hard-jointed child link per geom.\n"
        "Ignore: no collision model.");
  }
  return changed;
}

static void ShowCollisionOptions(MjcfImportOptions& options) {
  ImGui::Checkbox("Remesh", &options.mochiRemesh);
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Remesh each collision model into a clean, watertight surface.");
  }
  ImGui::SameLine();
  ImGui::Checkbox("Bake SDF", &options.mochiBakeSdf);
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Bake a signed distance field into each collision model.");
  }
}

// Lists the geoms a model is built from, one per line. The first line is a combo choosing between
// the body's visual and collision geoms, enabled when `editable`. Hovering a line shows its file.
static void
ShowSourceCell(MjcfLinkImportEntry const& entry, MjcfGeomSource& source, bool editable) {
  std::vector<MjcfGeomListing> const& geoms = entry.Geoms(source);
  ImGui::BeginDisabled(!editable);
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (ImGui::BeginCombo("##Source", geoms.empty() ? "<none>" : geoms.front().name.c_str())) {
    for (MjcfGeomSource const s : {MjcfGeomSource::Visual, MjcfGeomSource::Collision}) {
      std::string const label = std::string(s == MjcfGeomSource::Visual ? "Visual" : "Collision") +
          " (" + std::to_string(entry.GeomCount(s)) + ")";
      if (ImGui::Selectable(label.c_str(), source == s)) {
        source = s;
      }
      if (ImGui::IsItemHovered() && entry.GeomCount(s) > 0) {
        std::string names;
        for (MjcfGeomListing const& geom : entry.Geoms(s)) {
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

static void ShowOperationCombo(MjcfModelOperation& op, char const* convertLabel) {
  std::array<char const*, 2> const items = {convertLabel, "Ignore"};
  int current = static_cast<int>(op);
  if (ImGui::Combo("##Op", &current, items.data(), static_cast<int>(items.size()))) {
    op = static_cast<MjcfModelOperation>(current);
  }
}

static void ShowRenderModelsTab(
    std::vector<MjcfLinkImportEntry>& links,
    int& sourceChoice,
    int& opChoice,
    std::string& destSubdir) {
  ShowSourceRadios(sourceChoice);
  ShowOperationRadios(opChoice);
  ShowDestinationField(destSubdir);
  ApplyBlanketSource(links, sourceChoice, /*render=*/true);
  ApplyBlanketOperation(links, opChoice, /*render=*/true);

  auto const tableFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
      ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_ScrollY;
  if (ImGui::BeginTable("##RenderModels", 3, tableFlags, RemainingTableSize())) {
    ImGui::TableSetupColumn("Link", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableSetupColumn("Source", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableSetupColumn("Operation", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableHeadersRow();
    ImGui::PushFramelessWidgetStyle();
    for (size_t iRow = 0; iRow < links.size(); ++iRow) {
      auto& entry = links[iRow];
      ImGui::PushID(static_cast<int>(iRow));
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::AlignTextToFramePadding();
      ImGui::TextUnformatted(entry.linkName.c_str());
      ShowMissingWarning(entry);

      ImGui::TableNextColumn();
      ShowSourceCell(entry, entry.renderSource, sourceChoice == kSourceCustom);

      ImGui::TableNextColumn();
      ImGui::SetNextItemWidth(-FLT_MIN);
      bool const hasSource = entry.GeomCount(entry.renderSource) > 0;
      if (!hasSource) {
        entry.renderOp = MjcfModelOperation::Ignore;
      }
      ImGui::BeginDisabled(opChoice != kOpCustom || !hasSource);
      ShowOperationCombo(entry.renderOp, "Convert to GLB");
      ImGui::EndDisabled();
      ImGui::PopID();
    }
    ImGui::PopFramelessWidgetStyle();
    ImGui::EndTable();
  }
}

static void ShowCollisionModelsTab(
    std::vector<MjcfLinkImportEntry>& links,
    int& sourceChoice,
    int& opChoice,
    MjcfImportOptions& options) {
  ShowSourceRadios(sourceChoice);
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Operation:");
  ImGui::SameLine(kRadioCol0);
  ImGui::SetNextItemWidth(200.0f);
  ShowCollisionOpCombo("##Operation", opChoice, /*withCustom=*/true);
  ShowDestinationField(options.mochiDir);
  ShowCollisionOptions(options);
  ApplyBlanketSource(links, sourceChoice, /*render=*/false);
  ApplyBlanketCollisionOp(links, opChoice);

  auto const tableFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
      ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_ScrollY;
  if (ImGui::BeginTable("##CollisionModels", 3, tableFlags, RemainingTableSize())) {
    ImGui::TableSetupColumn("Link", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableSetupColumn("Source", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableSetupColumn("Operation", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableHeadersRow();
    ImGui::PushFramelessWidgetStyle();
    for (size_t iRow = 0; iRow < links.size(); ++iRow) {
      auto& entry = links[iRow];
      ImGui::PushID(static_cast<int>(iRow));
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::AlignTextToFramePadding();
      ImGui::TextUnformatted(entry.linkName.c_str());
      ShowMissingWarning(entry);

      ImGui::TableNextColumn();
      ShowSourceCell(entry, entry.mochiSource, sourceChoice == kSourceCustom);

      ImGui::TableNextColumn();
      ImGui::SetNextItemWidth(-FLT_MIN);
      bool const hasSource = entry.GeomCount(entry.mochiSource) > 0;
      if (!hasSource) {
        entry.mochiOp = MjcfModelOperation::Ignore;
      }
      ImGui::BeginDisabled(opChoice != kCollisionOpCustom || !hasSource);
      int op = CollisionOpChoice(entry);
      if (ShowCollisionOpCombo("##Op", op, /*withCustom=*/false)) {
        SetCollisionOp(entry, op);
      }
      ImGui::EndDisabled();
      ImGui::PopID();
    }
    ImGui::PopFramelessWidgetStyle();
    ImGui::EndTable();
  }
}

MjcfImporter::MjcfImporter(SuperDexStudio* studio)
    : Importer(studio),
      _displayName("MJCF"),
      _extensions({std::string(superdex::robotics::kMjcfExtension), ".mjcf"}) {}

std::string const& MjcfImporter::GetDisplayName() const {
  return _displayName;
}

std::vector<std::string> const& MjcfImporter::GetExtensions() const {
  return _extensions;
}

ImU32 MjcfImporter::GetColor() const {
  return GetAssetTypeColor(AssetType::Bot);
}

bool MjcfImporter::CanImportFile(mochi::Path const& path) const {
  std::ifstream stream(path.AsFilesystemPath(), std::ios::binary);
  std::string head(static_cast<size_t>(kSniffBytes), '\0');
  stream.read(head.data(), kSniffBytes);
  head.resize(static_cast<size_t>(stream.gcount()));
  // A <mujoco> element, not <mujocoinclude> or other text that merely starts the same way.
  for (size_t at = head.find("<mujoco"); at != std::string::npos;
       at = head.find("<mujoco", at + 1)) {
    size_t const next = at + std::string_view("<mujoco").size();
    if (next < head.size() &&
        (std::isspace(static_cast<unsigned char>(head[next])) || head[next] == '>' ||
         head[next] == '/')) {
      return true;
    }
  }
  return false;
}

bool MjcfImporter::SetOption(std::string_view key, std::string_view value) {
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
  if (key == "collision_operation") {
    static std::unordered_map<std::string_view, int> const kOperations = {
        {"union", 0}, {"shrink_wrap", 1}, {"separate", 2}, {"ignore", 3}};
    auto const it = kOperations.find(value);
    if (it == kOperations.end()) {
      return false;
    }
    _collisionOpChoice = it->second;
    ApplyBlanketCollisionOp(_options.links, _collisionOpChoice);
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

mochi::Path MjcfImporter::GetOutputAssetPath() const {
  return GetDestDir() / (_options.name + std::string(superdex::robotics::kBotExtension));
}

std::vector<int> MjcfImporter::SelectGeoms(MjcfLinkImportEntry const& entry, MjcfGeomSource source)
    const {
  std::vector<int> indices;
  auto const& geoms = _info.bodies[entry.body].geoms;
  for (int g = 0; g < static_cast<int>(geoms.size()); ++g) {
    if (HasRole(geoms[g], source) && IsBuildable(geoms[g])) {
      indices.push_back(g);
    }
  }
  return indices;
}

void MjcfImporter::OnBeginImport(mochi::Path const& path, mochi::Path const& destDir) {
  mochi::ErrorLog error;
  _info = {};
  _botPrefab = superdex::robotics::LoadBotPrefabFromMjcfFile(path.ToString(), _info, error);
  _prefabValid = error.IsOK() && !_botPrefab.links.empty();
  if (!_prefabValid) {
    MOCHI_LOG_ERROR("Failed to load MJCF: %s", path.ToString().c_str());
  }
  _options = {};
  _options.name = MakeUniqueFileName(
      std::string(_botPrefab.name), destDir, std::string(superdex::robotics::kBotExtension));
  _options.renderDir = std::string(superdex::robotics::kRenderSubdir);
  _options.mochiDir = std::string(superdex::robotics::kCollisionSubdir);

  _renderSourceChoice = kSourceVisual;
  _renderOpChoice = kOpConvertAll;
  _mochiSourceChoice = kSourceCollision;
  _collisionOpChoice = static_cast<int>(MjcfCollisionMerge::Union);

  for (int b = 0; b < static_cast<int>(_info.bodies.size()); ++b) {
    MjcfBodyReference const& body = _info.bodies[b];
    MjcfLinkImportEntry entry;
    entry.linkName = std::string(body.link);
    entry.body = b;
    for (MjcfGeomReference const& geom : body.geoms) {
      entry.numMissing += geom.type == MjcfGeomType::Mesh && geom.meshFile.empty() ? 1 : 0;
      if (!IsBuildable(geom)) {
        continue;
      }
      MjcfGeomListing const listing = ListGeom(geom);
      if (geom.isVisual) {
        entry.visualGeoms.push_back(listing);
      }
      if (geom.isCollision) {
        entry.collisionGeoms.push_back(listing);
      }
    }
    // A body without visual geoms is rendered from its collision geoms. A body without collision
    // geoms does not collide in MuJoCo, so it gets no collision model.
    entry.renderSource =
        entry.visualGeoms.empty() ? MjcfGeomSource::Collision : MjcfGeomSource::Visual;
    entry.renderOp = DefaultOp(entry, entry.renderSource);
    entry.mochiOp = DefaultOp(entry, entry.mochiSource);
    _options.links.push_back(std::move(entry));
  }

  // The listings above name the original files; geoms now switch to first-object OBJ copies.
  _objCopies.reset();
  mochi::ErrorLog tempDirError;
  _objCopies.emplace(mochi::CreateTempDirectory("mjcf_import", tempDirError));
  std::unordered_map<std::string, std::string> objFiles;
  std::vector<std::string> fitWarnings;
  for (MjcfBodyReference& body : _info.bodies) {
    for (MjcfGeomReference& geom : body.geoms) {
      std::string extension =
          std::filesystem::path(std::string(geom.meshFile)).extension().string();
      std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
      if (extension == ".obj") {
        auto const [file, inserted] = objFiles.try_emplace(std::string(geom.meshFile));
        if (inserted) {
          file->second = FirstObjObjectFile(file->first, _objCopies->Path());
        }
        geom.meshFile = file->second;
      }
      if (geom.type != MjcfGeomType::Mesh && !geom.meshFile.empty()) {
        FitPrimitiveToMesh(geom, fitWarnings);
      }
    }
  }
  for (std::string const& warning : fitWarnings) {
    _info.warnings.push_back(mochi::DynamicString{warning});
  }
}

void MjcfImporter::OnShowImportOptions() {
  if (!_prefabValid) {
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(230, 100, 100, 255));
    ImGui::TextWrapped(
        "Failed to load MJCF. The file could not be parsed or contains no bodies. See the log.");
    ImGui::PopStyleColor();
    return;
  }

  if (ImGui::BeginTabBar("##MjcfImportTabs")) {
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
      // It defaults to Free when the MJCF root has a <freejoint>.
      if (!_botPrefab.joints.empty()) {
        ImGui::ComboArticulatedJointType(
            "World Joint",
            _botPrefab.joints[0].type,
            ImGui::ArticulatedJointTypeFilter::HardFreeOnly);
      }

      ImGui::SeparatorText("Render (.glb)");
      ImGui::PushID("Render");
      std::array<char const*, 3> const sources = {"Visual", "Collision", "Custom"};
      int const numSources = static_cast<int>(sources.size());
      if (ImGui::Combo("Source", &_renderSourceChoice, sources.data(), numSources)) {
        ApplyBlanketSource(_options.links, _renderSourceChoice, /*render=*/true);
        ApplyBlanketOperation(_options.links, _renderOpChoice, /*render=*/true);
      }
      ImGui::InputTextWithHint("Destination", "(root)", &_options.renderDir);
      ImGui::PopID();

      ImGui::SeparatorText("Collision (.mochi.h5)");
      ImGui::PushID("Mochi");
      if (ImGui::Combo("Source", &_mochiSourceChoice, sources.data(), numSources)) {
        ApplyBlanketSource(_options.links, _mochiSourceChoice, /*render=*/false);
        ApplyBlanketCollisionOp(_options.links, _collisionOpChoice);
      }
      ImGui::InputTextWithHint("Destination", "(root)", &_options.mochiDir);
      if (ShowCollisionOpCombo("Operation", _collisionOpChoice, /*withCustom=*/true)) {
        ApplyBlanketCollisionOp(_options.links, _collisionOpChoice);
      }
      ShowCollisionOptions(_options);
      ImGui::PopID();

      if (!_info.warnings.empty()) {
        ImGui::Spacing();
        std::string const header =
            "Import Notes (" + std::to_string(_info.warnings.size()) + ")###Notes";
        if (ImGui::CollapsingHeader(header.c_str())) {
          for (auto const& warning : _info.warnings) {
            ImGui::TextWrapped("%s", warning.c_str());
          }
        }
      }
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Render Models")) {
      ShowRenderModelsTab(_options.links, _renderSourceChoice, _renderOpChoice, _options.renderDir);
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Collision Models")) {
      ShowCollisionModelsTab(_options.links, _mochiSourceChoice, _collisionOpChoice, _options);
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
}

bool MjcfImporter::OnFinishImport(std::vector<AsyncTask>& tasks, FinalizeFn& finalize) {
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
  auto outputPath = [&claimed](
                        mochi::Path const& dir, std::string const& stem, std::string_view ext) {
    std::string const extension(ext);
    std::string const unique = MakeUniqueFileName(
        stem, dir, extension, [&claimed](mochi::Path const& p) { return claimed.count(p) > 0; });
    mochi::Path result = dir / (unique + extension);
    claimed.insert(result);
    return result;
  };

  std::unordered_map<std::string, int> linkIndex;
  for (int i = 0; i < static_cast<int>(_botPrefab.links.size()); ++i) {
    linkIndex[std::string(_botPrefab.links[i].name)] = i;
  }
  // Studio assigns every model itself; ExpandMjcfGeomLinks only places the per-geom shapes of
  // Separate links, so clear the loader's raw mesh assignments.
  for (MjcfBodyReference& body : _info.bodies) {
    for (MjcfGeomReference& geom : body.geoms) {
      geom.shapeFile.clear();
      geom.renderModelFile.clear();
    }
  }

  _taskWarnings.clear();
  bool anySeparate = false;
  bool needRenderDir = false;
  bool needMochiDir = false;
  for (MjcfLinkImportEntry const& entry : _options.links) {
    MjcfBodyReference& body = _info.bodies[entry.body];
    auto const linkIt = linkIndex.find(entry.linkName);
    if (linkIt == linkIndex.end()) {
      MOCHI_LOG_WARNING("MJCF import: no link named \"%s\"; skipping it.", entry.linkName.c_str());
      continue;
    }
    superdex::robotics::BotLinkPrefab* link = &_botPrefab.links[linkIt->second];

    if (!body.hasInertial && !body.geoms.empty()) {
      std::vector<std::string>* warnings = &_taskWarnings.emplace_back();
      tasks.push_back(
          AsyncTask{
              "Compute the inertia of " + entry.linkName,
              [link, geoms = MjcfGeoms(body.geoms.begin(), body.geoms.end()), warnings](
                  AsyncCancelToken const& cancel) {
                if (cancel.IsCancelRequested()) {
                  return false;
                }
                if (std::optional<GeomsInertia> const inertia =
                        ComputeGeomsInertia(geoms, *warnings)) {
                  link->mass = inertia->mass;
                  link->centerOfMass = inertia->centerOfMass;
                  link->momentOfInertia = inertia->momentOfInertia;
                }
                return true;
              }});
    }

    if (entry.renderOp == MjcfModelOperation::Convert) {
      MjcfGeoms geoms;
      for (int g : SelectGeoms(entry, entry.renderSource)) {
        geoms.push_back(body.geoms[g]);
      }
      if (!geoms.empty()) {
        mochi::Path const dst = outputPath(renderDir, entry.linkName, kGlbExtension);
        std::vector<std::string>* warnings = &_taskWarnings.emplace_back();
        needRenderDir = true;
        tasks.push_back(
            AsyncTask{
                "Build " + dst.GetFilename(),
                [link, geoms = std::move(geoms), dst, renderConverter, warnings](
                    AsyncCancelToken const& cancel) {
                  if (cancel.IsCancelRequested() ||
                      !BuildLinkGlb(geoms, dst, renderConverter, *warnings)) {
                    return false;
                  }
                  link->renderModelFile = dst.ToString();
                  link->renderModelScale = kUnitScale;
                  link->renderModelRotation = mochi::Quaternion::Identity();
                  link->renderModelTranslation = {};
                  return true;
                }});
      }
    }

    if (entry.mochiOp != MjcfModelOperation::Convert) {
      continue;
    }
    std::vector<int> const selected = SelectGeoms(entry, entry.mochiSource);
    if (selected.empty()) {
      continue;
    }
    needMochiDir = true;
    // Geoms chosen as collision geometry collide, whatever their MJCF bitmasks said.
    for (int g : selected) {
      MjcfGeomReference& geom = body.geoms[g];
      if (!geom.isCollision) {
        geom.isCollision = true;
        geom.contype = 1;
        geom.conaffinity = 1;
      }
    }
    if (entry.merge == MjcfCollisionMerge::Separate) {
      anySeparate = true;
      int k = 0;
      for (int g : selected) {
        MjcfGeomReference* geom = &body.geoms[g];
        mochi::Path const dst =
            outputPath(mochiDir, entry.linkName + "_" + std::to_string(k++), kMochiH5Extension);
        std::vector<std::string>* warnings = &_taskWarnings.emplace_back();
        tasks.push_back(
            AsyncTask{
                "Build " + dst.GetFilename(),
                [geom, dst, collisionOptions, warnings](AsyncCancelToken const& cancel) {
                  if (cancel.IsCancelRequested() ||
                      !BuildGeomCollisionH5(*geom, collisionOptions, dst, *warnings)) {
                    return false;
                  }
                  // The file holds the scaled mesh; ExpandMjcfGeomLinks applies only the pose.
                  geom->shapeFile = dst.ToString();
                  geom->meshScale = kUnitScale;
                  return true;
                }});
      }
      continue;
    }

    MjcfGeoms geoms;
    for (int g : selected) {
      geoms.push_back(body.geoms[g]);
    }
    LinkCollisionOptions options = collisionOptions;
    options.merge = entry.merge;
    mochi::Path const dst = outputPath(mochiDir, entry.linkName, kMochiH5Extension);
    std::vector<std::string>* warnings = &_taskWarnings.emplace_back();
    tasks.push_back(
        AsyncTask{
            "Build " + dst.GetFilename(),
            [link, geoms = std::move(geoms), dst, options, warnings](
                AsyncCancelToken const& cancel) {
              if (cancel.IsCancelRequested() ||
                  !BuildLinkCollisionH5(geoms, options, dst, *warnings)) {
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
    if (anySeparate) {
      superdex::robotics::ExpandMjcfGeomLinks(_botPrefab, _info, error);
      if (!error.IsOK()) {
        MOCHI_LOG_ERROR(
            "Failed to lay out the Separate Links collision geoms; the bot was not saved.");
        return false;
      }
    }
    superdex::robotics::ApplyMjcfContactFilter(_botPrefab, _info);
    std::string notes;
    for (auto const& warnings : _taskWarnings) {
      for (auto const& warning : warnings) {
        notes += "\n  " + warning;
      }
    }
    if (!notes.empty()) {
      MOCHI_LOG_WARNING("MJCF import of \"%s\":%s", _options.name.c_str(), notes.c_str());
    }
    superdex::robotics::SaveToFile(_botPrefab, outPath.ToString(), error);
    if (!error.IsOK()) {
      MOCHI_LOG_ERROR("Failed to save bot: %s", outPath.ToString().c_str());
      return false;
    }
    if (!superdex::robotics::FindBotsRoot(outPath.AsFilesystemPath()).has_value()) {
      _studio->CreateRootMarker(destDir, mochi::ErrorLog{});
    }
    outAssetPath = outPath;
    return true;
  };
  return true;
}

bool MjcfImporter::CanImport() {
  std::error_code ec;
  _nameConflicts = !_options.name.empty() &&
      std::filesystem::exists(GetOutputAssetPath().AsFilesystemPath(), ec);
  return _prefabValid && !_options.name.empty() && !_nameConflicts;
}

std::vector<std::string> MjcfImporter::SiblingFoldersToEnsure() const {
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

ImVec2 MjcfImporter::GetModalSize() const {
  return {960.0f, 1080.0f};
}

} // namespace superdex::studio
