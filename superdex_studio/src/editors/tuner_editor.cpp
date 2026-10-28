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

#include "editors/tuner_editor.h"

#include "app/app.h"
#include "assets/mochi_prefab_asset.h"
#include "assets/tuner_asset.h"
#include "rendering/debug_text.h"
#include "rendering/measure_tool.h"
#include "ui/imgui_widgets.h"

#include <superdex_asset_tuner/load_math.h>
#include <superdex_asset_tuner/scene_targets.h>
#include <superdex_asset_tuner/tuner_runner.h>

#include <mochi_core/utils/math_utils.h>
#include <mochi_physics/mochi_physics_experimental.h>
#include <mochi_renderer/debug.h>
#include <mochi_renderer/type_conversions.h>

#include <imguios/fonts/icons_font_awesome5.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <optional>

namespace superdex::studio {

namespace tuner = superdex::tuner;
using mochi::DynamicString;
using mochi::real;
using mochi::Real3;
using mochi_renderer::SceneObject;

namespace {

struct NamedOption {
  char const* name;
  char const* description;
};

constexpr std::array<NamedOption, 7> kProbeKinds{{
    {"Body Pose", "Position and rotation vector of the body origin, in world frame."},
    {"Body Velocity", "Linear and angular velocity of the body, in world frame."},
    {"Joint State", "Position and velocity of each of the joint's DoFs."},
    {"Contact Force", "Total contact force on the body, in world frame."},
    {"Solver Stats",
     "Nonlinear iterations, residual, line-search iterations and convergence status (0 none, 1 "
     "converged, 2 stopped, 3 diverged) of each step."},
    {"Performance", "Wall-clock solve time of each step. Can vary between runs."},
    {"Penetration",
     "Approximate volume (m^3) by which the body overlaps other actors: depth times contact area, "
     "summed over its overlapping contact points."},
}};
constexpr filament::math::float4 kSelectedLoadColor{1.0f, 1.0f, 0.2f, 1.0f};
constexpr ImVec4 kWarningColor{1.0f, 0.7f, 0.2f, 1.0f};
constexpr ImVec4 kErrorColor{1.0f, 0.45f, 0.35f, 1.0f};
// Plot history cap, so a long session cannot grow without bound (~1.5 h at 100 Hz).
constexpr size_t kMaxPlotSamples = 500'000;
constexpr double kTimeEpsilon = 1e-9;

// Joint Effort

// Joint Effort weights 1-3 DoFs (see TunerSession::ResolveLoad).
bool TakesJointEffort(int numDofs) {
  return numDofs >= 1 && numDofs <= 3;
}

void Tooltip(char const* text) {
  ImGui::ItemTooltipWrapped(text);
}

// Combo over `options`, with each option's description as its tooltip (and the current option's on
// the closed combo).
template <typename Enum, size_t N>
bool DescribedCombo(char const* label, Enum& value, std::array<NamedOption, N> const& options) {
  auto const current = static_cast<size_t>(value);
  bool changed = false;
  if (ImGui::BeginCombo(label, current < N ? options[current].name : "?")) {
    for (size_t i = 0; i < N; ++i) {
      if (ImGui::Selectable(options[i].name, i == current)) {
        value = static_cast<Enum>(i);
        changed = true;
      }
      Tooltip(options[i].description);
      if (i == current) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }
  if (current < N) {
    Tooltip(options[current].description);
  }
  return changed;
}

// Bubble tint for a card: the model editor's neutral bubble grey with a faint shade of `accent`,
// darker when disabled or dimmed (another card is selected), and lighter when selected.
ImU32 CardColor(ImVec4 const& accent, bool enabled, bool selected = false, bool dimmed = false) {
  ImVec4 const neutral = !enabled ? ImVec4(0.13f, 0.13f, 0.15f, 1.0f)
      : selected                  ? ImVec4(0.28f, 0.28f, 0.32f, 1.0f)
      : dimmed                    ? ImVec4(0.16f, 0.16f, 0.18f, 1.0f)
                                  : ImVec4(0.22f, 0.22f, 0.25f, 1.0f);
  float const tint = !enabled ? 0.12f : selected ? 0.32f : dimmed ? 0.16f : 0.25f;
  // Darkened so the tint shifts the hue more than it lifts the brightness.
  constexpr float kAccentScale = 0.6f;
  auto const mix = [&](float n, float a) { return n + (a * kAccentScale - n) * tint; };
  return ImGui::GetColorU32(
      ImVec4(mix(neutral.x, accent.x), mix(neutral.y, accent.y), mix(neutral.z, accent.z), 1.0f));
}

constexpr ImVec4 kNeutralAccent{0.35f, 0.35f, 0.45f, 1.0f};

// Moves element `from` to index `to`, shifting the ones between (works for any contiguous array).
template <typename Container>
void MoveItem(Container& items, size_t from, size_t to) {
  auto const begin = items.begin();
  if (from < to) {
    std::rotate(begin + from, begin + from + 1, begin + to + 1);
  } else if (from > to) {
    std::rotate(begin + to, begin + from, begin + from + 1);
  }
}

// Automatic ground height shown by DrawRunFieldOverride. The reflection widget takes a plain
// function pointer, so the value is passed through here (UI thread only), set just before drawing.
float sAutoGroundHeightForRunUi = 0.0f;

// Run Settings widget override: Ground Height shows the automatic height until overridden, and an
// override starts from it.
bool DrawRunFieldOverride(char const* label, SReflect::FieldTypeInfo const& field, void* fieldPtr) {
  if (std::string_view(field._name) != "groundHeight") {
    return false;
  }
  auto& value = *static_cast<std::optional<real>*>(fieldPtr);
  ImGui::PushID("groundHeight");
  bool isOverridden = value.has_value();
  if (ImGui::Checkbox("##override", &isOverridden)) {
    value = isOverridden ? std::optional<real>(sAutoGroundHeightForRunUi) : std::nullopt;
  }
  Tooltip(
      "Override the ground height. Unchecked: the prefab's lowest point, like the prefab editor.");
  ImGui::SameLine(0, ImGui::GetStyle().ItemInnerSpacing.x);
  ImGui::SetNextItemWidth(
      ImGui::CalcItemWidth() - ImGui::GetFrameHeight() - ImGui::GetStyle().ItemInnerSpacing.x);
  real shown = value.value_or(sAutoGroundHeightForRunUi);
  ImGui::BeginDisabled(!isOverridden);
  if (ImGui::DragReal(
          label, &shown, 0.001f, 0.0f, 0.0f, isOverridden ? "%.4f m" : "auto: %.4f m") &&
      isOverridden) {
    value = shown;
  }
  ImGui::EndDisabled();
  Tooltip(
      isOverridden ? "Ground plane height; the viewport ground moves with it."
                   : "Automatic: the lowest point of the prefab. Check the box to override.");
  ImGui::PopID();
  return true;
}

// Channel suffixes a probe produces (as in signals.csv), for its card body.
std::string ProbeChannelHint(tuner::ProbeKind kind) {
  switch (kind) {
    case tuner::ProbeKind::BodyPose:
      return "pos.{x,y,z}, rotvec.{x,y,z}, rotvecUnwrapped.{x,y,z}";
    case tuner::ProbeKind::BodyVelocity:
      return "vel.{x,y,z}, angvel.{x,y,z}";
    case tuner::ProbeKind::JointState:
      return "q<i>, qd<i> per DoF";
    case tuner::ProbeKind::ContactForce:
      return "contactForce.{x,y,z}";
    case tuner::ProbeKind::SolverStats:
      return "solver.iters, solver.residual, solver.lineSearchIters, solver.convergence";
    case tuner::ProbeKind::Performance:
      return "perf.solveMs";
    case tuner::ProbeKind::Penetration:
      return "penetration.volume";
  }
  return {};
}

bool ProbeHasTarget(tuner::ProbeKind kind) {
  return kind != tuner::ProbeKind::SolverStats && kind != tuner::ProbeKind::Performance;
}
// Dropdown of `candidates` with a filter box. A target that is not a candidate (e.g. renamed in the
// prefab) is kept and flagged rather than silently replaced.
bool TargetPicker(
    char const* label,
    DynamicString& target,
    std::vector<std::string> const& candidates) {
  bool const known =
      std::find(candidates.begin(), candidates.end(), std::string_view(target)) != candidates.end();
  std::string const preview = target.empty() ? std::string("(pick)")
      : known                                ? std::string(std::string_view(target))
              : std::string(std::string_view(target)) + "  (not found)";
  if (!known && !target.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.45f, 0.35f, 1.0f));
  }
  bool const open = ImGui::BeginCombo(label, preview.c_str(), ImGuiComboFlags_HeightLarge);
  if (!known && !target.empty()) {
    ImGui::PopStyleColor();
  }
  if (!open) {
    return false;
  }
  // Only one combo is open at a time, so one filter serves every picker.
  static ImGuiTextFilter filter;
  if (ImGui::IsWindowAppearing()) {
    filter.Clear();
    ImGui::SetKeyboardFocusHere();
  }
  filter.Draw("##filter", -FLT_MIN);
  bool changed = false;
  for (std::string const& candidate : candidates) {
    if (!filter.PassFilter(candidate.c_str())) {
      continue;
    }
    if (ImGui::Selectable(candidate.c_str(), std::string_view(target) == candidate)) {
      target = candidate;
      changed = true;
    }
  }
  if (candidates.empty()) {
    ImGui::TextDisabled("No valid targets in this prefab.");
  }
  ImGui::EndCombo();
  return changed;
}

} // namespace

//--------------------------------------------------------------------------------------------------
// AssetEditor
//--------------------------------------------------------------------------------------------------

TunerEditor::TunerEditor(SuperDexStudio* studio, TunerAsset* asset)
    : AssetEditor(studio, asset),
      _tunerAsset(asset),
      _stage(studio, "TunerEditorStage"),
      _simSync(studio->GetEditorToRendererSpaceConverter()),
      _outputDir((asset->GetPath().GetParentPath().AsFilesystemPath() / "tuner_runs").string()) {}

void TunerEditor::Initialize() {
  if (!_tunerAsset->IsReadOnly()) {
    _undoStack.Initialize(
        [this] { return TakeUndoSnapshot(); },
        [this](std::string const& json, int) { RestoreUndoSnapshot(json); });
  }
  _viewport = Viewport::Create(_studio, _studio->GetViewSettings());
  // The gizmo edits the box of the selected box stop condition (move and scale; boxes stay
  // axis-aligned).
  _viewport->showTransformGizmoTarget = [this] {
    return !_mochiScene.IsSimulating() && SelectedStopBoxIndex() >= 0;
  };
  _viewport->getTransformGizmoTarget =
      [this]() -> std::optional<std::pair<mochi::TransformRT, mochi::Real3>> {
    int const box = SelectedStopBoxIndex();
    if (box < 0) {
      return std::nullopt;
    }
    tuner::StopCondition const& stop =
        _tunerAsset->GetSpec().stopConditions[static_cast<size_t>(box)];
    Real3 const center = (stop.boxMin + stop.boxMax) * real{0.5};
    Real3 const size = stop.boxMax - stop.boxMin;
    return std::pair{mochi::TransformRT(mochi::Quaternion::Identity(), center), size};
  };
  _viewport->onTransformGizmoDelta =
      [this](mochi::TransformRT const& delta, mochi::Real3 const& scale) {
        int const box = SelectedStopBoxIndex();
        if (box < 0) {
          return;
        }
        tuner::StopCondition& stop =
            _tunerAsset->GetSpec().stopConditions[static_cast<size_t>(box)];
        Real3 const center = delta.TransformPoint((stop.boxMin + stop.boxMax) * real{0.5});
        Real3 half = (stop.boxMax - stop.boxMin) * real{0.5};
        constexpr real kMinHalfExtent = 1e-4f;
        for (int d = 0; d < 3; ++d) {
          half[d] = std::max(half[d] * scale[d], kMinHalfExtent);
        }
        stop.boxMin = center - half;
        stop.boxMax = center + half;
        OnSpecEdited(false);
      };
  // Clicking in the viewport deselects the stop condition box.
  _viewport->onSceneSelectionChanged = [this](std::vector<SceneObject*> const&) {
    _selectedStopCard = 0;
  };
  _viewport->RegisterShowCommand(
      {.name = "Physics Debug Draw",
       .onToggle = [this] { _mochiScene.ToggleDebugDraw(); },
       .getState = [this] { return _mochiScene.IsDebugDrawEnabled(); },
       .shortcut = ImGuiKey_P});
  auto const registerStageType = [this](char const* name, StageType type, ImGuiKey key) {
    _viewport->RegisterShowCommand(
        {.name = name,
         .onToggle =
             [this, type] {
               _stageType = type;
               Restage();
             },
         .getState = [this, type] { return _stageType == type; },
         .shortcut = key});
  };
  registerStageType("Render Only", StageType::RenderModelOnly, ImGuiKey_1);
  registerStageType("Collision Only", StageType::MochiModelOnly, ImGuiKey_2);
  registerStageType(
      "Render (Collision Fallback)", StageType::RenderModelFallbackToMochiModel, ImGuiKey_3);

  _mochiScene.Initialize(_studio->GetMochiContext(), "TunerEditor");
  ApplyRunSettings();
  _mochiScene.createPhysicsActors = [this](mochi::Scene* s) { CreatePhysicsActors(s); };
  _mochiScene.destroyPhysicsActors = [this](mochi::Scene* s) { DestroyPhysicsActors(s); };
  // Runs on the UI thread once the actors exist: start the per-session drag controller and reset
  // the UI view of the session.
  _mochiScene.registerPreStepCallback = [this](mochi::AsyncScene*) {
    _dragController = std::make_unique<PhysicsDragController>(
        _mochiScene,
        &_stage,
        _studio->GetAppSettings().physicsDrag,
        _studio->GetRendererToEditorSpaceConverter());
    auto const loads = _session.GetLoads();
    _boundLoads.assign(loads.begin(), loads.end());
    _channels.clear();
    for (auto const& name : _session.GetChannelNames()) {
      ChannelHistory& channel = _channels.emplace_back();
      channel.name = std::string(std::string_view(name));
    }
    _sampleTimes.clear();
    _hasSample = false;
    _stopEvent.reset();
    return mochi::CallbackHandle{};
  };
  _mochiScene.registerPostStepCallback = [this](mochi::AsyncScene* scene) {
    return scene->RegisterPostStepCallback(
        "TunerEditor::ExtractActors",
        [this](mochi::StepInfo const& info) { _simSync.Produce(info.scene); });
  };
  _mochiScene.onStopPhysics = [this]() { OnStopPhysics(); };
  BindSceneObjectDragHooks(*_viewport, _dragController);
  _stage.BindRenderScene(_viewport->GetRenderScene());
  _viewport->SetSceneStage(&_stage);
  BindSceneStageMeasureTargets(
      *_viewport,
      _stage,
      [this] { return _mochiScene.IsSimulating(); },
      [this] { return _mochiScene.IsPaused(); });
  Restage();
  _viewport->FocusCameraOnScene();
}

void TunerEditor::OnHandleInputs() {
  if (_tunerAsset->GetPrefabAsset() != nullptr) {
    _mochiScene.HandleHotkeys();
  }
}

void TunerEditor::OnRender(Renderer const* renderer) {
  if (_mochiScene.IsSimulating()) {
    _mochiScene.UpdateStats();
    _simSync.Apply(_stage, _studio->GetEditorToRendererSpaceConverter());
    DrainSamples();
    // A tuner whose targets do not resolve would silently run without its loads.
    if (!_bindError.empty()) {
      MOCHI_LOG_ERROR("Asset Tuner: %s", _bindError.c_str());
      _mochiScene.DestroyMochiScene();
    }
  }
  auto* debugDraw = _viewport->GetRenderScene()->GetDebugDraw();
  _mochiScene.DrawDebug(debugDraw, _studio->GetEditorToRendererSpaceConverter());
  if (_dragController) {
    _dragController->DrawDebug(
        debugDraw, _viewport->GetDebugText(), _studio->GetEditorToRendererSpaceConverter());
  }
  DrawStopBoxes();
  _viewport->RenderScene(renderer);
}

void TunerEditor::Shutdown() {
  if (_mochiScene.IsSimulating()) {
    _mochiScene.DestroyMochiScene();
  }
  _stage.Clear();
  _viewport.reset();
}

void TunerEditor::OnActivate() {
  if (!_mochiScene.IsSimulating()) {
    Restage();
  }
}

void TunerEditor::OnDeactivate() {
  if (_mochiScene.IsSimulating()) {
    _mochiScene.DestroyMochiScene();
  }
}

void TunerEditor::Refresh() {
  if (_mochiScene.IsSimulating()) {
    _mochiScene.DestroyMochiScene();
  }
  Restage();
}

void TunerEditor::ShowTabContents() {
  ImGui::BeginChild("Viewport_Child", ImVec2(0, 0), 0, ImGuiWindowFlags_NoMove);
  _viewport->enableSceneObjectDrag = _mochiScene.IsSimulating();
  _viewport->ShowViewportContents(true);
  _viewport->ShowStatsOverlay(
      _mochiScene.GetStepsPerSecond(),
      _hasSample ? std::optional<float>(static_cast<float>(_lastSample.time)) : std::nullopt);
  ImGui::BeginDisabled(_tunerAsset->GetPrefabAsset() == nullptr);
  _mochiScene.ShowPlayToolbarOverViewport();
  ImGui::EndDisabled();
  ShowStopBanner();
  if (IsRecording()) {
    // Bottom-right, clear of the stats overlay in the bottom-left corner.
    std::string const status = "REC";
    constexpr float kMargin = 12.0f;
    ImVec2 const size = ImGui::CalcTextSize(status.c_str());
    ImVec2 const pos{
        ImGui::GetWindowWidth() - size.x - kMargin, ImGui::GetWindowHeight() - size.y - kMargin};
    ImVec2 const screen{ImGui::GetWindowPos().x + pos.x, ImGui::GetWindowPos().y + pos.y};
    ImVec2 const pad = ImGui::GetStyle().FramePadding;
    ImGui::GetWindowDrawList()->AddRectFilled(
        ImVec2(screen.x - pad.x, screen.y - pad.y),
        ImVec2(screen.x + size.x + pad.x, screen.y + size.y + pad.y),
        ImGui::GetColorU32(ImGuiCol_Button),
        size.y * 0.5f + pad.y);
    ImGui::SetCursorPos(pos);
    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "REC");
  }
  ImGui::EndChild();
}

std::vector<AssetEditor::WindowDeclaration> TunerEditor::GetDefaultWindows() {
  using Dock = AssetEditor::DockRegion;
  return {
      {"Asset Tuner Run", true, Dock::SidePanelTop, false},
      {"Asset Tuner Prefab Overrides", false, Dock::SidePanelBottom, false},
      {"Physics Settings", false, Dock::SidePanelBottom, false},
      MeasureWindowDeclaration()};
}

std::vector<AssetEditor::WindowDeclaration> TunerEditor::GetAuxiliaryWindows() const {
  return GetDefaultWindows();
}

void TunerEditor::ShowAuxiliaryWindows() {
  RefreshTargets();
  if (bool& open = _studio->GetWindowVisible("Asset Tuner Run")) {
    ShowRunWindow(&open);
  }
  if (bool& open = _studio->GetWindowVisible("Asset Tuner Prefab Overrides")) {
    ShowOverridesWindow(&open);
  }
  if (bool& open = _studio->GetWindowVisible("Physics Settings")) {
    ShowPhysicsSettingsWindow(&open);
  }
  ShowMeasureWindow();
}

// Only debug draw is editable here: the time step, gravity, solver and ground come from the tuner
// (Asset Tuner Run), so a Studio run matches the CLI.
void TunerEditor::ShowPhysicsSettingsWindow(bool* open) {
  if (!ImGui::Begin("Physics Settings", open)) {
    ImGui::End();
    return;
  }
  ImGui::TextWrapped(
      "Time step, Gravity and Ground Plane settings are in the Asset Tuner Run tab.\n"
      "Solver settings are set from the prefab.");
  if (ImGui::CollapsingHeader("Debug Draw", ImGuiTreeNodeFlags_DefaultOpen)) {
    _mochiScene.ShowDebugDrawSection();
  }
  ImGui::End();
}

bool TunerEditor::CanUndoRedo() const {
  return !_mochiScene.IsSimulating();
}

void TunerEditor::ApplySceneViewSettings(mochi_renderer::SceneViewSettings const& viewSettings) {
  if (_viewport && _viewport->GetRenderScene()) {
    _viewport->GetRenderScene()->ApplyViewSettings(viewSettings);
  }
}

void TunerEditor::OnAppSettingsChanged(AppSettings const& settings) {
  if (_dragController) {
    _dragController->SetSettings(settings.physicsDrag);
  }
}

void TunerEditor::ForEachReferencedPath(
    std::function<void(mochi::Path const&)> const& callback) const {
  AssetEditor::ForEachReferencedPath(callback);
  callback(_tunerAsset->GetPrefabPath());
}

//--------------------------------------------------------------------------------------------------
// Staging and simulation
//--------------------------------------------------------------------------------------------------

void TunerEditor::Restage() {
  _describedTargetsStale = true;
  MochiPrefabAsset* prefabAsset = _tunerAsset->GetPrefabAsset();
  if (prefabAsset == nullptr) {
    _stage.Clear();
    return;
  }
  // Nested prefabs edited in their own editors since the prefab was loaded (saved or not).
  ResolveNestedPrefabsFromMemory(
      prefabAsset->GetPrefab(), _studio->GetAssetManager(), prefabAsset->GetAssetsRoot());
  ValidateOverrides();
  _stage.StagePrefab(prefabAsset->GetPrefab(), prefabAsset->GetAssetsRoot(), _stageType);
  // The ground stays put mid-run (it only changes the view there).
  if (_mochiScene.IsSimulating()) {
    return;
  }
  // As in the prefab editor: the automatic ground sits at the lowest point of what is shown.
  _autoGroundHeight = _viewport->UpdateGroundPlane();
  ApplyRunSettings();
}

void TunerEditor::RefreshTargets() {
  // Wait until no widget is being edited. Otherwise typing an override would build a new scene on
  // every key press.
  if (!_describedTargetsStale || ImGui::IsAnyItemActive()) {
    return;
  }
  _describedTargetsStale = false;
  _describedTargets.reset();
  _targets = {};
  if (_tunerAsset->GetPrefabAsset() == nullptr) {
    return;
  }
  // In a temporary scene of the studio's context (never a second context; see DescribeTuner).
  tuner::SceneTargets described;
  std::string error;
  if (!tuner::DescribeTuner(
          _tunerAsset->GetSpec(),
          _tunerAsset->GetPath().AsFilesystemPath(),
          _studio->GetMochiContext(),
          described,
          error)) {
    MOCHI_LOG_ERROR("Asset Tuner: could not load the prefab into Mochi: %s", error.c_str());
    return;
  }
  // Unnamed actors cannot be targeted by name.
  for (tuner::RigidTarget const& rigid : described.rigid) {
    if (!rigid.isStatic && !rigid.name.empty()) {
      _targets.rigidBodies.emplace_back(std::string_view(rigid.name));
    }
  }
  for (tuner::ArticulatedTarget const& articulated : described.articulated) {
    std::string const actor(std::string_view(articulated.name));
    for (size_t i = 0; i < articulated.links.size(); ++i) {
      if (actor.empty() || articulated.links[i].empty()) {
        continue;
      }
      std::string link = actor + "/" + std::string(std::string_view(articulated.links[i]));
      if (i < articulated.joints.size() && !articulated.joints[i].name.empty()) {
        _targets.joints.push_back(
            {.name = actor + "/" + std::string(std::string_view(articulated.joints[i].name)),
             .link = link,
             .numDofs = articulated.joints[i].dofCount});
      }
      _targets.links.push_back(std::move(link));
    }
  }
  _describedTargets = std::move(described);
}

// Mirrors the superdex_asset_tuner CLI setup so a run in Studio matches the same run on the command
// line: fixed dt, default gravity and solver (the prefab and RunSpec override them), and the
// RunSpec ground plane. Debug draw is only visual, so the session's own choices are kept.
void TunerEditor::ApplyRunSettings() {
  tuner::RunSpec const& run = _tunerAsset->GetSpec().run;
  PhysicsSettings settings = _studio->GetAppSettings().physics;
  if (_debugSettingsInitialized) {
    settings.debug = _mochiScene.GetDebugSettings();
  }
  _debugSettingsInitialized = true;
  settings.scene.gravity = tuner::kDefaultGravity;
  settings.scene.useFixedTimeStep = true;
  settings.scene.fixedTimeStepSeconds = run.dt;
  settings.scene.throttleToRealTime = true;
  settings.solver = mochi::SolverParams{};
  settings.studio.groundPlane = run.groundPlane;
  _mochiScene.SetSettings(settings);
  // The viewport ground and the physics plane always share one height, so what is drawn is what the
  // bodies land on. Not moved mid-run (the settings are locked while simulating).
  float const groundHeight = EffectiveGroundHeight();
  _mochiScene.SetGroundPlaneHeight(groundHeight);
  _mochiScene.SetGroundPlaneContact(run.groundContact.value_or(mochi::ContactParams{}));
  if (_viewport && !_mochiScene.IsSimulating()) {
    _viewport->SetGroundPlaneHeight(groundHeight);
  }
}

float TunerEditor::EffectiveGroundHeight() const {
  auto const& override = _tunerAsset->GetSpec().run.groundHeight;
  return override ? static_cast<float>(*override) : _autoGroundHeight;
}

// Sim thread; the UI thread is blocked in MochiAsyncScene::CreateScene.
void TunerEditor::CreatePhysicsActors(mochi::Scene* scene) {
  _bindError.clear();
  _stopPausedOnSim = false;
  // The context is shared by the whole app. Studio runs one simulation at a time, which lets this
  // run set the threading mode. A run that fails to bind puts it back here, any other run in
  // DestroyPhysicsActors.
  mochi::Context* context = _studio->GetMochiContext();
  _savedSingleThreaded = context->IsSingleThreaded();
  context->SetIsSingleThreaded(_tunerAsset->GetSpec().run.singleThreaded);
  if (!BindPhysicsActors(scene)) {
    RestoreThreadingMode();
  }
}

// Sim thread, from CreatePhysicsActors.
bool TunerEditor::BindPhysicsActors(mochi::Scene* scene) {
  MochiPrefabAsset* prefabAsset = _tunerAsset->GetPrefabAsset();
  if (prefabAsset == nullptr) {
    _bindError = "Prefab '" + _tunerAsset->GetPrefabPath().ToString() + "' is not loaded.";
    return false;
  }
  tuner::TunerSpec const& spec = _tunerAsset->GetSpec();
  mochi::prefab::ScenePrefab overridden;
  mochi::prefab::ScenePrefab* prefab = &prefabAsset->GetPrefab();
  if (!spec.prefabOverrides.empty()) {
    overridden = *prefab;
    if (!tuner::ApplyPrefabOverrides(overridden, spec.prefabOverrides, _bindError)) {
      return false;
    }
    prefab = &overridden;
  }
  mochi::ErrorLog error;
  mochi::prefab::EnsureFullyLoaded(
      *prefab, prefabAsset->GetAssetsRoot(), _studio->GetMochiContext(), error);
  auto const result = mochi::prefab::AddToScene(*prefab, scene, {}, error);
  if (!error.IsOK()) {
    _bindError = "Failed to add the prefab to the scene (see log).";
    return false;
  }
  tuner::ApplyRunSpecOverrides(*scene, spec.run);
  _simSync.Track(mochi::MakeConstSpan(result.actors));
  if (!_session.Bind(*scene, spec, _bindError)) {
    return false;
  }
  _loadsCallback = scene->RegisterPreStepCallback(
      "TunerEditor::Loads", [this](mochi::StepInfo const& info) { _session.OnPreStep(info); });
  _probesCallback = scene->RegisterPostStepCallback(
      "TunerEditor::Probes", [this, sample = tuner::Sample{}](mochi::StepInfo const& info) mutable {
        _session.OnPostStep(info, sample);
        // Pause on this step, like the CLI stops on it, so the scene the viewport shows is the one
        // that triggered. Only the first trigger pauses; resuming runs on without stop checks.
        if (sample.stopCondition >= 0 && !_stopPausedOnSim.exchange(true)) {
          _mochiScene.GetAsyncScene()->Pause(true);
        }
        std::lock_guard const lock(_samplesMutex);
        _pendingSamples.push_back(sample);
      });
  if (_recordNextRun && OpenRunWriter()) {
    mochi::experimental::RecordingParams recording;
    recording.recordContactPoints = spec.run.record.contactPoints;
    recording.recordStaticActorState = spec.run.record.staticActors;
    mochi::Error recordError;
    mochi::experimental::StartRecording(
        scene, _writer.GetSimRecordingPath().string(), recording, recordError);
    _runSummary.simRecording = recordError.IsOK();
  }
  return true;
}

// Sim thread; the UI thread is blocked in MochiAsyncScene::DestroyMochiScene.
void TunerEditor::DestroyPhysicsActors(mochi::Scene* scene) {
  if (mochi::experimental::IsRecording(scene, mochi::ErrorAssert{})) {
    mochi::experimental::StopRecording(scene, mochi::ErrorAssert{});
  }
  for (mochi::CallbackHandle* handle : {&_loadsCallback, &_probesCallback}) {
    if (handle->IsValid()) {
      scene->CancelCallback(*handle);
      *handle = {};
    }
  }
  _session.Unbind(*scene);
  _simSync.Untrack(scene);
  RestoreThreadingMode();
}

void TunerEditor::RestoreThreadingMode() {
  if (_savedSingleThreaded) {
    _studio->GetMochiContext()->SetIsSingleThreaded(*_savedSingleThreaded);
    _savedSingleThreaded.reset();
  }
}

void TunerEditor::OnStopPhysics() {
  _dragController.reset();
  DrainSamples();
  StopRecording();
  _simSync.Reset();
  _stage.ResetWorldTransforms(_studio->GetEditorToRendererSpaceConverter());
  _hasSample = false;
  _stopEvent.reset();
  _bindError.clear();
}

void TunerEditor::DrainSamples() {
  std::vector<tuner::Sample> samples;
  {
    std::lock_guard const lock(_samplesMutex);
    samples.swap(_pendingSamples);
  }
  double const duration = _tunerAsset->GetSpec().run.duration;
  bool stopNow = false;
  for (tuner::Sample& sample : samples) {
    bool const inRun = !_stopAtDuration || sample.time <= duration + kTimeEpsilon;
    // The first triggered stop condition ends the run like on the CLI: the sim thread paused on
    // this step, and the recording ends with this sample.
    bool const stopsHere = sample.stopCondition >= 0 && !_stopEvent;
    if (IsRecording() && inRun) {
      _writer.WriteSample(sample);
      tuner::UpdateChannelSummaries(
          sample.values, _runSummary.steps == 0, mochi::MakeSpan(_runSummary.channels));
      ++_runSummary.steps;
      _runSummary.simTime = sample.time;
    }
    if (stopsHere) {
      _stopEvent = StopEvent{
          .condition = sample.stopCondition,
          .message = std::string(std::string_view(sample.stopMessage)),
          .time = sample.time};
      auto const& stops = _tunerAsset->GetSpec().stopConditions;
      if (IsRecording()) {
        _runSummary.status = "stopped";
        _runSummary.message = sample.stopMessage;
        if (sample.stopCondition < static_cast<int>(stops.size())) {
          _runSummary.stopCondition = stops[static_cast<size_t>(sample.stopCondition)].name;
        }
      }
      MOCHI_LOG("Asset Tuner stopped at t = %.3f s: %s", sample.time, sample.stopMessage.c_str());
      stopNow = true;
    }
    if (_sampleTimes.size() < kMaxPlotSamples && sample.values.size() == _channels.size()) {
      _sampleTimes.push_back(static_cast<float>(sample.time));
      for (size_t i = 0; i < _channels.size(); ++i) {
        _channels[i].values.push_back(static_cast<float>(sample.values[i]));
      }
    }
    _lastSample = std::move(sample);
    _hasSample = true;
    if (stopNow) {
      break; // keep the stop sample as the last one shown
    }
  }
  if (stopNow) {
    StopRecording();
  }
  if (_stopAtDuration && _hasSample && _lastSample.time >= duration - kTimeEpsilon &&
      _mochiScene.IsSimulating() && !_mochiScene.IsPaused()) {
    _mochiScene.Pause(true);
    StopRecording();
  }
}

//--------------------------------------------------------------------------------------------------
// Edits
//--------------------------------------------------------------------------------------------------

void TunerEditor::OnSpecEdited(bool structural) {
  _tunerAsset->SetDirty(true);
  if (structural) {
    GetUndoStack().PushNow();
  } else {
    GetUndoStack().MarkEdited();
  }
  if (!_mochiScene.IsSimulating()) {
    ApplyRunSettings();
    return;
  }
  if (LiveLoadsCompatible()) {
    _mochiScene.GetAsyncScene()->QueueCommand(
        [this, loads = _tunerAsset->GetSpec().loads](mochi::Scene*) {
          std::string error;
          _session.UpdateLoads(loads, error);
        });
  }
}

bool TunerEditor::LiveLoadsCompatible() const {
  auto const& loads = _tunerAsset->GetSpec().loads;
  if (loads.size() != _boundLoads.size()) {
    return false;
  }
  for (size_t i = 0; i < loads.size(); ++i) {
    // A pose path's drive is attached at its point, so moving the point needs a new run.
    if (loads[i].name != _boundLoads[i].name || loads[i].type != _boundLoads[i].type ||
        loads[i].target != _boundLoads[i].target ||
        (loads[i].type == tuner::LoadType::PosePath &&
         loads[i].pointLocal != _boundLoads[i].pointLocal)) {
      return false;
    }
  }
  return true;
}

std::string TunerEditor::TakeUndoSnapshot() const {
  return tuner::SaveTunerSpecToJson(_tunerAsset->GetSpec());
}

void TunerEditor::RestoreUndoSnapshot(std::string const& json) {
  tuner::TunerSpec spec;
  std::string error;
  if (!tuner::LoadTunerSpecFromJson(json, {}, spec, error)) {
    MOCHI_LOG_ERROR("Asset Tuner undo failed: %s", error.c_str());
    return;
  }
  _tunerAsset->GetSpec() = std::move(spec);
  _tunerAsset->SetDirty(!GetUndoStack().IsAtSavedState());
  ValidateOverrides();
  ApplyRunSettings();
}

void TunerEditor::ValidateOverrides() {
  // This runs when the prefab or its overrides change. Masses may have changed too.
  _describedTargetsStale = true;
  auto const& overrides = _tunerAsset->GetSpec().prefabOverrides;
  _overrideErrors.assign(overrides.size(), {});
  MochiPrefabAsset const* prefabAsset = _tunerAsset->GetPrefabAsset();
  if (prefabAsset == nullptr) {
    return;
  }
  // Each override is checked alone so every broken one gets its own message.
  for (size_t i = 0; i < overrides.size(); ++i) {
    if (overrides[i].path.empty()) {
      _overrideErrors[i] = "Empty path.";
      continue;
    }
    mochi::prefab::ScenePrefab copy = prefabAsset->GetPrefab();
    tuner::ApplyPrefabOverrides(
        copy, Span<tuner::PrefabOverride const>(&overrides[i], 1), _overrideErrors[i]);
  }
}

void TunerEditor::CardListState::Resize(size_t count) {
  while (cards.size() < count) {
    cards.push_back({.id = nextId++, .collapsed = true});
  }
  cards.resize(count);
}

void TunerEditor::CardListState::AddExpanded() {
  cards.push_back({.id = nextId++, .collapsed = false});
}

//--------------------------------------------------------------------------------------------------
// Recording
//--------------------------------------------------------------------------------------------------

bool TunerEditor::OpenRunWriter() {
  // Record the ground height actually used, so rerunning the resolved spec reproduces this run.
  tuner::TunerSpec spec = _tunerAsset->GetSpec();
  if (spec.run.groundPlane) {
    spec.run.groundHeight = EffectiveGroundHeight();
  }
  _runSummary = {};
  for (auto const& name : _session.GetChannelNames()) {
    tuner::ChannelSummary channel;
    channel.name = name;
    _runSummary.channels.push_back(std::move(channel));
  }
  std::string error;
  if (!_writer.Open(
          std::filesystem::path(_outputDir),
          _tunerAsset->GetName(),
          "studio",
          spec,
          _tunerAsset->GetPrefabPath().AsFilesystemPath(),
          error)) {
    MOCHI_LOG_ERROR("Asset Tuner recording: %s", error.c_str());
    return false;
  }
  _writer.WriteHeader(_session.GetChannelNames());
  return true;
}

void TunerEditor::StartRecording() {
  if (IsRecording() || !_mochiScene.IsSimulating() || !OpenRunWriter()) {
    return;
  }
  tuner::RecordSpec const& record = _tunerAsset->GetSpec().run.record;
  mochi::experimental::RecordingParams recording;
  recording.recordContactPoints = record.contactPoints;
  recording.recordStaticActorState = record.staticActors;
  mochi::AsyncScene* asyncScene = _mochiScene.GetAsyncScene();
  asyncScene->QueueCommand(
      [this, path = _writer.GetSimRecordingPath().string(), recording](mochi::Scene* scene) {
        mochi::Error error;
        mochi::experimental::StartRecording(scene, path, recording, error);
        _runSummary.simRecording = error.IsOK();
      });
  asyncScene->WaitForQueuedCommands();
}

void TunerEditor::StopRecording() {
  if (!IsRecording()) {
    return;
  }
  if (_mochiScene.IsSimulating()) {
    mochi::AsyncScene* asyncScene = _mochiScene.GetAsyncScene();
    asyncScene->QueueCommand([](mochi::Scene* scene) {
      if (mochi::experimental::IsRecording(scene, mochi::ErrorAssert{})) {
        mochi::experimental::StopRecording(scene, mochi::ErrorAssert{});
      }
    });
    asyncScene->WaitForQueuedCommands();
  }
  std::string error;
  if (!_writer.Close(_runSummary, error)) {
    MOCHI_LOG_ERROR("Asset Tuner recording: %s", error.c_str());
  }
  _lastRunDir = std::string(std::string_view(_runSummary.runDir));
  MOCHI_LOG("Asset Tuner run recorded to %s", _lastRunDir.c_str());
}

//--------------------------------------------------------------------------------------------------
// Viewport
//--------------------------------------------------------------------------------------------------

StagedActor const* TunerEditor::FindStagedBody(std::string_view target) const {
  // A joint target is drawn on the link it drives.
  for (auto const& joint : _targets.joints) {
    if (joint.name == target) {
      target = joint.link;
      break;
    }
  }
  int const index = _stage.FindIndexByName(target);
  return index < 0 ? nullptr : &_stage.GetActors()[static_cast<size_t>(index)];
}

std::vector<std::string> TunerEditor::GetTargetCandidates(tuner::LoadType type) const {
  switch (type) {
    case tuner::LoadType::Force:
    case tuner::LoadType::Torque:
    case tuner::LoadType::PosePath: {
      std::vector<std::string> bodies = _targets.rigidBodies;
      bodies.insert(bodies.end(), _targets.links.begin(), _targets.links.end());
      return bodies;
    }
    case tuner::LoadType::JointEffort: {
      std::vector<std::string> joints;
      for (auto const& joint : _targets.joints) {
        if (TakesJointEffort(joint.numDofs)) {
          joints.push_back(joint.name);
        }
      }
      return joints;
    }
  }
  return {};
}

int TunerEditor::SelectedStopBoxIndex() const {
  if (_selectedStopCard == 0) {
    return -1;
  }
  auto const& stops = _tunerAsset->GetSpec().stopConditions;
  auto const& cards = _stopCards.cards;
  for (size_t i = 0; i < cards.size() && i < stops.size(); ++i) {
    if (cards[i].id == _selectedStopCard) {
      return stops[i].kind != tuner::StopKind::ChannelThreshold ? static_cast<int>(i) : -1;
    }
  }
  return -1;
}

void TunerEditor::DrawStopBoxes() {
  auto* debugDraw = _viewport->GetRenderScene()->GetDebugDraw();
  auto const& stops = _tunerAsset->GetSpec().stopConditions;
  if (debugDraw == nullptr || stops.empty()) {
    return;
  }
  using filament::math::float3;
  using filament::math::float4;
  auto const& toRenderer = _studio->GetEditorToRendererSpaceConverter();
  auto const point = [&](Real3 const& p) {
    return mochi_renderer::ToFilament<float>(toRenderer.TranslationToOutput(p));
  };
  filament::math::double3 const camera = _viewport->GetRenderScene()->GetCameraPosition();
  float3 const cameraPos{
      static_cast<float>(camera.x), static_cast<float>(camera.y), static_cast<float>(camera.z)};
  int const selected = SelectedStopBoxIndex();
  DebugText* debugText = _viewport->GetDebugText();
  for (size_t i = 0; i < stops.size(); ++i) {
    tuner::StopCondition const& stop = stops[i];
    if (!stop.enabled) {
      continue;
    }
    bool const triggered = _stopEvent && _stopEvent->condition == static_cast<int>(i);
    if (stop.kind == tuner::StopKind::ChannelThreshold) {
      // Position thresholds ("<body>.pos.x|y|z") are drawn as a translucent plane at the threshold,
      // centered on the body; other channels have no place in the scene.
      std::string_view const channel = stop.channel;
      constexpr std::string_view kPos = ".pos.";
      size_t const at = channel.rfind(kPos);
      if (at == std::string_view::npos || at + kPos.size() + 1 != channel.size()) {
        continue;
      }
      int const axis = channel.back() - 'x';
      StagedActor const* body = FindStagedBody(channel.substr(0, at));
      if (axis < 0 || axis > 2 || body == nullptr) {
        continue;
      }
      Real3 center = body->worldTransform.GetTranslation();
      center[axis] = stop.threshold;
      int const u = (axis + 1) % 3;
      int const v = (axis + 2) % 3;
      float const half = 0.12f * length(cameraPos - point(center));
      auto const corner = [&](float su, float sv) {
        Real3 p = center;
        p[u] += su * half;
        p[v] += sv * half;
        return point(p);
      };
      std::array<float3, 4> const quad{corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)};
      float4 const fill =
          triggered ? float4{1.0f, 0.2f, 0.15f, 0.30f} : float4{1.0f, 0.65f, 0.1f, 0.18f};
      float4 const edge =
          triggered ? float4{1.0f, 0.2f, 0.15f, 1.0f} : float4{1.0f, 0.65f, 0.1f, 0.9f};
      // Both windings, so the plane shows from either side.
      debugDraw->DrawSolidTriangle(quad[0], quad[1], quad[2], fill);
      debugDraw->DrawSolidTriangle(quad[0], quad[2], quad[3], fill);
      debugDraw->DrawSolidTriangle(quad[0], quad[2], quad[1], fill);
      debugDraw->DrawSolidTriangle(quad[0], quad[3], quad[2], fill);
      for (size_t e = 0; e < 4; ++e) {
        float3 const& p0 = quad[e];
        float3 const& p1 = quad[(e + 1) % 4];
        debugDraw->DrawSolidCylinder(
            p0, p1 - p0, 0.0014f * length(cameraPos - p0), edge, 8, true, false);
      }
      char label[160];
      std::snprintf(
          label,
          sizeof(label),
          ICON_FA_RULER_VERTICAL " %s: %s %s %.3g",
          stop.name.c_str(),
          stop.channel.c_str(),
          stop.comparison == tuner::StopComparison::Above ? ">" : "<",
          static_cast<double>(stop.threshold));
      debugText->Draw(quad[2], label, edge, ImVec2{0, -14});
      continue;
    }
    bool const isSelected = static_cast<int>(i) == selected;
    float4 const color = triggered ? float4{1.0f, 0.2f, 0.15f, 1.0f}
        : isSelected               ? kSelectedLoadColor
                                   : float4{1.0f, 0.65f, 0.1f, 0.9f};
    std::array<float3, 8> corners;
    for (int c = 0; c < 8; ++c) {
      Real3 const p{
          (c & 1) ? stop.boxMax[0] : stop.boxMin[0],
          (c & 2) ? stop.boxMax[1] : stop.boxMin[1],
          (c & 4) ? stop.boxMax[2] : stop.boxMin[2]};
      corners[static_cast<size_t>(c)] = point(p);
    }
    // Edges as thin tubes (camera-scaled like the load arrows) so they read over the scene.
    for (int a = 0; a < 8; ++a) {
      for (int bit : {1, 2, 4}) {
        int const b = a | bit;
        if (b == a) {
          continue;
        }
        float3 const& p0 = corners[static_cast<size_t>(a)];
        float3 const& p1 = corners[static_cast<size_t>(b)];
        float const radius =
            (isSelected || triggered ? 0.0022f : 0.0014f) * length(cameraPos - (p0 + p1) * 0.5f);
        debugDraw->DrawSolidCylinder(p0, p1 - p0, radius, color, 8, true, false);
      }
    }
    debugText->Draw(
        corners[7], std::string(ICON_FA_CUBE " ") + stop.name.c_str(), color, ImVec2{0, -14});
  }
}

void TunerEditor::ShowStopBanner() {
  if (!_stopEvent || !_mochiScene.IsSimulating()) {
    return;
  }
  // Top-center, below the play toolbar.
  std::string const text = std::string(ICON_FA_HAND_PAPER " Stopped at t = ") +
      std::to_string(_stopEvent->time).substr(0, 5) + " s: " + _stopEvent->message;
  char const* const hint = "Play resumes (stop conditions stay off for this run); Stop resets.";
  ImVec2 const textSize = ImGui::CalcTextSize(text.c_str());
  ImVec2 const hintSize = ImGui::CalcTextSize(hint);
  float const width = std::max(textSize.x, hintSize.x);
  ImVec2 const pad{12.0f, 8.0f};
  ImVec2 const windowPos = ImGui::GetWindowPos();
  float const x = windowPos.x + 0.5f * (ImGui::GetWindowWidth() - width);
  float const y = windowPos.y + 64.0f;
  ImDrawList* drawList = ImGui::GetWindowDrawList();
  drawList->AddRectFilled(
      ImVec2(x - pad.x, y - pad.y),
      ImVec2(x + width + pad.x, y + textSize.y + hintSize.y + 4.0f + pad.y),
      ImGui::GetColorU32(ImVec4(0.45f, 0.08f, 0.06f, 0.92f)),
      8.0f);
  drawList->AddText(
      ImVec2(x, y), ImGui::GetColorU32(ImVec4(1.0f, 0.9f, 0.85f, 1.0f)), text.c_str());
  drawList->AddText(
      ImVec2(x, y + textSize.y + 4.0f), ImGui::GetColorU32(ImVec4(1.0f, 0.75f, 0.7f, 0.85f)), hint);
}

//--------------------------------------------------------------------------------------------------
// Windows
//--------------------------------------------------------------------------------------------------

void TunerEditor::ShowStopConditionCards(bool& structural) {
  tuner::TunerSpec& spec = _tunerAsset->GetSpec();
  auto& stops = spec.stopConditions;
  _stopCards.Resize(stops.size());

  // Channels a threshold can watch: the last run's, plus those the spec's loads and probes will
  // produce (so conditions can be set up before the first Play).
  std::vector<std::string> channels;
  auto const addChannel = [&](std::string name) {
    if (std::find(channels.begin(), channels.end(), name) == channels.end()) {
      channels.push_back(std::move(name));
    }
  };
  for (tuner::LoadSpec const& load : spec.loads) {
    std::string const prefix = "load." + std::string(std::string_view(load.name));
    addChannel(prefix + ".mag");
    if (load.type == tuner::LoadType::PosePath) {
      addChannel(prefix + ".error");
      addChannel(prefix + ".angleError");
    }
  }
  for (tuner::ProbeSpec const& probe : spec.probes) {
    std::string const target(std::string_view(probe.target));
    auto const add = [&](std::initializer_list<char const*> suffixes) {
      for (char const* suffix : suffixes) {
        addChannel(target + "." + suffix);
      }
    };
    switch (probe.kind) {
      case tuner::ProbeKind::BodyPose:
        add(
            {"pos.x",
             "pos.y",
             "pos.z",
             "rotvec.x",
             "rotvec.y",
             "rotvec.z",
             "rotvecUnwrapped.x",
             "rotvecUnwrapped.y",
             "rotvecUnwrapped.z"});
        break;
      case tuner::ProbeKind::BodyVelocity:
        add({"vel.x", "vel.y", "vel.z", "angvel.x", "angvel.y", "angvel.z"});
        break;
      case tuner::ProbeKind::ContactForce:
        add({"contactForce.x", "contactForce.y", "contactForce.z"});
        break;
      case tuner::ProbeKind::SolverStats:
        addChannel("solver.iters");
        addChannel("solver.residual");
        addChannel("solver.lineSearchIters");
        addChannel("solver.convergence");
        break;
      case tuner::ProbeKind::Performance:
        addChannel("perf.solveMs");
        break;
      case tuner::ProbeKind::Penetration:
        add({"penetration.volume"});
        break;
      case tuner::ProbeKind::JointState:
        break; // DoF count is only known once bound; the last run's channels cover these
    }
  }
  for (ChannelHistory const& channel : _channels) {
    addChannel(channel.name);
  }
  std::vector<std::string> const bodies = GetTargetCandidates(tuner::LoadType::Force);
  auto const bodyPosition = [&](std::string_view target) -> std::optional<Real3> {
    StagedActor const* staged = FindStagedBody(target);
    return staged ? std::optional(staged->worldTransform.GetTranslation()) : std::nullopt;
  };
  auto const centerBoxOn = [&](tuner::StopCondition& stop, Real3 const& center) {
    Real3 const half = (stop.boxMax - stop.boxMin) * real{0.5};
    stop.boxMin = center - half;
    stop.boxMax = center + half;
  };
  // A threshold on a probe channel the spec does not produce yet: the probe that would.
  auto const probeFor = [](std::string_view channel) -> std::optional<tuner::ProbeSpec> {
    size_t const dot =
        channel.find('.', channel.rfind('/') == std::string_view::npos ? 0 : channel.rfind('/'));
    if (dot == std::string_view::npos || channel.starts_with("load.")) {
      return std::nullopt;
    }
    std::string_view const quantity = channel.substr(dot + 1);
    tuner::ProbeSpec probe;
    probe.target = DynamicString(channel.substr(0, dot));
    if (quantity.starts_with("pos.") || quantity.starts_with("rotvec.") ||
        quantity.starts_with("rotvecUnwrapped.")) {
      probe.kind = tuner::ProbeKind::BodyPose;
    } else if (quantity.starts_with("vel.") || quantity.starts_with("angvel.")) {
      probe.kind = tuner::ProbeKind::BodyVelocity;
    } else if (quantity.starts_with("contactForce.")) {
      probe.kind = tuner::ProbeKind::ContactForce;
    } else if (quantity.starts_with("penetration.")) {
      probe.kind = tuner::ProbeKind::Penetration;
    } else {
      return std::nullopt;
    }
    return probe;
  };

  if (stops.empty()) {
    ImGui::TextDisabled("None: runs last the full duration.");
  }
  int const selectedBox = SelectedStopBoxIndex();
  int removeIndex = -1;
  ReorderableBubbleList list;
  list.count = stops.size();
  list.id = [&](size_t i) { return _stopCards.cards[i].id; };
  list.bubbleColor = [&](size_t i, bool isPlaceholder) {
    bool const triggered = _stopEvent && _stopEvent->condition == static_cast<int>(i);
    return CardColor(
        triggered ? kErrorColor : kWarningColor,
        !isPlaceholder && stops[i].enabled,
        !isPlaceholder && _stopCards.cards[i].id == _selectedStopCard);
  };
  list.ghostLabel = [&](size_t i) { return std::string(std::string_view(stops[i].name)); };
  list.move = [&](size_t from, size_t to) {
    if (from != to) {
      MoveItem(stops, from, to);
      MoveItem(_stopCards.cards, from, to);
      structural = true;
    }
  };
  list.drawContent = [&](size_t i) {
    tuner::StopCondition& stop = stops[i];
    float const top = ImGui::GetItemRectMin().y;
    bool const isBox = stop.kind != tuner::StopKind::ChannelThreshold;
    bool const enters = stop.kind == tuner::StopKind::EntersBox;
    bool const triggered = _stopEvent && _stopEvent->condition == static_cast<int>(i);
    std::string label = (triggered ? std::string(ICON_FA_HAND_PAPER " ") : std::string()) +
        std::string(std::string_view(stop.name)) + "  [";
    if (isBox) {
      label += (enters ? "enters box: " : "leaves box: ") +
          (stop.target.empty() ? std::string("no target")
                               : std::string(std::string_view(stop.target)));
    } else {
      char threshold[32];
      std::snprintf(threshold, sizeof(threshold), "%.4g", static_cast<double>(stop.threshold));
      label += (stop.channel.empty() ? std::string("no channel")
                                     : std::string(std::string_view(stop.channel))) +
          (stop.comparison == tuner::StopComparison::Above ? " > " : " < ") + threshold;
    }
    label += "]";
    bool const wasEnabled = stop.enabled;
    bool remove = false;
    bool const bodyOpen = DrawBubbleHeader(
        i,
        &stop.enabled,
        "When unchecked this condition is not checked.",
        label,
        _stopCards.cards[i].collapsed,
        "Remove this stop condition.",
        remove);
    structural |= stop.enabled != wasEnabled;
    if (remove) {
      removeIndex = static_cast<int>(i);
    }
    if (bodyOpen) {
      ImGui::BeginDisabled(!stop.enabled);
      structural |= ImGui::InputText("Name", &stop.name);
      Tooltip("Shown in the viewport and in summary.json when this condition stops a run.");
      constexpr std::array<NamedOption, 3> kStopKinds{{
          {"Leaves Box", "Stops once the target's origin leaves an axis-aligned box."},
          {"Enters Box", "Stops once the target's origin enters an axis-aligned box."},
          {"Channel Threshold",
           "Stops once a channel (e.g. block.pos.z) goes above or below a threshold."},
      }};
      structural |= DescribedCombo("Kind", stop.kind, kStopKinds);
      if (isBox) {
        structural |= TargetPicker("Target", stop.target, bodies);
        Tooltip(
            enters ? "Rigid body or articulated link whose origin must stay outside the box."
                   : "Rigid body or articulated link whose origin must stay inside the box.");
        structural |= ImGui::DragRealXYZ("Box Min", stop.boxMin, 0.005f, 0.0f, 0.0f, "%.3f m");
        structural |= ImGui::DragRealXYZ("Box Max", stop.boxMax, 0.005f, 0.0f, 0.0f, "%.3f m");
        Tooltip(
            "World-space corners of the box. Select the card to move and scale it with the "
            "viewport gizmo (W, R).");
        for (int d = 0; d < 3; ++d) {
          if (stop.boxMin[d] > stop.boxMax[d]) {
            std::swap(stop.boxMin[d], stop.boxMax[d]);
          }
        }
        std::optional<Real3> const position = bodyPosition(stop.target);
        ImGui::BeginDisabled(!position || _mochiScene.IsSimulating());
        if (ImGui::Button(ICON_FA_CROSSHAIRS " Center on Target")) {
          centerBoxOn(stop, *position);
          structural = true;
        }
        ImGui::EndDisabled();
        Tooltip("Move the box so it is centered on the target's current position.");
        if (static_cast<int>(i) == selectedBox && !_mochiScene.IsSimulating()) {
          ImGui::SameLine();
          ImGui::TextDisabled("Gizmo: W move, R scale");
        }
      } else {
        structural |= TargetPicker("Channel", stop.channel, channels);
        Tooltip("Channel to watch, as in Asset Tuner Plots and signals.csv.");
        constexpr std::array<NamedOption, 2> kComparisons{{
            {"Above", "Stops once the channel is greater than the threshold."},
            {"Below", "Stops once the channel is less than the threshold."},
        }};
        structural |= DescribedCombo("When", stop.comparison, kComparisons);
        // Step relative to the value, so tiny thresholds (e.g. overlap volumes in m^3) stay
        // editable.
        float const thresholdSpeed =
            stop.threshold != 0 ? 0.01f * std::abs(static_cast<float>(stop.threshold)) : 0.001f;
        structural |=
            ImGui::DragReal("Threshold", &stop.threshold, thresholdSpeed, 0.0f, 0.0f, "%.4g");
        Tooltip(
            "In the channel's units (m, m/s, N, m^3, ...). Position thresholds are drawn as a "
            "plane in the viewport. Ctrl+click to type a value.");
        bool const produced =
            std::find(channels.begin(), channels.end(), std::string_view(stop.channel)) !=
            channels.end();
        if (!stop.channel.empty() && !produced) {
          ImGui::TextColored(
              kWarningColor, ICON_FA_EXCLAMATION_TRIANGLE " No probe produces this channel.");
          if (std::optional<tuner::ProbeSpec> probe = probeFor(stop.channel)) {
            ImGui::SameLine();
            if (ImGui::SmallButton("Add Probe")) {
              spec.probes.push_back(std::move(*probe));
              _probeCards.AddExpanded();
              structural = true;
            }
            Tooltip("Add the probe that produces this channel.");
          }
        }
      }
      double const minStart = 0.0;
      structural |= ImGui::DragScalar(
          "Check From", ImGuiDataType_Double, &stop.start, 0.01f, &minStart, nullptr, "%.3f s");
      Tooltip("Not checked before this simulation time, e.g. to let the scene settle.");
      structural |= ImGui::DragScalar(
          "Hold For", ImGuiDataType_Double, &stop.holdTime, 0.01f, &minStart, nullptr, "%.3f s");
      Tooltip(
          "Only stop once the condition has held continuously this long, e.g. to ignore brief "
          "spikes on impact. 0: stop at once.");
      ImGui::EndDisabled();
    }
    float const bottom = ImGui::GetCursorScreenPos().y;
    ImVec2 const windowPos = ImGui::GetWindowPos();
    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) &&
        ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
        ImGui::IsMouseHoveringRect(
            ImVec2(windowPos.x, top),
            ImVec2(windowPos.x + ImGui::GetWindowWidth(), bottom),
            false)) {
      _selectedStopCard = _stopCards.cards[i].id;
    }
  };
  DrawReorderableBubbleList(_stopCards.drag, list);

  if (removeIndex >= 0) {
    stops.erase(stops.begin() + removeIndex);
    _stopCards.cards.erase(_stopCards.cards.begin() + removeIndex);
    structural = true;
  }
  if (ImGui::BeginCombo(
          "##addstop",
          ICON_FA_PLUS " Add Stop Condition",
          ImGuiComboFlags_NoArrowButton | ImGuiComboFlags_HeightLarge)) {
    std::string const firstBody = bodies.empty() ? std::string() : bodies.front();
    auto const add = [&](tuner::StopCondition stop) {
      stops.push_back(std::move(stop));
      _stopCards.AddExpanded();
      _selectedStopCard = _stopCards.cards.back().id;
      structural = true;
    };
    if (ImGui::Selectable("Leaves Box")) {
      tuner::StopCondition stop;
      stop.name = "escaped";
      stop.kind = tuner::StopKind::LeavesBox;
      stop.target = DynamicString(firstBody);
      stop.boxMin = {-0.25f, -0.25f, -0.25f};
      stop.boxMax = {0.25f, 0.25f, 0.25f};
      if (std::optional<Real3> const position = bodyPosition(firstBody)) {
        centerBoxOn(stop, *position);
      }
      add(std::move(stop));
    }
    Tooltip("Stop once a body leaves a box (0.5 m, centered on it; adjust in the viewport).");
    if (ImGui::Selectable("Enters Box")) {
      tuner::StopCondition stop;
      stop.name = "reached";
      stop.kind = tuner::StopKind::EntersBox;
      stop.target = DynamicString(firstBody);
      stop.boxMin = {-0.25f, -0.25f, -0.25f};
      stop.boxMax = {0.25f, 0.25f, 0.25f};
      // Beside the body rather than around it, so it does not trigger on the first step.
      if (std::optional<Real3> const position = bodyPosition(firstBody)) {
        centerBoxOn(stop, *position + Real3{0.6f, 0.0f, 0.0f});
      }
      add(std::move(stop));
    }
    Tooltip(
        "Stop once a body enters a box (0.5 m, placed beside it along +X; move it in the "
        "viewport).");
    if (ImGui::Selectable("Falls Below Height")) {
      // A threshold on the body's height, 5 cm under the ground, with the probe it needs.
      tuner::StopCondition stop;
      stop.name = "fell";
      stop.kind = tuner::StopKind::ChannelThreshold;
      stop.channel = DynamicString(firstBody + ".pos.z");
      stop.comparison = tuner::StopComparison::Below;
      stop.threshold = EffectiveGroundHeight() - 0.05f;
      if (std::optional<tuner::ProbeSpec> probe = probeFor(stop.channel); probe &&
          std::find(channels.begin(), channels.end(), firstBody + ".pos.z") == channels.end()) {
        spec.probes.push_back(std::move(*probe));
        _probeCards.AddExpanded();
      }
      add(std::move(stop));
    }
    Tooltip(
        "Stop once a body's height drops 5 cm below the ground (adds its Body Pose probe if "
        "needed).");
    if (ImGui::Selectable("Interpenetration")) {
      // A threshold on the body's overlap volume, held briefly so impacts do not trigger it.
      std::string const channel = firstBody + ".penetration.volume";
      tuner::StopCondition stop;
      stop.name = "interpenetrated";
      stop.kind = tuner::StopKind::ChannelThreshold;
      stop.channel = DynamicString(channel);
      stop.comparison = tuner::StopComparison::Above;
      stop.threshold = 1e-5f;
      stop.holdTime = 0.05;
      if (std::optional<tuner::ProbeSpec> probe = probeFor(stop.channel);
          probe && std::find(channels.begin(), channels.end(), channel) == channels.end()) {
        spec.probes.push_back(std::move(*probe));
        _probeCards.AddExpanded();
      }
      add(std::move(stop));
    }
    Tooltip(
        "Stop once a body overlaps other actors by more than 10 cm^3 for 50 ms (adds its "
        "Penetration probe if needed). Normal contact already overlaps a little, and how much "
        "scales with body size and load (a resting 10 cm, 1 kg cube reads ~3 cm^3), so plot the "
        "channel first and set the threshold above it.");
    if (ImGui::Selectable("Channel Threshold")) {
      tuner::StopCondition stop;
      stop.name = "threshold";
      stop.kind = tuner::StopKind::ChannelThreshold;
      stop.channel = channels.empty() ? DynamicString() : DynamicString(channels.front());
      add(std::move(stop));
    }
    Tooltip("Stop once any channel goes above or below a value.");
    ImGui::EndCombo();
  }
  Tooltip(
      "Stop conditions end a run early: the CLI stops (status \"stopped\" in summary.json), "
      "Studio pauses and shows why.");
  if (_mochiScene.IsSimulating()) {
    ImGui::TextDisabled("Stop condition changes apply on the next run.");
  }
}

void TunerEditor::ShowProbeCards(bool& structural) {
  auto& probes = _tunerAsset->GetSpec().probes;
  _probeCards.Resize(probes.size());
  auto const candidatesFor = [&](tuner::ProbeKind kind) {
    if (kind != tuner::ProbeKind::JointState) {
      return GetTargetCandidates(tuner::LoadType::Force);
    }
    std::vector<std::string> joints;
    for (auto const& joint : _targets.joints) {
      if (joint.numDofs > 0) {
        joints.push_back(joint.name);
      }
    }
    return joints;
  };

  int removeIndex = -1;
  ReorderableBubbleList list;
  list.count = probes.size();
  list.id = [&](size_t i) { return _probeCards.cards[i].id; };
  list.bubbleColor = [&](size_t, bool isPlaceholder) {
    return CardColor(kNeutralAccent, !isPlaceholder);
  };
  list.ghostLabel = [&](size_t i) {
    return std::string(kProbeKinds[static_cast<size_t>(probes[i].kind)].name);
  };
  list.move = [&](size_t from, size_t to) {
    if (from != to) {
      MoveItem(probes, from, to);
      MoveItem(_probeCards.cards, from, to);
      structural = true;
    }
  };
  list.drawContent = [&](size_t i) {
    tuner::ProbeSpec& probe = probes[i];
    bool const hasTarget = ProbeHasTarget(probe.kind);
    std::string label = kProbeKinds[static_cast<size_t>(probe.kind)].name;
    if (hasTarget) {
      label += "  [" +
          (probe.target.empty() ? std::string("no target")
                                : std::string(std::string_view(probe.target))) +
          "]";
    }
    bool remove = false;
    bool const bodyOpen = DrawBubbleHeader(
        i, nullptr, nullptr, label, _probeCards.cards[i].collapsed, "Remove this probe.", remove);
    if (remove) {
      removeIndex = static_cast<int>(i);
    }
    if (bodyOpen) {
      structural |= DescribedCombo("Kind", probe.kind, kProbeKinds);
      if (hasTarget) {
        structural |= TargetPicker("Target", probe.target, candidatesFor(probe.kind));
        Tooltip(
            probe.kind == tuner::ProbeKind::JointState
                ? "Joint to sample, as <articulated actor>/<joint>."
                : "Rigid body or articulated link to sample.");
      }
      if (probe.kind == tuner::ProbeKind::Penetration) {
        structural |= ImGui::Checkbox("Ignore Ground", &probe.ignoreGround);
        Tooltip("Leave out overlap with ground planes (static, unbounded actors).");
      }
      ImGui::TextDisabled("Channels: %s", ProbeChannelHint(probe.kind).c_str());
    }
  };
  DrawReorderableBubbleList(_probeCards.drag, list);

  if (removeIndex >= 0) {
    probes.erase(probes.begin() + removeIndex);
    _probeCards.cards.erase(_probeCards.cards.begin() + removeIndex);
    structural = true;
  }
  if (ImGui::BeginCombo(
          "##addprobe",
          ICON_FA_PLUS " Add Probe",
          ImGuiComboFlags_NoArrowButton | ImGuiComboFlags_HeightLarge)) {
    for (size_t k = 0; k < kProbeKinds.size(); ++k) {
      if (ImGui::Selectable(kProbeKinds[k].name)) {
        tuner::ProbeSpec probe;
        probe.kind = static_cast<tuner::ProbeKind>(k);
        if (ProbeHasTarget(probe.kind)) {
          auto const candidates = candidatesFor(probe.kind);
          if (!candidates.empty()) {
            probe.target = candidates.front();
          }
        }
        probes.push_back(std::move(probe));
        _probeCards.AddExpanded();
        structural = true;
      }
      Tooltip(kProbeKinds[k].description);
    }
    ImGui::EndCombo();
  }
  Tooltip("Add a probe. Its channels appear in Asset Tuner Plots and in the recorded signals.csv.");
  if (_mochiScene.IsSimulating()) {
    ImGui::TextDisabled("Probe changes apply on the next run.");
  }
}

void TunerEditor::ShowRunWindow(bool* open) {
  if (!ImGui::Begin("Asset Tuner Run", open)) {
    ImGui::End();
    return;
  }
  tuner::TunerSpec& spec = _tunerAsset->GetSpec();
  bool changed = false;
  bool structural = false;

  ImGui::SeparatorText("Prefab");
  {
    mochi::DynamicString prefabPath(_tunerAsset->GetPrefabPath().ToString());
    ImGui::InputText("Prefab", &prefabPath, ImGuiInputTextFlags_ReadOnly);
    Tooltip("The prefab this asset tuner simulates. Solver settings are from the prefab");
    bool const prefabLoaded = _tunerAsset->GetPrefabAsset() != nullptr;
    ImGui::BeginDisabled(!prefabLoaded);
    if (ImGui::Button(ICON_FA_FOLDER_OPEN " Open Prefab")) {
      _studio->OpenPath(_tunerAsset->GetPrefabPath());
    }
    ImGui::EndDisabled();
    Tooltip("Open the prefab in its own editor tab.");
    if (!prefabLoaded) {
      ImGui::SameLine();
      ImGui::TextColored(kErrorColor, ICON_FA_EXCLAMATION_TRIANGLE " Failed to load (see log).");
    }
  }

  ImGui::SeparatorText("Recording");
  bool const recording = IsRecording();
  bool const armed = recording || _recordNextRun;
  ImGui::PushStyleColor(
      ImGuiCol_Button,
      armed ? ImVec4(0.70f, 0.12f, 0.12f, 1.0f) : ImGui::GetStyleColorVec4(ImGuiCol_Button));
  char const* const recordLabel = recording ? ICON_FA_STOP " Stop Recording###record"
      : armed                               ? ICON_FA_CIRCLE " Armed###record"
                                            : ICON_FA_CIRCLE " Record###record";
  if (ImGui::Button(recordLabel)) {
    if (armed) {
      // Stops the current recording (if any) and disarms.
      StopRecording();
      _recordNextRun = false;
    } else {
      _recordNextRun = true;
      StartRecording(); // no-op until Play when not simulating
    }
  }
  ImGui::PopStyleColor();
  Tooltip(
      recording ? "Recording. Click to stop, write the run folder, and disarm (later runs are not "
                  "recorded)."
          : armed
          ? "Armed: every run from Play is recorded. Click to disarm."
          : "Record the current run (or, when stopped, every run from Play) to a tuner_run folder"
            "specified below.\nClick again to disarm.");
  ImGui::SameLine();
  if (recording) {
    ImGui::TextColored(kErrorColor, "Recording");
  } else if (armed) {
    ImGui::TextUnformatted("Recording each run");
  } else {
    ImGui::TextDisabled("Not recording");
  }
  ImGui::Checkbox("Stop at Duration", &_stopAtDuration);
  Tooltip("Pauses the simulation (and end the recording) when reaching the run duration.");

  mochi::DynamicString outputDir(_outputDir);
  if (ImGui::InputText("Output Folder", &outputDir)) {
    _outputDir = std::string(std::string_view(outputDir));
  }
  Tooltip("Each recording creates a timestamped run folder in here.");
  if (ImGui::Button("Browse##output")) {
    mochi::Path const chosen =
        SuperDexStudio::GetFolderDialogPath("Asset Tuner Output Folder", mochi::Path{_outputDir});
    if (!chosen.IsEmpty()) {
      _outputDir = chosen.ToString();
    }
  }
  Tooltip("Pick the folder recordings are written to.");
  if (!_lastRunDir.empty()) {
    mochi::DynamicString lastRun(_lastRunDir);
    ImGui::InputText("Last Run", &lastRun, ImGuiInputTextFlags_ReadOnly);
    Tooltip("Folder of the most recent recording.");
    if (ImGui::Button(ICON_FA_COPY " Copy Path")) {
      ImGui::SetClipboardText(_lastRunDir.c_str());
    }
    Tooltip("Copy the run folder path, e.g. to pass to analysis scripts.");
  }

  ImGui::SeparatorText("Run Settings");
  ImGui::BeginDisabled(_mochiScene.IsSimulating());
  std::optional<real> const groundHeightBefore = spec.run.groundHeight;
  sAutoGroundHeightForRunUi = _autoGroundHeight;
  changed |= ImGui::SimpleReflectionStruct(spec.run, DrawRunFieldOverride);
  changed |= spec.run.groundHeight != groundHeightBefore;
  ImGui::EndDisabled();
  if (_mochiScene.IsSimulating()) {
    ImGui::TextDisabled("Stop the simulation to change run settings.");
  }

  // Each card list numbers its cards from 1, so each gets its own ID scope.
  ImGui::SeparatorText("Probes");
  ImGui::PushID("probes");
  ShowProbeCards(structural);
  ImGui::PopID();

  ImGui::SeparatorText("Stop Conditions");
  ImGui::PushID("stopConditions");
  ShowStopConditionCards(structural);
  ImGui::PopID();

  if (changed || structural) {
    OnSpecEdited(structural);
  }
  ImGui::End();
}

void TunerEditor::ShowOverridesWindow(bool* open) {
  if (!ImGui::Begin("Asset Tuner Prefab Overrides", open)) {
    ImGui::End();
    return;
  }
  auto& overrides = _tunerAsset->GetSpec().prefabOverrides;
  _overrideCards.Resize(overrides.size());
  if (_overrideErrors.size() != overrides.size()) {
    ValidateOverrides();
  }
  ImGui::TextWrapped(
      "Values set on the prefab before each Play, without changing the prefab file. The viewport "
      "shows the unmodified prefab.");
  ImGui::Separator();
  bool changed = false;
  bool structural = false;

  int removeIndex = -1;
  ReorderableBubbleList list;
  list.count = overrides.size();
  list.id = [&](size_t i) { return _overrideCards.cards[i].id; };
  list.bubbleColor = [&](size_t i, bool isPlaceholder) {
    bool const broken = i < _overrideErrors.size() && !_overrideErrors[i].empty();
    return CardColor(broken ? kErrorColor : kNeutralAccent, !isPlaceholder);
  };
  list.ghostLabel = [&](size_t i) { return std::string(std::string_view(overrides[i].path)); };
  list.move = [&](size_t from, size_t to) {
    if (from != to) {
      MoveItem(overrides, from, to);
      MoveItem(_overrideCards.cards, from, to);
      MoveItem(_overrideErrors, from, to);
      structural = true;
    }
  };
  list.drawContent = [&](size_t i) {
    tuner::PrefabOverride& entry = overrides[i];
    std::string const& error = _overrideErrors[i];
    std::string label = (error.empty() ? "" : ICON_FA_EXCLAMATION_TRIANGLE " ") +
        (entry.path.empty() ? std::string("(no path)")
                            : std::string(std::string_view(entry.path))) +
        " = " + std::string(std::string_view(entry.value));
    bool remove = false;
    bool const bodyOpen = DrawBubbleHeader(
        i,
        nullptr,
        nullptr,
        label,
        _overrideCards.cards[i].collapsed,
        "Remove this override.",
        remove);
    if (remove) {
      removeIndex = static_cast<int>(i);
    }
    if (bodyOpen) {
      changed |= ImGui::InputText("Path", &entry.path);
      Tooltip(
          "Dot-separated JSON path into the prefab, with [index] or [name=...] selectors.\n\n"
          "Examples:\n"
          "  actors.rigid[name=block].contact.coulombFrictionCoefficient\n"
          "  prefabs[name=Nested1].actors.rigid[name=Rigid1].isStatic");
      changed |= ImGui::InputText("Value", &entry.value);
      Tooltip(
          "JSON value, e.g. 0.8, true or [0, 0, 1]. Text that is not valid JSON is passed as a string.");
      if (!error.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, kErrorColor);
        ImGui::TextWrapped("%s", error.c_str());
        ImGui::PopStyleColor();
      } else {
        ImGui::TextDisabled("Applies on the next Play.");
      }
    }
  };
  DrawReorderableBubbleList(_overrideCards.drag, list);

  if (removeIndex >= 0) {
    overrides.erase(overrides.begin() + removeIndex);
    _overrideCards.cards.erase(_overrideCards.cards.begin() + removeIndex);
    structural = true;
  }
  if (ImGui::Button(ICON_FA_PLUS " Add Override", ImVec2(-FLT_MIN, 0))) {
    overrides.emplace_back();
    _overrideCards.AddExpanded();
    structural = true;
  }
  Tooltip("Add a prefab override.");
  if (changed || structural) {
    ValidateOverrides();
    OnSpecEdited(structural);
  }
  ImGui::End();
}

} // namespace superdex::studio
