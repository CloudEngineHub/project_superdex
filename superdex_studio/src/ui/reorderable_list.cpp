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

#include "ui/reorderable_list.h"

#include "ui/imgui_widgets.h"

#include <imgui_internal.h> // ImGuiWindow / WorkRect, to bound the header width
#include <imguios/fonts/icons_font_awesome5.h>

#include <algorithm>
#include <vector>

namespace superdex::studio {

float HeaderExtraPadY() {
  return IM_TRUNC(ImGui::GetFontSize() * 0.2f);
}

void PushHeaderRowFramePadding() {
  ImVec2 const padding = ImGui::GetStyle().FramePadding;
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(padding.x, padding.y + HeaderExtraPadY()));
}

void DrawReorderableBubbleList(DragReorderState& state, ReorderableBubbleList const& list) {
  constexpr float kBubbleRounding = 6.0f;
  // Space between a bubble's edge and its widgets.
  constexpr float kBubblePadX = 4.0f;
  constexpr float kBubblePadY = 4.0f;
  constexpr float kBubbleGap = 8.0f;

  ImDrawList* const drawList = ImGui::GetWindowDrawList();

  // Resolve the active drag (tracked by stable id) to an array index; cancel it if it vanished.
  int draggingIndex = -1;
  if (state.draggingId != 0) {
    for (std::size_t i = 0; i < list.count; ++i) {
      if (list.id(i) == state.draggingId) {
        draggingIndex = static_cast<int>(i);
        break;
      }
    }
    if (draggingIndex < 0) {
      state.draggingId = 0;
      state.dropSlot = -1;
    }
  }
  bool const dragging = state.draggingId != 0;

  float const contentX0 = ImGui::GetCursorScreenPos().x;
  float const contentWidth = ImGui::GetContentRegionAvail().x;
  float const sectionTop = ImGui::GetCursorScreenPos().y;

  // Rendered row bounds in array order (the dragged row records its thin placeholder's bounds).
  std::vector<float> topY;
  std::vector<float> bottomY;
  int startDragIndex = -1; // a grip that begins a drag this frame

  ImGuiWindow* const window = ImGui::GetCurrentWindow();
  // Ends a row at `bottom`, kBubbleGap above whatever follows (the gap item adds no item spacing).
  auto const endRow = [&](float bottom) {
    ImGui::SetCursorScreenPos(ImVec2(contentX0, bottom));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ImGui::GetStyle().ItemSpacing.x, 0.0f));
    ImGui::Dummy(ImVec2(0.0f, kBubbleGap));
    ImGui::PopStyleVar();
  };

  for (std::size_t i = 0; i < list.count; ++i) {
    float const rowTop = ImGui::GetCursorScreenPos().y;
    if (dragging && static_cast<int>(i) == draggingIndex) {
      // The grabbed row is replaced by a thin placeholder bubble marking its original slot.
      float const rowBottom = rowTop + ImGui::GetFrameHeight() * 0.35f;
      drawList->AddRectFilled(
          ImVec2(contentX0, rowTop),
          ImVec2(contentX0 + contentWidth, rowBottom),
          list.bubbleColor(i, /*isPlaceholder=*/true),
          kBubbleRounding);
      topY.push_back(rowTop);
      bottomY.push_back(rowBottom);
      endRow(rowBottom);
      continue;
    }

    ImGui::PushID(static_cast<int>(list.id(i)));
    // Widgets on the front draw-list channel; the bubble background is filled behind them after.
    drawList->ChannelsSplit(2);
    drawList->ChannelsSetCurrent(1);
    // Inset the row's widgets by the bubble padding: the group starts kBubblePadX in (lines inside
    // it wrap back to that x), and the right edges that size widgets and framed headers move in by
    // the same amount until the row ends.
    ImGui::SetCursorScreenPos(ImVec2(contentX0 + kBubblePadX, rowTop + kBubblePadY));
    float const savedWorkRectMaxX = window->WorkRect.Max.x;
    float const savedContentRegionMaxX = window->ContentRegionRect.Max.x;
    window->WorkRect.Max.x -= kBubblePadX;
    window->ContentRegionRect.Max.x -= kBubblePadX;
    ImGui::BeginGroup();
    // As tall as the row's header bar (see CollapsingHeaderWithToggles).
    PushHeaderRowFramePadding();
    ImGui::Button("=", ImVec2(list.gripWidth, 0.0f)); // drag grip
    ImGui::PopStyleVar();
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("Drag to reorder.");
    }
    if (!dragging && ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
      startDragIndex = static_cast<int>(i);
    }
    list.drawContent(i);
    ImGui::EndGroup();
    window->WorkRect.Max.x = savedWorkRectMaxX;
    window->ContentRegionRect.Max.x = savedContentRegionMaxX;

    float const rowBottom = ImGui::GetItemRectMax().y + kBubblePadY;
    drawList->ChannelsSetCurrent(0);
    drawList->AddRectFilled(
        ImVec2(contentX0, rowTop),
        ImVec2(contentX0 + contentWidth, rowBottom),
        list.bubbleColor(i, /*isPlaceholder=*/false),
        kBubbleRounding);
    drawList->ChannelsMerge();
    topY.push_back(rowTop);
    bottomY.push_back(rowBottom);

    ImGui::PopID();
    endRow(rowBottom);
  }

  float const sectionBottom = ImGui::GetCursorScreenPos().y;

  // --- drag: highlighted drop line(s) + ghost, with hysteresis on the slot switch ---
  if (dragging && !topY.empty()) {
    std::size_t const rowCount = topY.size(); // includes the dragged row's placeholder
    // Gap g sits before row g (gap rowCount == below the last row).
    std::vector<float> gapY(rowCount + 1);
    gapY[0] = topY[0];
    for (std::size_t g = 1; g < rowCount; ++g) {
      gapY[g] = 0.5f * (bottomY[g - 1] + topY[g]);
    }
    gapY[rowCount] = bottomY[rowCount - 1];

    int const maxSlot = static_cast<int>(rowCount);
    int slot = state.dropSlot;
    if (slot < 0 || slot > maxSlot) {
      slot = std::clamp(draggingIndex, 0, maxSlot);
    }

    ImVec2 const mouse = ImGui::GetMousePos();
    bool const offSection = mouse.y < sectionTop || mouse.y > sectionBottom ||
        mouse.x < contentX0 || mouse.x > contentX0 + contentWidth;
    if (offSection) {
      slot = draggingIndex; // dropping off the section snaps back to the original slot
    } else {
      // Only switch to a neighbouring gap once the cursor is more than 60% of the way to it, so the
      // highlight does not jitter around a separator.
      constexpr float kSwitchFraction = 0.6f;
      while (slot < maxSlot &&
             mouse.y > gapY[slot] + kSwitchFraction * (gapY[slot + 1] - gapY[slot])) {
        ++slot;
      }
      while (slot > 0 && mouse.y < gapY[slot] - kSwitchFraction * (gapY[slot] - gapY[slot - 1])) {
        --slot;
      }
    }
    if (list.clampSlot) {
      slot = list.clampSlot(slot, draggingIndex);
    }
    state.dropSlot = slot;

    // The placeholder occupies draggingIndex, so a gap g maps to a destination array index of g
    // when above the placeholder and g-1 when below it; the two gaps flanking it both mean "no
    // move".
    int const destIndex = (slot <= draggingIndex) ? slot : slot - 1;
    bool const atOriginal = destIndex == draggingIndex;

    ImU32 const dropColor = ImGui::GetColorU32(ImVec4(0.30f, 0.75f, 1.0f, 1.0f));
    auto const drawDropLine = [&](float y) {
      drawList->AddLine(ImVec2(contentX0, y), ImVec2(contentX0 + contentWidth, y), dropColor, 3.0f);
    };
    if (atOriginal) {
      drawDropLine(gapY[draggingIndex]);
      drawDropLine(gapY[draggingIndex + 1]);
    } else {
      drawDropLine(gapY[slot]);
    }

    // Ghost of the grabbed row's label, attached to the cursor.
    std::string const label = list.ghostLabel(static_cast<std::size_t>(draggingIndex));
    ImVec2 const textSize = ImGui::CalcTextSize(label.c_str());
    ImDrawList* const foreground = ImGui::GetForegroundDrawList(ImGui::GetWindowViewport());
    ImVec2 const p0(mouse.x + 14.0f, mouse.y + 6.0f);
    ImVec2 const p1(p0.x + textSize.x + 12.0f, p0.y + textSize.y + 8.0f);
    foreground->AddRectFilled(p0, p1, ImGui::GetColorU32(ImVec4(0.25f, 0.25f, 0.28f, 0.9f)), 4.0f);
    foreground->AddText(
        ImVec2(p0.x + 6.0f, p0.y + 4.0f),
        ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 1.0f, 1.0f)),
        label.c_str());

    // Release: commit the move to the destination (a no-op when it is the original position).
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
      list.move(static_cast<std::size_t>(draggingIndex), static_cast<std::size_t>(destIndex));
      state.draggingId = 0;
      state.dropSlot = -1;
    }
  } else if (dragging && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
    // Nothing visible to drop against (should not happen); just end the drag.
    state.draggingId = 0;
    state.dropSlot = -1;
  }

  // Begin a drag that was initiated on a grip this frame (engages next frame).
  if (startDragIndex >= 0) {
    state.draggingId = list.id(static_cast<std::size_t>(startDragIndex));
    state.dropSlot = startDragIndex;
  }
}

bool DrawBubbleHeader(
    std::size_t index,
    bool* enabled,
    char const* enabledTooltip,
    std::string const& label,
    bool& collapsed,
    char const* removeTooltip,
    bool& outRemove) {
  // Array index, right-justified in a fixed 2-digit-wide column so indices stay aligned once they
  // reach double digits (and the following widgets stay at a constant x).
  ImGui::SameLine();
  float const indexColStart = ImGui::GetCursorPosX();
  float const indexColWidth = ImGui::CalcTextSize("00").x;
  std::string const indexText = std::to_string(index);
  float const indexTextWidth = ImGui::CalcTextSize(indexText.c_str()).x;
  ImGui::SetCursorPosX(indexColStart + std::max(0.0f, indexColWidth - indexTextWidth));
  ImGui::TextUnformatted(indexText.c_str());

  bool const isEnabled = enabled == nullptr || *enabled;
  if (enabled != nullptr) {
    ImGui::SameLine();
    ImGui::SetCursorScreenPos(
        ImVec2(ImGui::GetCursorScreenPos().x, ImGui::GetCursorScreenPos().y + HeaderExtraPadY()));
    ImGui::Checkbox("##enabled", enabled);
    if (enabledTooltip != nullptr) {
      ImGui::ItemTooltipWrapped(enabledTooltip);
    }
  }

  ImGui::SameLine();
  // Grey the collapse header's label when disabled, but leave it clickable (visual only).
  if (!isEnabled) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
  }
  // Reserve room on the right for the remove button so the framed header stops before it. A framed
  // CollapsingHeader always spans the full work rect, so temporarily shrink the work rect's right
  // edge around it. The width matches the trash button's natural width (glyph + frame padding).
  float const removeWidth =
      ImGui::CalcTextSize(ICON_FA_TRASH).x + ImGui::GetStyle().FramePadding.x * 2.0f;
  float const removeGap = ImGui::GetStyle().ItemSpacing.x;
  ImGuiWindow* const window = ImGui::GetCurrentWindow();
  float const savedWorkRectMaxX = window->WorkRect.Max.x;
  window->WorkRect.Max.x -= removeWidth + removeGap;
  ImGui::SetNextItemOpen(!collapsed);
  std::string const header = label + "###header";
  PushHeaderRowFramePadding();
  bool const bodyOpen = ImGui::CollapsingHeader(header.c_str());
  ImGui::PopStyleVar();
  window->WorkRect.Max.x = savedWorkRectMaxX;
  if (!isEnabled) {
    ImGui::PopStyleColor();
  }
  collapsed = !bodyOpen;

  ImGui::SameLine();
  ImGui::SetCursorPosX(ImGui::GetContentRegionMax().x - removeWidth);
  PushHeaderRowFramePadding();
  if (ImGui::Button(ICON_FA_TRASH)) {
    outRemove = true;
  }
  ImGui::PopStyleVar();
  if (removeTooltip != nullptr) {
    ImGui::ItemTooltipWrapped(removeTooltip);
  }
  return bodyOpen;
}

} // namespace superdex::studio
