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

#include <imgui.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace superdex::studio {

// Extra vertical frame padding for header rows, so the toggle icons sit inset in the bar.
float HeaderExtraPadY();

// Frame padding for header-row widgets: taller than the default by HeaderExtraPadY(). Pair with
// ImGui::PopStyleVar().
void PushHeaderRowFramePadding();

// Persistent drag state

// Persistent drag state for one drag-to-reorder "bubble" list (see DrawReorderableBubbleList).
// While a drag is active `draggingId` is the stable id of the grabbed row (0 = not dragging) and
// `dropSlot` is the highlighted insertion gap (the final array index on release).
struct DragReorderState {
  uint64_t draggingId = 0;
  int dropSlot = -1;
};

// Configuration for DrawReorderableBubbleList
struct ReorderableBubbleList {
  std::size_t count = 0;
  // Stable, nonzero id for row i (tracks the drag across reorders/frames).
  std::function<uint64_t(std::size_t)> id;
  // Background color for row i. isPlaceholder is true for the thin stand-in drawn in the dragged
  // row's original slot (use a muted variant).
  std::function<ImU32(std::size_t index, bool isPlaceholder)> bubbleColor;
  // Short label drawn on the cursor-attached drag ghost.
  std::function<std::string(std::size_t)> ghostLabel;
  // Draws row i's body. The drag grip is drawn first; the body should start with ImGui::SameLine()
  // to continue on the grip's row.
  std::function<void(std::size_t)> drawContent;
  // Optional: constrain the raw destination gap for a drag (e.g. to pin a row). Null = no clamp.
  std::function<int(int slot, int draggingIndex)> clampSlot;
  // Commits a reorder from index `from` to index `to` (called once on release).
  std::function<void(std::size_t from, std::size_t to)> move;
  // Width of the drag grip button. 0: fits its "=" label.
  float gripWidth = 0.0f;
};

// Draws `list` as a vertical stack of rounded "bubbles" spanning the available width, each with a
// left-edge drag grip that reorders the rows: a thin placeholder marks the grabbed row's slot, a
// blue drop line shows where it will land (60% hysteresis so it doesn't jitter at separators; snaps
// back when the cursor leaves the section), and a cursor-attached ghost carries the row's label. On
// release, list.move(from, to) commits the reorder. State (the active drag + cached slot) persists
// in @p state across frames. Each bubble pads its widgets on all sides, so rows need no margins of
// their own.
void DrawReorderableBubbleList(DragReorderState& state, ReorderableBubbleList const& list);

// Moves array[from] to index `to`, shifting the elements between them by one: the reorder that
// ReorderableBubbleList::move commits, for each array kept in the list's order.
template <typename Array>
void MoveElement(Array& array, std::size_t from, std::size_t to) {
  auto const begin = array.begin();
  auto const f = static_cast<std::ptrdiff_t>(from);
  auto const t = static_cast<std::ptrdiff_t>(to);
  if (f < t) {
    std::rotate(begin + f, begin + f + 1, begin + t + 1);
  } else if (t < f) {
    std::rotate(begin + t, begin + f, begin + f + 1);
  }
}

// Header row for a bubble (call first in ReorderableBubbleList::drawContent):
//   [index] [enable checkbox] [collapsible header] ... [trash]
// `enabled` may be null for rows without an enable toggle. Returns true if the body is open.
// Sets `outRemove` when the trash button is pressed.
bool DrawBubbleHeader(
    std::size_t index,
    bool* enabled,
    char const* enabledTooltip,
    std::string const& label,
    bool& collapsed,
    char const* removeTooltip,
    bool& outRemove);

} // namespace superdex::studio
