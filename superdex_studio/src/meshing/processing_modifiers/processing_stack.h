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

// Operations on a whole modifier stack -- the ordered modifiers of one model's processing pipeline.
// The Model Editor and headless tools share them, so both build a stack the same way.

#include "meshing/processing_modifiers/processing_modifier.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace superdex::studio {

using ModifierStack = std::vector<std::unique_ptr<MeshProcessingModifier>>;

// The source-first list of enabled modifier indices feeding @p index (with @p index last), or empty
// when no source is reachable upstream. Stops at the nearest source -- one modifier's Generate
// rebuilds its own segment, not the segments feeding it through files.
[[nodiscard]] std::vector<std::size_t> BuildGenerationChain(
    ModifierStack const& stack,
    std::size_t index);

// The whole enabled stack in order, from its first enabled source -- what Build/Export All builds,
// spanning every source->...->export segment. Empty when no enabled source exists.
[[nodiscard]] std::vector<std::size_t> BuildFullGenerationChain(ModifierStack const& stack);

// The modifier index whose output an edge-swap at @p index references (its explicit upstream
// target, else the nearest preceding source), or -1 if none.
[[nodiscard]] int ReferenceModifierIndex(ModifierStack const& stack, std::size_t index);

// The file the stack's source (first) modifier reads, which every export modifier derives its
// default output path from. Empty when the stack is empty or its source has no file yet.
[[nodiscard]] std::string StackSourceFilePath(
    ModifierStack const& stack,
    ModifierRunContext const& ctx);

// Re-derives every export modifier's path from @p sourceFilePath while its Auto toggle is on (see
// MeshProcessingMethod::RefreshAutoExportPath), so an export writes its file on Build/Export All
// without the user opening Browse, and an Auto path always matches what Browse would offer. Cheap
// enough to call every frame, and safe to: an Auto path is never serialized, so re-deriving it
// cannot dirty the saved snapshot however much it moves.
void RefreshAutoExportPaths(ModifierStack& stack, std::string const& sourceFilePath);

// Per-modifier flag marking export modifiers that share an output file with another enabled export
// modifier. Two exports aimed at the same path silently overwrite each other; the Auto path is
// deliberately not disambiguated (it has to keep the model's own name to stay associated with it),
// so the fix is to turn Auto off on one of them. Compares resolved paths, so hand-picked duplicates
// are caught as well.
[[nodiscard]] std::vector<bool> FindCollidingExportPaths(ModifierStack const& stack);

} // namespace superdex::studio
