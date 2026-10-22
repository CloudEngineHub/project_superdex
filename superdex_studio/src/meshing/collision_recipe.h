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

// The processing pipeline ("recipe") that builds a link's collision model from its geoms, as the
// URDF and MJCF importers write it, so the model can be reopened, tuned and rebuilt in the Model
// Editor.

#include "meshing/processing_modifiers/processing_serialization.h"
#include "meshing/processing_modifiers/processing_stack.h"
#include "meshing/processing_modifiers/source_parts.h"

#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/utils/error.h>

#include <filesystem>
#include <string>
#include <vector>

namespace superdex::studio {

// How a link's collision geoms are merged into one mesh.
enum class CollisionMerge { Union, ShrinkWrap };

// What a recipe's source reads.
enum class CollisionRecipeSource {
  // The parts' files: a lone mesh "From File" by the source for its type, else Source from Parts.
  Files,
  // The model's CAD Model in the Model Editor (e.g. `cad/<model>.stl`), holding the lone mesh part,
  // read "From Model Viewer" and placed by the editor's CAD transform.
  CadModel,
  // The model's Render Model in the Model Editor (e.g. the importer's `render/<model>.glb` of the
  // same geoms), already in the model's frame, read "From Model Viewer".
  RenderModel,
};

struct CollisionRecipeOptions {
  CollisionMerge merge = CollisionMerge::Union;
  // Rebuild the merged surface into a clean, watertight mesh (Alpha Wrap, then Isotropic Remesh).
  bool remesh = true;
  bool bakeSdf = true;
  CollisionRecipeSource source = CollisionRecipeSource::Files;
};

// A recipe: the modifiers and the editor state (the CAD transform) saved with them.
struct CollisionRecipe {
  ModifierStack modifiers;
  ProcessingEditorState editorState;
};

// The part a Source from Parts reads for @p geom, posed in its link's frame. A mesh geom names
// @p meshFile (e.g. a copy of the geom's file); a primitive keeps its type and size.
SourcePart PartFromGeom(superdex::robotics::GeomReference const& geom, std::string const& meshFile);

// The pipeline exporting @p parts to @p exportPath: a source (see CollisionRecipeSource), Merge
// when there are several parts or they are shrink-wrapped, the optional remesh, and Export Mochi
// Model. A lone mesh read from its file is placed by an Apply SRT Transformation.
CollisionRecipe MakeCollisionRecipe(
    std::vector<SourcePart> parts,
    CollisionRecipeOptions const& options,
    std::string const& exportPath);

// Writes @p recipe to @p pipelinePath (a .StudioProcessing.json), with its paths stored relative to
// it, or as they are unless @p relativePaths.
bool WriteCollisionRecipe(
    CollisionRecipe const& recipe,
    std::filesystem::path const& pipelinePath,
    bool relativePaths,
    mochi::Error& error);

} // namespace superdex::studio
