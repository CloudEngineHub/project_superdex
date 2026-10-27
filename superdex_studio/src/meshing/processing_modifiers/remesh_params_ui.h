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

// Shared meshing parameter widgets used by the Wrap Mesh (alpha wrap), Remesh (ACVD / surface
// Delaunay / incremental isotropic), and Tetrahedralize methods. They operate on primitive refs so
// each method can own its own reflected props struct (the neutral mochi::mesh parameter structs are
// not reflected).

#include "meshing/processing_modifiers/processing_method.h" // ModifierTooltip

namespace superdex::studio::processing {

// Length input labeled @p name: percent of the bbox average dimension when @p relativeToMeshSize,
// else mm. @p description starts its tooltip.
void ShowLengthControl(
    char const* name,
    char const* description,
    double& length,
    bool relativeToMeshSize,
    ModifierTooltip const& tooltip);

// Edge-size length control plus the "Relative to Mesh Size" toggle.
void ShowEdgeSizeControl(
    double& edgeSize,
    bool& relativeToMeshSize,
    ModifierTooltip const& tooltip);

// Sharp-feature detection controls (the angle is disabled unless @p detectFeatures).
void ShowFeatureControls(
    bool& detectFeatures,
    double& sharpFeatureAngle,
    ModifierTooltip const& tooltip);

} // namespace superdex::studio::processing
