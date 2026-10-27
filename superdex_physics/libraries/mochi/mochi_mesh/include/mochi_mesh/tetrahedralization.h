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

#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/utils/error.h>
#include <mochi_mesh/mochi_mesh_cli_types.h>

namespace mochi::mesh {

/// @brief [Experimental] Generate a tetrahedral volume mesh from a closed triangle surface.
/// @warning This API may change in future releases.
///
/// @details Every output node is referenced by at least one tetrahedron, so the result can be used
/// directly as a soft actor shape. The output is deterministic for a given input and parameters.
///
/// @param[in] surfaceMesh Input triangle mesh (nodesPerElement must be 3). Must be closed and free
/// of self-intersections.
/// @param[in] params Tetrahedralization parameters
/// @param[in,out] error Error status
/// @return Tetrahedral mesh (nodesPerElement == 4)
[[nodiscard]] MeshData Tetrahedralize(
    MeshDataView const& surfaceMesh,
    TetrahedralizationParams const& params,
    Error& error);

} // namespace mochi::mesh
