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

#include <mochi_core/geometry/model_data.h>
#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_renderer/path.h>
#include <mochi_renderer/utils.h>
#include <superdex_robotics/utils/file_utils.h>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

// Builds the render and collision models of links for the URDF and MJCF importers. Geoms come from
// superdex::robotics::UrdfImportInfo or MjcfImportInfo, posed in their link's frame, in Mochi
// space.

namespace superdex::studio {

using GeomList = std::vector<superdex::robotics::GeomReference>;

// How a link's collision geoms are merged into one mesh.
enum class CollisionMerge {
  Union, // the boolean union of the geoms
  ShrinkWrap, // a shrink-wrap around the geoms
};

// Render mesh sections of one geom, in the link frame (Mochi space). Mesh geoms are read from their
// OBJ, STL, DAE, or GLB file; primitives are tessellated. A geom with a color of its own takes it
// (and its 2D texture, when the mesh has texture coordinates), unless it keeps the materials of a
// mesh that has some (see GeomReference::keepMeshMaterials); otherwise a mesh keeps its own
// colors. Appends a note to `warnings` for anything that could not be honored.
[[nodiscard]] std::vector<mochi_renderer::MeshSection> GeomRenderSections(
    superdex::robotics::GeomReference const& geom,
    std::vector<std::string>& warnings);

// Writes one .glb holding every geom in `geoms`, in the link frame, converted to the renderer's
// space by `toRender`. Returns false if no geom produced geometry or the file could not be written.
bool BuildLinkGlb(
    GeomList const& geoms,
    mochi::Path const& outputPath,
    mochi::CoordinateSpaceConverter const& toRender,
    std::vector<std::string>& warnings);

// A geom's collision surface as a closed triangle mesh, with the mesh scale applied. When
// `inLinkFrame` is true it is also posed in the link frame; otherwise it stays in the mesh's own
// frame, for @ref superdex::robotics::ExpandMjcfGeomLinks to place. Mesh geoms are loaded from
// their file; primitives are tessellated. Empty if the geom has no usable geometry.
[[nodiscard]] mochi::MeshData GeomCollisionMesh(
    superdex::robotics::GeomReference const& geom,
    bool inLinkFrame,
    std::vector<std::string>& warnings);

// MuJoCo reads only the first object of an OBJ file: the faces before an `o` or `g` line that
// follows faces. If `objPath` holds more objects, writes a copy keeping only the first object's
// faces to `outDir`, with its own copy of each material library (texture paths pointing at the
// originals), and returns the copy's path; otherwise returns `objPath`.
[[nodiscard]] std::string FirstObjObjectFile(
    std::string const& objPath,
    std::filesystem::path const& outDir);

// Sizes and poses a primitive geom that names a mesh as MuJoCo does (see
// @ref superdex::robotics::GeomReference::fitScale), then detaches the mesh. Returns false,
// with a warning, and leaves the geom unchanged if the mesh cannot be loaded or encloses no volume.
bool FitPrimitiveToMesh(
    superdex::robotics::GeomReference& geom,
    std::vector<std::string>& warnings);

struct GeomsInertia {
  mochi::real mass = 0; // [kg]
  mochi::Real3 centerOfMass{}; // [m], in the link frame
  mochi::Real6 momentOfInertia{}; // [kg m^2], about the center of mass: xx, xy, xz, yy, yz, zz
};

// A body's inertia inferred from its geoms, as MuJoCo does for a body without an <inertial>: every
// geom (visual or not) weighs its `mass`, or else its `density` (MuJoCo's default 1000 kg/m^3)
// times its enclosed volume. Empty if no geom has mass.
[[nodiscard]] std::optional<GeomsInertia> ComputeGeomsInertia(
    GeomList const& geoms,
    std::vector<std::string>& warnings);

struct LinkCollisionOptions {
  CollisionMerge merge = CollisionMerge::Union;
  bool remesh = true;
  bool bakeSdf = true;
};

// Writes one .mochi.h5 holding `geoms` merged per `options.merge`, in the link frame: their
// shrink-wrap for ShrinkWrap, otherwise their boolean union (a lone geom as it is). Spheres are
// tessellated too: Mochi needs a surface mesh on every dynamic link that has contact. Returns false
// if no geom produced geometry or the merge or save failed.
bool BuildLinkCollisionH5(
    GeomList const& geoms,
    LinkCollisionOptions const& options,
    mochi::Path const& outputPath,
    std::vector<std::string>& warnings);

// Writes one geom's collision surface to a .mochi.h5 in the mesh's own frame (see
// @ref GeomCollisionMesh), for the Separate layout.
bool BuildGeomCollisionH5(
    superdex::robotics::GeomReference const& geom,
    LinkCollisionOptions const& options,
    mochi::Path const& outputPath,
    std::vector<std::string>& warnings);

} // namespace superdex::studio
