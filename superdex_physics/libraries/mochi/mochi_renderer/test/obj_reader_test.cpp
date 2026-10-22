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

// Unit tests for the OBJ reader's material/section support. Fixtures are written
// to temp files and parsed via the public ReadObjFromFile API.

#include <gtest/gtest.h>

#include <mochi_renderer/utils.h>

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/log.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace mochi_renderer;

namespace {

// Two triangles, each bound to a distinct material via usemtl, plus an .mtl
// library defining the colors. Red has a specular exponent (Ns); Green does not.
char const* const kTwoMaterialMtl = R"MTL(
newmtl Red
Kd 1 0 0
Ns 50
newmtl Green
Kd 0 1 0
)MTL";

char const* const kTwoMaterialObj = R"OBJ(
mtllib obj_reader_test.mtl
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
usemtl Red
f 1 2 3
usemtl Green
f 1 2 4
)OBJ";

// Single triangle, no material binding.
char const* const kNoMaterialObj = R"OBJ(
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
)OBJ";

} // namespace

TEST(ObjReaderTest, TwoMaterialSectionsResolved) {
  // The OBJ and its .mtl sidecar must share a directory, and the .mtl name must
  // match the `mtllib` directive in kTwoMaterialObj.
  mochi::TempDirCleanup const dir =
      mochi::CreateTempDirectory("obj_reader_two_material", mochi::test::ExpectOK{});
  std::filesystem::path const objPath = dir.Path() / "model.obj";
  std::ofstream(dir.Path() / "obj_reader_test.mtl") << kTwoMaterialMtl;
  std::ofstream(objPath) << kTwoMaterialObj;

  std::vector<MeshSection> const sections = ReadObjFromFile(objPath.string().c_str());
  ASSERT_EQ(sections.size(), 2u);

  // First-seen material order: Red, then Green.
  MeshSection const& red = sections[0];
  EXPECT_FALSE(red.hasNormals);
  EXPECT_EQ(red.positions.size(), 9u);
  std::vector<int> const expectedIndices = {0, 1, 2};
  EXPECT_EQ(red.indices, expectedIndices);
  EXPECT_NEAR(red.baseColor[0], 1.0f, 1e-6f);
  EXPECT_NEAR(red.baseColor[1], 0.0f, 1e-6f);
  EXPECT_NEAR(red.baseColor[2], 0.0f, 1e-6f);
  EXPECT_NEAR(red.metallic, 0.0f, 1e-6f);
  // Ns 50 → roughness = sqrt(2/52).
  EXPECT_NEAR(red.roughness, std::sqrt(2.0f / 52.0f), 1e-5f);

  MeshSection const& green = sections[1];
  EXPECT_NEAR(green.baseColor[0], 0.0f, 1e-6f);
  EXPECT_NEAR(green.baseColor[1], 1.0f, 1e-6f);
  EXPECT_NEAR(green.baseColor[2], 0.0f, 1e-6f);
  // No Ns → tiny_obj_loader's default shininess of 1 → roughness = sqrt(2/3).
  EXPECT_NEAR(green.roughness, std::sqrt(2.0f / 3.0f), 1e-5f);
}

TEST(ObjReaderTest, NoMaterialYieldsSingleDefaultSection) {
  mochi::TempFileCleanup const obj =
      mochi::CreateTempFile("obj_reader_no_material", ".obj", mochi::test::ExpectOK{});
  std::ofstream(obj.Path()) << kNoMaterialObj;

  std::vector<MeshSection> const sections = ReadObjFromFile(obj.Path().string().c_str());
  ASSERT_EQ(sections.size(), 1u);

  // Default MeshSection material when no usemtl/mtllib is present.
  MeshSection const& s = sections[0];
  EXPECT_EQ(s.positions.size(), 9u);
  EXPECT_NEAR(s.baseColor[0], 0.5f, 1e-6f);
  EXPECT_NEAR(s.baseColor[1], 0.5f, 1e-6f);
  EXPECT_NEAR(s.baseColor[2], 0.5f, 1e-6f);
  EXPECT_NEAR(s.roughness, 0.5f, 1e-6f);
}

TEST(ObjReaderTest, MissingFileReturnsEmpty) {
  // The reader warns on a failed open; silence it so the test harness does not
  // treat the expected warning as a failure.
  auto const prevLogFn = mochi::GetLogCallback();
  mochi::SetLogCallback(nullptr);
  MOCHI_DEFER(mochi::SetLogCallback(prevLogFn));

  EXPECT_TRUE(ReadObjFromFile("/nonexistent/path/missing.obj").empty());
}

// Texture coordinates come through per corner, flipped to glTF's top-left origin, and the diffuse
// map becomes the base color texture.
TEST(ObjReaderTest, TexcoordsAndDiffuseMap) {
  mochi::TempDirCleanup const dir =
      mochi::CreateTempDirectory("obj_reader_texture", mochi::test::ExpectOK{});
  std::vector<char> const png = {
      static_cast<char>(0x89), 'P', 'N', 'G', '\r', '\n', static_cast<char>(0x1A), '\n', 7};
  std::ofstream(dir.Path() / "albedo.png", std::ios::binary).write(png.data(), png.size());
  std::ofstream(dir.Path() / "textured.mtl") << "newmtl Skin\nKd 1 1 1\nmap_Kd albedo.png\n";
  std::filesystem::path const objPath = dir.Path() / "model.obj";
  std::ofstream(objPath) << R"OBJ(
mtllib textured.mtl
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 0.25
usemtl Skin
f 1/1 2/2 3/3
)OBJ";

  std::vector<MeshSection> const sections = ReadObjFromFile(objPath.string().c_str());
  ASSERT_EQ(sections.size(), 1u);
  MeshSection const& s = sections[0];
  EXPECT_TRUE(s.hasTexcoords);
  std::vector<float> const expectedTexcoords = {0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.75f};
  EXPECT_EQ(s.texcoords, expectedTexcoords);
  ASSERT_NE(s.baseColorTexture, nullptr);
  EXPECT_EQ(s.baseColorTexture->mimeType, "image/png");
  EXPECT_EQ(s.baseColorTexture->bytes.size(), png.size());
}

TEST(ObjReaderTest, PartialTexcoordsAreDropped) {
  // One face of the material has texture coordinates and one does not, so the section has none.
  mochi::TempDirCleanup const dir =
      mochi::CreateTempDirectory("obj_reader_partial_texcoords", mochi::test::ExpectOK{});
  std::filesystem::path const objPath = dir.Path() / "model.obj";
  std::ofstream(objPath) << "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nvt 0 0\nvt 1 0\nvt 0 1\n"
                            "f 1/1 2/2 3/3\nf 1 2 4\n";

  std::vector<MeshSection> const sections = ReadObjFromFile(objPath.string().c_str());
  ASSERT_EQ(sections.size(), 1u);
  EXPECT_FALSE(sections[0].hasTexcoords);
  EXPECT_TRUE(sections[0].texcoords.empty());
  EXPECT_FALSE(BuildGlbFromMeshSections(sections).empty());
}

TEST(ObjReaderTest, UnsupportedDiffuseMapFallsBackToColor) {
  mochi::TempDirCleanup const dir =
      mochi::CreateTempDirectory("obj_reader_unsupported_texture", mochi::test::ExpectOK{});
  std::ofstream(dir.Path() / "albedo.bmp", std::ios::binary) << "BM not a png or jpeg";
  std::ofstream(dir.Path() / "textured.mtl") << "newmtl Skin\nKd 0 1 0\nmap_Kd albedo.bmp\n";
  std::filesystem::path const objPath = dir.Path() / "model.obj";
  std::ofstream(objPath) << "mtllib textured.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\n"
                            "usemtl Skin\nf 1/1 2/1 3/1\n";
  auto const prevLogFn = mochi::GetLogCallback();
  mochi::SetLogCallback(nullptr);
  MOCHI_DEFER(mochi::SetLogCallback(prevLogFn));

  std::vector<MeshSection> const sections = ReadObjFromFile(objPath.string().c_str());
  ASSERT_EQ(sections.size(), 1u);
  EXPECT_EQ(sections[0].baseColorTexture, nullptr);
  std::array<float, 4> const green = {0.0f, 1.0f, 0.0f, 1.0f};
  EXPECT_EQ(sections[0].baseColor, green);
}

TEST(ObjReaderTest, MissingDiffuseMapFallsBackToColor) {
  mochi::TempDirCleanup const dir =
      mochi::CreateTempDirectory("obj_reader_missing_texture", mochi::test::ExpectOK{});
  std::ofstream(dir.Path() / "textured.mtl") << "newmtl Skin\nKd 0 0 1\nmap_Kd missing.png\n";
  std::filesystem::path const objPath = dir.Path() / "model.obj";
  std::ofstream(objPath) << "mtllib textured.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\n"
                            "usemtl Skin\nf 1/1 2/1 3/1\n";
  // The missing map is reported on the warning channel, which the test harness otherwise treats
  // as a failure.
  auto const prevLogFn = mochi::GetLogCallback();
  mochi::SetLogCallback(nullptr);
  MOCHI_DEFER(mochi::SetLogCallback(prevLogFn));

  std::vector<MeshSection> const sections = ReadObjFromFile(objPath.string().c_str());
  ASSERT_EQ(sections.size(), 1u);
  EXPECT_EQ(sections[0].baseColorTexture, nullptr);
  std::array<float, 4> const blue = {0.0f, 0.0f, 1.0f, 1.0f};
  EXPECT_EQ(sections[0].baseColor, blue);
}
