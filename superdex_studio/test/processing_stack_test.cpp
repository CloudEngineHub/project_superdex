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

// Covers the operations on a whole modifier stack
// (meshing/processing_modifiers/processing_stack.h). The stacks are built from a test method, so no
// mesh operation or file is involved.

#include "meshing/processing_modifiers/processing_stack.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace superdex::studio;

namespace {

// A method that passes its input through and reports whatever reference and export path it was
// given. Its Auto export path is the source path plus ".out".
class TestMethod : public MeshProcessingMethod {
 public:
  TestMethod(int referenceIndex, std::string exportPath, bool autoExportPath)
      : _referenceIndex(referenceIndex),
        _exportPath(std::move(exportPath)),
        _autoExportPath(autoExportPath) {}

  [[nodiscard]] char const* Name() const override {
    return "Test";
  }
  void ShowParams(ModifierGuiContext const& /*gui*/) override {}
  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& /*error*/) const override {
    return input;
  }
  void SerializeProps(picojson::value& /*out*/) const override {}
  void DeserializeProps(picojson::value const& /*in*/) override {}
  [[nodiscard]] std::string PropsSignature(ModifierRunContext const& /*ctx*/) const override {
    return {};
  }
  [[nodiscard]] int ReferenceIndex() const override {
    return _referenceIndex;
  }
  [[nodiscard]] std::string SourceFilePath(ModifierRunContext const& ctx) const override {
    return ctx.cadFilePath;
  }
  [[nodiscard]] bool ProvidesFileExport() const override {
    return _autoExportPath || !_exportPath.empty();
  }
  [[nodiscard]] std::string ExportPath() const override {
    return _exportPath;
  }
  void RefreshAutoExportPath(std::string const& sourceFilePath) override {
    if (_autoExportPath) {
      _exportPath = sourceFilePath + ".out";
    }
  }

 private:
  int _referenceIndex;
  std::string _exportPath;
  bool _autoExportPath;
};

struct ModifierSpec {
  ModifierKind kind = ModifierKind::Transform;
  bool enabled = true;
  int referenceIndex = MeshProcessingMethod::kReferencePrecedingSource;
  std::string exportPath;
  bool autoExportPath = false;
};

ModifierStack MakeStack(std::vector<ModifierSpec> const& specs) {
  ModifierStack stack;
  for (ModifierSpec const& spec : specs) {
    std::vector<std::unique_ptr<MeshProcessingMethod>> methods;
    methods.push_back(
        std::make_unique<TestMethod>(spec.referenceIndex, spec.exportPath, spec.autoExportPath));
    auto modifier = std::make_unique<MeshProcessingModifier>("Test", spec.kind, std::move(methods));
    modifier->enabled = spec.enabled;
    stack.push_back(std::move(modifier));
  }
  return stack;
}

constexpr ModifierKind kSource = ModifierKind::Source;
constexpr ModifierKind kTransform = ModifierKind::Transform;
constexpr ModifierKind kExport = ModifierKind::Export;

} // namespace

TEST(ProcessingStackTest, GenerationChainClimbsEnabledModifiersToTheNearestSource) {
  ModifierStack const stack = MakeStack(
      {{.kind = kSource},
       {.kind = kSource},
       {.kind = kTransform, .enabled = false},
       {.kind = kTransform},
       {.kind = kExport}});

  std::vector<std::size_t> const expected{1, 3, 4};
  EXPECT_EQ(BuildGenerationChain(stack, 4), expected);
}

TEST(ProcessingStackTest, GenerationChainIsEmptyWithoutAnUpstreamSource) {
  ModifierStack const stack =
      MakeStack({{.kind = kSource, .enabled = false}, {.kind = kTransform}, {.kind = kExport}});

  EXPECT_TRUE(BuildGenerationChain(stack, 2).empty());
  EXPECT_TRUE(BuildGenerationChain(stack, 7).empty());
}

TEST(ProcessingStackTest, FullGenerationChainSpansEverySegmentFromTheFirstSource) {
  ModifierStack const stack = MakeStack(
      {{.kind = kTransform},
       {.kind = kSource},
       {.kind = kTransform, .enabled = false},
       {.kind = kExport},
       {.kind = kSource},
       {.kind = kExport}});

  std::vector<std::size_t> const expected{1, 3, 4, 5};
  EXPECT_EQ(BuildFullGenerationChain(stack), expected);
}

TEST(ProcessingStackTest, ReferenceIsAnUpstreamModifierOrTheNearestPrecedingSource) {
  ModifierStack const stack = MakeStack(
      {{.kind = kSource},
       {.kind = kTransform},
       {.kind = kSource},
       {.kind = kTransform, .referenceIndex = 1},
       {.kind = kTransform},
       {.kind = kTransform, .referenceIndex = 5}});

  EXPECT_EQ(ReferenceModifierIndex(stack, 3), 1); // explicit, upstream
  EXPECT_EQ(ReferenceModifierIndex(stack, 4), 2); // nearest preceding source
  EXPECT_EQ(ReferenceModifierIndex(stack, 5), -1); // explicit, but not upstream
  EXPECT_EQ(ReferenceModifierIndex(stack, 0), -1); // nothing precedes the first modifier
}

TEST(ProcessingStackTest, SourceFilePathComesFromTheFirstModifier) {
  ModifierRunContext ctx;
  ctx.cadFilePath = "cad/part.step";

  EXPECT_EQ(StackSourceFilePath(MakeStack({}), ctx), "");
  EXPECT_EQ(
      StackSourceFilePath(MakeStack({{.kind = kSource}, {.kind = kExport}}), ctx), ctx.cadFilePath);
}

TEST(ProcessingStackTest, RefreshAutoExportPathsUpdatesOnlyAutoExports) {
  ModifierStack stack = MakeStack(
      {{.kind = kSource},
       {.kind = kExport, .autoExportPath = true},
       {.kind = kExport, .exportPath = "chosen.glb"}});

  RefreshAutoExportPaths(stack, "cad/part.step");

  EXPECT_EQ(stack[1]->ExportPath(), "cad/part.step.out");
  EXPECT_EQ(stack[2]->ExportPath(), "chosen.glb");
}

TEST(ProcessingStackTest, CollidingExportPathsCompareNormalizedPathsOfEnabledExports) {
  ModifierStack const stack = MakeStack(
      {{.kind = kSource},
       {.kind = kExport, .exportPath = "render/part.glb"},
       {.kind = kExport, .exportPath = "render/../render/part.glb"},
       {.kind = kExport, .enabled = false, .exportPath = "render/part.glb"},
       {.kind = kExport, .exportPath = "collision/part.mochi.h5"}});

  std::vector<bool> const expected{false, true, true, false, false};
  EXPECT_EQ(FindCollidingExportPaths(stack), expected);
}
