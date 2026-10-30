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

#include <mochi_core/geometry/model_data.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <superdex_robotics/core/loader.h>
#include <superdex_robotics/utils/bot_utils.h>

#include <optional>

using namespace mochi;
using namespace superdex::robotics;

namespace {

DynamicArray<DynamicString> const kLinkNames = {"palm", "finger", "tip"};

BotPrefab MakeSkinnedBot() {
  BotPrefab bot;
  bot.skin.emplace();
  return bot;
}

SkinningData
MakeSkinning(int weightsPerNode, DynamicArray<int> indices, DynamicArray<real> weights) {
  SkinningData skinning;
  skinning.weightsPerNode = weightsPerNode;
  skinning.indices = std::move(indices);
  skinning.weights = std::move(weights);
  return skinning;
}

/* Serves one collision model (or a load failure when there is none) for any path, recording the
 * paths it was asked for. */
struct FakeModelLoader : FileBotLoader {
  std::optional<ModelData> model;
  mutable DynamicArray<std::string> requestedPaths;

  ModelData LoadModelData(std::string_view path, superdex::Error& error) const override {
    requestedPaths.emplace_back(path);
    MOCHI_ERROR_IF(!model.has_value(), error, "no such model");
    return model.value_or(ModelData{});
  }
};

ModelData MakeSkinnedModel(SkinningData skinning) {
  ModelData model;
  model.mesh.emplace().skinning = std::move(skinning);
  return model;
}

} // namespace

TEST(SkinCoverageTest, BoneLinkIndices_EmptyWithoutSkin) {
  EXPECT_TRUE(ComputeSkinBoneLinkIndices(BotPrefab{}, MakeConstSpan(kLinkNames)).empty());
}

// A standalone skinned bot records no bone->link names: skin bone b is link b.
TEST(SkinCoverageTest, BoneLinkIndices_IdentityWithoutBoneLinkNames) {
  DynamicArray<int> const expected = {0, 1, 2};
  EXPECT_EQ(ComputeSkinBoneLinkIndices(MakeSkinnedBot(), MakeConstSpan(kLinkNames)), expected);
}

// A composed bot names each bone's link; bones resolve by name, and an unknown name maps to none.
TEST(SkinCoverageTest, BoneLinkIndices_ResolvesBoneLinkNames) {
  BotPrefab bot = MakeSkinnedBot();
  bot._skinBoneLinks = {"tip", "palm", "removed"};
  DynamicArray<int> const expected = {2, 0, kIndexNone};
  EXPECT_EQ(ComputeSkinBoneLinkIndices(bot, MakeConstSpan(kLinkNames)), expected);
}

TEST(SkinCoverageTest, CoveredLinks_NoneWithoutSkin) {
  SkinningData const skinning = MakeSkinning(1, {0, 1, 2}, {1_r, 1_r, 1_r});
  DynamicArray<bool> const expected = {false, false, false};
  EXPECT_EQ(ComputeSkinCoveredLinks(BotPrefab{}, MakeConstSpan(kLinkNames), &skinning), expected);
}

// nonCollidingLinks names the covered links exactly, regardless of what the skin weights.
TEST(SkinCoverageTest, CoveredLinks_NonCollidingLinksTakePrecedence) {
  BotPrefab bot = MakeSkinnedBot();
  bot.skin->nonCollidingLinks = DynamicArray<DynamicString>{"finger", "unknown"};
  SkinningData const skinning = MakeSkinning(1, {0, 1, 2}, {1_r, 1_r, 1_r});
  DynamicArray<bool> const expected = {false, true, false};
  EXPECT_EQ(ComputeSkinCoveredLinks(bot, MakeConstSpan(kLinkNames), &skinning), expected);
}

// An empty but set nonCollidingLinks covers nothing; the skinning is not consulted.
TEST(SkinCoverageTest, CoveredLinks_EmptyNonCollidingLinksCoverNothing) {
  BotPrefab bot = MakeSkinnedBot();
  bot.skin->nonCollidingLinks = DynamicArray<DynamicString>{};
  SkinningData const skinning = MakeSkinning(1, {0, 1, 2}, {1_r, 1_r, 1_r});
  DynamicArray<bool> const expected = {false, false, false};
  EXPECT_EQ(ComputeSkinCoveredLinks(bot, MakeConstSpan(kLinkNames), &skinning), expected);
}

// Without nonCollidingLinks the skin covers the links its collision mesh weights, through the
// bone->link mapping. Zero-weight padding influences, unmapped bones and bones out of the mapping's
// range cover nothing.
TEST(SkinCoverageTest, CoveredLinks_FromWeightedBones) {
  BotPrefab bot = MakeSkinnedBot();
  // Bone 0 -> tip, bone 1 -> finger, bone 2 -> palm, bone 3 -> unmapped.
  bot._skinBoneLinks = {"tip", "finger", "palm", "removed"};
  // Two influences per vertex. Bone 2 (the palm) only ever has zero weight; bones -1 and 4 are out
  // of range.
  SkinningData const skinning =
      MakeSkinning(2, {0, 2, 1, 0, 3, 2, 4, -1}, {1_r, 0_r, 0.5_r, 0.5_r, 1_r, 0_r, 0.5_r, 0.5_r});
  DynamicArray<bool> const expected = {false, true, true};
  EXPECT_EQ(ComputeSkinCoveredLinks(bot, MakeConstSpan(kLinkNames), &skinning), expected);
}

TEST(SkinCoverageTest, CoveredLinks_NoneWithoutSkinning) {
  DynamicArray<bool> const expected = {false, false, false};
  EXPECT_EQ(
      ComputeSkinCoveredLinks(MakeSkinnedBot(), MakeConstSpan(kLinkNames), nullptr), expected);
}

// With nonCollidingLinks set, the skin's collision mesh is not read.
TEST(SkinCoverageTest, MapSkinOntoLinks_NonCollidingLinksNeedNoCollisionMesh) {
  BotPrefab bot = MakeSkinnedBot();
  bot.skin->shapeFile = "skin.mochi.h5";
  bot.skin->nonCollidingLinks = DynamicArray<DynamicString>{"tip"};
  FakeModelLoader const loader;
  Error error;
  SkinLinkMapping const mapping = MapSkinOntoLinks(bot, MakeConstSpan(kLinkNames), loader, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  DynamicArray<int> const expectedBones = {0, 1, 2};
  DynamicArray<bool> const expectedCovered = {false, false, true};
  EXPECT_EQ(mapping.boneLinks, expectedBones);
  EXPECT_EQ(mapping.coveredLinks, expectedCovered);
  EXPECT_TRUE(loader.requestedPaths.empty());
}

// Without nonCollidingLinks, coverage comes from the collision mesh the loader serves.
TEST(SkinCoverageTest, MapSkinOntoLinks_CoverageFromCollisionMeshThroughLoader) {
  BotPrefab bot = MakeSkinnedBot();
  bot.skin->shapeFile = "skin.mochi.h5";
  FakeModelLoader loader;
  loader.model = MakeSkinnedModel(MakeSkinning(1, {1, 2}, {1_r, 1_r}));
  Error error;
  SkinLinkMapping const mapping = MapSkinOntoLinks(bot, MakeConstSpan(kLinkNames), loader, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  DynamicArray<bool> const expectedCovered = {false, true, true};
  DynamicArray<std::string> const expectedPaths = {"skin.mochi.h5"};
  EXPECT_EQ(mapping.coveredLinks, expectedCovered);
  EXPECT_EQ(loader.requestedPaths, expectedPaths);
}

// Coverage that depends on a collision mesh that cannot be read, or that has no skinning, fails
// rather than covering nothing.
TEST(SkinCoverageTest, MapSkinOntoLinks_FailsWithoutReadableSkinning) {
  BotPrefab bot = MakeSkinnedBot();
  bot.skin->shapeFile = "skin.mochi.h5";

  FakeModelLoader const missing;
  Error missingError;
  EXPECT_TRUE(
      MapSkinOntoLinks(bot, MakeConstSpan(kLinkNames), missing, missingError).boneLinks.empty());
  EXPECT_FALSE(missingError.IsOK());

  FakeModelLoader unskinned;
  unskinned.model.emplace().mesh.emplace();
  Error unskinnedError;
  EXPECT_TRUE(MapSkinOntoLinks(bot, MakeConstSpan(kLinkNames), unskinned, unskinnedError)
                  .boneLinks.empty());
  EXPECT_FALSE(unskinnedError.IsOK());
}

TEST(SkinCoverageTest, MapSkinOntoLinks_EmptyWithoutSkin) {
  FakeModelLoader const loader;
  Error error;
  SkinLinkMapping const mapping =
      MapSkinOntoLinks(BotPrefab{}, MakeConstSpan(kLinkNames), loader, error);
  EXPECT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_TRUE(mapping.boneLinks.empty());
  EXPECT_TRUE(mapping.coveredLinks.empty());
}
