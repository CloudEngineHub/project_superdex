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

#include <mochi_core/test/mochi_test_helpers.h>
#include <superdex_robotics/utils/bot_utils.h>

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
