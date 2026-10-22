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

#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/file_utils.h>

#include <string_view>

// Internal MJCF import interface for superdex_robotics. The implementation is a hand-rolled
// tinyxml2 parser (mjcf_import_utils.cpp) and is a private implementation detail -- the public
// entry points live in superdex_robotics/utils/file_utils.h (LoadBotPrefabFromMjcf*,
// ExpandMjcfGeomLinks, ApplyMjcfContactFilter).

namespace superdex::robotics {
using namespace mochi;

/* @brief Parse an MJCF file into one link per body, with no geoms assigned. */
void LoadBotPrefabFromMjcfFile(
    BotPrefab& outData,
    std::string_view path,
    MjcfImportInfo& outInfo,
    Error& error);

/* @brief Parse an MJCF XML string. Includes and asset files cannot be resolved. */
void LoadBotPrefabFromMjcfXml(
    BotPrefab& outData,
    std::string_view xmlString,
    MjcfImportInfo& outInfo,
    Error& error);

} // namespace superdex::robotics
