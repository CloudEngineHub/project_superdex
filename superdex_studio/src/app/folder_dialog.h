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

#include <optional>
#include <string>

namespace superdex::studio {

// Shows the platform's modern folder picker (on Windows, the Explorer-style dialog with Quick
// Access/favorites in its navigation pane). Returns the chosen folder as UTF-8, an empty string if
// the user cancelled, or nullopt if no native picker is available so the caller should fall back.
std::optional<std::string> ShowNativeFolderDialog(
    char const* title,
    std::string const& defaultPath);

} // namespace superdex::studio
