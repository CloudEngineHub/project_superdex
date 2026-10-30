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

#include <mochi_core/utils/error.h>

#include <arrow/array.h>

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace superdex::recorder::detail {

/* @brief One Rerun chunk: rows of one entity, with a column per timeline and per component. */
struct RrdChunk {
  /* Without Rerun's leading `/`, as the recorder writes it. */
  std::string entityPath;
  /* Sequence numbers or nanoseconds, by timeline name. Static chunks have no timelines. */
  std::map<std::string, std::shared_ptr<arrow::Int64Array>> timelines;
  /* By component name: each row's list of values, or null where the row does not update it. */
  std::vector<std::pair<std::string, std::shared_ptr<arrow::ListArray>>> components;
};

/* @brief Decodes every chunk of the one recording in an RRD, in file order.
 * Accepts only whole files written by the Rerun release that the recorder links. */
std::vector<RrdChunk> ReadRrdChunks(std::filesystem::path const& path, mochi::Error& error);

} // namespace superdex::recorder::detail
