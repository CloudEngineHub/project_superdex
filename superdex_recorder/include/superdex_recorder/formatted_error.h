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
#include <mochi_core/utils/log.h>

#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>

namespace superdex::recorder::detail {

/* @brief Sets an error whose description is built at run time, such as one that names an actor.
 * mochi::Error keeps only a pointer to its description, so each distinct description is kept for
 * the rest of the process. */
inline void
SetFormattedError(mochi::Error& error, std::string description, char const* file, int line) {
  if (!error.IsOK()) {
    return;
  }
  static std::mutex mutex;
  static std::unordered_set<std::string> descriptions;
  std::lock_guard const lock{mutex};
  error.SetFirstError(descriptions.insert(std::move(description)).first->c_str(), file, line);
}

} // namespace superdex::recorder::detail

/* Like MOCHI_ERROR_SET, with a printf-style description. */
#define SUPERDEX_ERROR_SET(error, ...)             \
  ::superdex::recorder::detail::SetFormattedError( \
      error, ::mochi::Format(__VA_ARGS__), __FILE__, __LINE__)
