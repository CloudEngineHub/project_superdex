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

#include "app/command_line.h"

#include <array>
#include <string_view>
#include <utility>

namespace superdex::studio {

bool ParseCommandLine(std::vector<std::string> const& args, CommandLine& out, std::string& error) {
  out = {};
  ProcessOptions process;
  // The options that take a value, and where each value goes.
  std::array<std::pair<std::string_view, std::string*>, 5> const valueOptions{{
      {"--process", &process.pipelinePath},
      {"--out", &process.outDir},
      {"--cad", &process.slots.cadPath},
      {"--render", &process.slots.renderPath},
      {"--mochi", &process.slots.mochiPath},
  }};
  std::vector<std::string> positional;
  std::vector<std::string> unknown;

  for (std::size_t i = 0; i < args.size(); ++i) {
    std::string_view const arg = args[i];
    if (arg == "-h" || arg == "--help") {
      out.help = true;
      continue;
    }
    if (!arg.starts_with("--")) {
      positional.push_back(args[i]);
      continue;
    }
    std::size_t const equals = arg.find('=');
    std::string_view const name = arg.substr(0, equals);
    std::string* target = nullptr;
    for (auto const& [option, destination] : valueOptions) {
      if (name == option) {
        target = destination;
      }
    }
    if (target == nullptr) {
      unknown.emplace_back(name);
      continue;
    }
    std::string value;
    if (equals != std::string_view::npos) {
      value = arg.substr(equals + 1);
    } else if (i + 1 < args.size() && !args[i + 1].starts_with("--")) {
      value = args[++i];
    }
    if (value.empty()) {
      error = std::string(name) + " needs a value";
      return false;
    }
    if (!target->empty()) {
      error = std::string(name) + " is given more than once";
      return false;
    }
    *target = std::move(value);
  }

  if (process.pipelinePath.empty()) {
    for (auto const& [option, destination] : valueOptions) {
      if (!destination->empty()) {
        error = std::string(option) + " only applies with --process";
        return false;
      }
    }
    out.ignoredOptions = std::move(unknown);
    return true;
  }
  if (!unknown.empty()) {
    error = "unknown option '" + unknown.front() + "'";
    return false;
  }
  if (!positional.empty()) {
    error = "unexpected argument '" + positional.front() + "'";
    return false;
  }
  out.process = std::move(process);
  return true;
}

std::string CommandLineUsage() {
  return R"(Usage:
  superdex_studio
      Start SuperDex Studio.

  superdex_studio --process <pipeline> [--out <dir>] [--cad <file>] [--render <file>] [--mochi <file>]
      Run a Model Editor processing pipeline (a .StudioProcessing.json) without opening a window:
      every enabled modifier in order, each export writing its file as soon as its stage has
      run, as Build/Export All does. Prints one line
      per stage and exits with 0 when every stage and export succeeded, 1 otherwise.

      The models a "From Model Viewer" source reads are found next to the pipeline, the way the
      Model Editor finds them for the model that owns it: <asset>/intermediates/<name>.StudioProcessing.json
      pairs with <name>.step, <name>.glb and so on in <asset>. Exports on Auto are named after the
      model the source read.

      Give a value that begins with -- as --option=<value>.

      --out <dir>      Write every export into <dir>, keeping each file name.
      --cad <file>     The CAD model (STEP or STL) to use instead.
      --render <file>  The render model to use instead.
      --mochi <file>   The mochi model (.mochi.h5) to use instead.

  superdex_studio --help
      Show this help.
)";
}

} // namespace superdex::studio
