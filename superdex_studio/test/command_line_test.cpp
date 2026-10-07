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

// Covers superdex_studio's command-line parsing (app/command_line.h).

#include "app/command_line.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace superdex::studio;

namespace {

// Parses @p args, expecting success.
CommandLine Parse(std::vector<std::string> const& args) {
  CommandLine commandLine;
  std::string error;
  EXPECT_TRUE(ParseCommandLine(args, commandLine, error)) << error;
  return commandLine;
}

// Parses @p args, expecting failure, and returns the error.
std::string ParseError(std::vector<std::string> const& args) {
  CommandLine commandLine;
  std::string error;
  EXPECT_FALSE(ParseCommandLine(args, commandLine, error));
  return error;
}

} // namespace

TEST(CommandLineTest, NoOptionsStartTheGui) {
  CommandLine const commandLine = Parse({});

  EXPECT_FALSE(commandLine.help);
  EXPECT_FALSE(commandLine.process.has_value());
}

TEST(CommandLineTest, TheGuiStillIgnoresFileArguments) {
  EXPECT_FALSE(Parse({"model.step"}).process.has_value());
}

TEST(CommandLineTest, TheGuiListsTheUnknownOptionsItIgnores) {
  CommandLine const commandLine = Parse({"--procss", "a.json", "--verbose"});

  EXPECT_FALSE(commandLine.process.has_value());
  EXPECT_EQ(commandLine.ignoredOptions, (std::vector<std::string>{"--procss", "--verbose"}));
}

TEST(CommandLineTest, ProcessTakesEveryOptionInEitherSpelling) {
  CommandLine const commandLine = Parse(
      {"--process",
       "intermediates/part.StudioProcessing.json",
       "--out=build",
       "--cad",
       "cad/part.step",
       "--render=render/part.glb",
       "--mochi",
       "collision/part.mochi.h5"});

  ASSERT_TRUE(commandLine.process.has_value());
  EXPECT_EQ(commandLine.process->pipelinePath, "intermediates/part.StudioProcessing.json");
  EXPECT_EQ(commandLine.process->outDir, "build");
  EXPECT_EQ(commandLine.process->slots.cadPath, "cad/part.step");
  EXPECT_EQ(commandLine.process->slots.renderPath, "render/part.glb");
  EXPECT_EQ(commandLine.process->slots.mochiPath, "collision/part.mochi.h5");
}

TEST(CommandLineTest, HelpIsRecognizedInBothSpellings) {
  EXPECT_TRUE(Parse({"--help"}).help);
  EXPECT_TRUE(Parse({"-h"}).help);
}

TEST(CommandLineTest, AValueThatBeginsWithDashesTakesTheEqualsForm) {
  CommandLine const commandLine = Parse({"--process", "a.json", "--cad=--odd.step"});
  ASSERT_TRUE(commandLine.process.has_value());
  EXPECT_EQ(commandLine.process->slots.cadPath, "--odd.step");
}

TEST(CommandLineTest, MistakesAreReported) {
  EXPECT_EQ(ParseError({"--process", "a.json", "--procss"}), "unknown option '--procss'");
  EXPECT_EQ(ParseError({"--process"}), "--process needs a value");
  EXPECT_EQ(ParseError({"--process="}), "--process needs a value");
  EXPECT_EQ(ParseError({"--process", "--out", "x"}), "--process needs a value");
  EXPECT_EQ(
      ParseError({"--process", "a.json", "--out", "x", "--out", "y"}),
      "--out is given more than once");
  EXPECT_EQ(ParseError({"--out", "x"}), "--out only applies with --process");
  EXPECT_EQ(ParseError({"--process", "a.json", "b.json"}), "unexpected argument 'b.json'");
}
