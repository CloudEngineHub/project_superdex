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

// Loading and saving pipeline files (meshing/processing_modifiers/processing_serialization.h) with
// the real modifiers.

#include "meshing/processing_modifiers/processing_serialization.h"

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/file_utils.h>

#include <picojson/picojson.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

using namespace superdex::studio;
namespace fs = std::filesystem;

namespace {

class ProcessingSerializationTest : public ::testing::Test {
 protected:
  // Writes @p document as a pipeline file, loads it and saves it again. Returns the saved document.
  picojson::object LoadAndSave(std::string const& document) {
    fs::path const path = _root / "intermediates" / "part.StudioProcessing.json";
    fs::create_directories(path.parent_path());
    std::ofstream(path) << document;
    LoadedPipeline loaded;
    mochi::Error error;
    EXPECT_TRUE(LoadProcessingPipeline(path.string(), path, loaded, error))
        << error.GetDescription();
    std::string const text =
        SerializeProcessingPipeline(loaded.modifiers, loaded.editorState, {}, path);
    picojson::value saved;
    std::string parseError;
    picojson::parse(saved, text.begin(), text.end(), &parseError);
    EXPECT_EQ(parseError, "");
    return saved.get<picojson::object>();
  }

  static picojson::object const& Properties(picojson::object const& document, int modifier) {
    return document.at("modifiers")
        .get<picojson::array>()
        .at(modifier)
        .get<picojson::object>()
        .at("properties")
        .get<picojson::object>();
  }

  mochi::TempDirCleanup _rootCleanup =
      mochi::CreateTempDirectory("processing_serialization_test", mochi::test::ExpectOK{});
  fs::path _root = _rootCleanup.Path();
};

} // namespace

// Before the STEP mesher's rewrite the Isotropic backend was called CadMesher, and bodySelection
// chose the bodies to mesh, 0 for all of them, which is what combineTouchingSolids now defaults to.
// Isotropic is also the backend's default, so only the reader's silence shows that CadMesher was
// recognized rather than replaced.
TEST_F(ProcessingSerializationTest, ReadsCadSourcesSavedBeforeTheStepMesherRewrite) {
  testing::internal::CaptureStdout();
  picojson::object const saved = LoadAndSave(R"({
  "version": 1,
  "modifiers": [
    {"modifier": "Source from CAD Model", "method": "From Model Viewer", "enabled": true,
     "collapsed": true,
     "properties": {"backend": "CadMesher", "bodySelection": 0, "linearDeflection": 0.02}},
    {"modifier": "Source from CAD Model", "method": "From File", "enabled": false,
     "collapsed": true,
     "properties": {"path": "../cad/part.step",
                    "step": {"backend": "CadMesher", "bodySelection": 0}}}
  ]
})");
  std::string const log = testing::internal::GetCapturedStdout();

  EXPECT_EQ(log.find("[Simple Reflection]"), std::string::npos) << log;
  picojson::object const& viewer = Properties(saved, 0);
  EXPECT_EQ(viewer.at("backend").get<std::string>(), "Isotropic");
  EXPECT_TRUE(viewer.at("combineTouchingSolids").get<bool>());
  EXPECT_EQ(viewer.count("bodySelection"), 0u);
  EXPECT_DOUBLE_EQ(viewer.at("linearDeflection").get<double>(), 0.02);
  picojson::object const& step = Properties(saved, 1).at("step").get<picojson::object>();
  EXPECT_EQ(step.at("backend").get<std::string>(), "Isotropic");
  EXPECT_TRUE(step.at("combineTouchingSolids").get<bool>());
  EXPECT_EQ(step.count("bodySelection"), 0u);
}
