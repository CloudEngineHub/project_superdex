# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#-----------------------------------------------------------------------------
# superdex_asset_tuner_static
#-----------------------------------------------------------------------------

file(GLOB _tuner_srcs
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
)
file(GLOB _tuner_headers "${CMAKE_CURRENT_SOURCE_DIR}/include/superdex_asset_tuner/*.h")

add_library(superdex_asset_tuner_static STATIC ${_tuner_srcs} ${_tuner_headers})

target_include_directories(superdex_asset_tuner_static
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(superdex_asset_tuner_static
    PUBLIC
        mochi_core
        mochi_physics
)

target_compile_definitions(superdex_asset_tuner_static PRIVATE MOCHI_USE_REFLECTION=1)

mochi_set_compiler_options(superdex_asset_tuner_static)
mochi_set_output_directories(superdex_asset_tuner_static)

if (MOCHI_BUILD_SHARED)
    set_target_properties(superdex_asset_tuner_static PROPERTIES
        POSITION_INDEPENDENT_CODE ${MOCHI_BUILD_SHARED}
    )
endif ()

#-----------------------------------------------------------------------------
# superdex_asset_tuner_test
#-----------------------------------------------------------------------------

if (MOCHI_BUILD_TESTS)
    # The tests resolve prefab shapes against the SuperDex assets root.
    set(MOCHI_TEST_ASSETS_DIR "${PROJECT_SOURCE_DIR}/assets")
    mochi_add_test_executable(superdex_asset_tuner_test "${CMAKE_CURRENT_SOURCE_DIR}/test")
    target_include_directories(superdex_asset_tuner_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/test")
    # The tests have no main.cpp of their own.
    target_link_libraries(superdex_asset_tuner_test PRIVATE superdex_asset_tuner_static gtest_main)
endif ()

unset(_tuner_srcs)
unset(_tuner_headers)
