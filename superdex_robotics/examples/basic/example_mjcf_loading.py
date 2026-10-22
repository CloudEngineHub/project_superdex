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

"""Example: Loading a robot from an MJCF (MuJoCo XML) file with `superdex.robotics`

Loads a robot directly from a MuJoCo .xml file and simulates it. Bodies become
links, <inertial> becomes link inertia, and mesh geoms become collision shapes and
render models. A body with several geoms gets one extra Hard-jointed link per geom.

PITFALL: The runtime MJCF loader only loads *mesh* geoms. Primitive geoms (box,
capsule, cylinder, ellipsoid, sphere) are skipped with a warning, so the bundled
arm's box "tool" has no collision geometry here.

PITFALL: Raw MuJoCo meshes are used as-is. Some menagerie meshes have degenerate
triangles or are not closed, and Mochi cannot use those as dynamic link shapes.

NOTE: For production assets, prefer importing MJCF files through SuperDex Studio.
Its importer converts primitives, merges each body's geoms into one watertight
collider, and emits a .superdex_bot with preprocessed assets. This runtime path is
best for quick prototyping.

Usage:
    cd <path_to_superdex_robotics>
    python3 examples/basic/example_mjcf_loading.py
    python3 examples/basic/example_mjcf_loading.py /path/to/robot.xml
"""

import argparse

import superdex.physics as sdp
import superdex.robotics as sdr
from superdex.physics.paths import resolve_asset


def get_default_mjcf_path() -> str:
    """Resolve the path to the bundled FR3 arm MJCF."""
    return str(resolve_asset("test/mjcf/fr3_arm.xml"))


def main() -> None:
    """Load a robot from an MJCF file and simulate it."""
    parser = argparse.ArgumentParser(description="MJCF bot loading example")
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=None,
        help="Path to an MJCF .xml file (default: fr3_arm.xml)",
    )
    args = parser.parse_args()

    mjcf_path = args.path if args.path else get_default_mjcf_path()

    # Initialize the physics engine before creating scenes or actors.
    sdp.initialize(num_worker_threads=0)

    # MuJoCo and SuperDex both use Z-up, so gravity points down the -Z axis.
    scene = sdp.create_scene("MJCF Bot Loading Example")
    scene.set_gravity([0, 0, -9.81])

    # Parse the MJCF into a bot prefab. A top-level body without a <freejoint> is
    # welded to the world, so the arm's base stays put. The "home" keyframe
    # becomes the bot's default pose.
    bot_prefab = sdr.load_bot_prefab_from_mjcf_file(mjcf_path)

    robotics_context = sdr.create_context()
    bot = sdr.create_bot(scene, bot_prefab, robotics_context)
    bot_actor = bot.get_articulated_actor()

    plane_shape = sdp.create_plane_shape(normal=[0, 0, 1], distance=0)
    scene.create_rigid_actor(name="ground", shape=plane_shape, is_static=True)

    print(f"Robot: {bot_prefab.name}")
    print(f"  Links: {len(bot_prefab.links)}")
    print(f"  Joints: {len(bot_prefab.joints)}")
    print(f"  DOFs: {bot_actor.get_num_dofs()}")

    time_step = 1.0 / 60.0
    sdp.get_debug_server().set_coordinate_space(
        sdp.CoordinateSpace(axes=sdp.CoordinateSpaceAxes.FLU)
    )
    if sdp.debugger.attach():
        while sdp.debugger.is_attached():
            scene.step(time_step)

    sdr.destroy_bot(scene, bot)
    sdp.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
