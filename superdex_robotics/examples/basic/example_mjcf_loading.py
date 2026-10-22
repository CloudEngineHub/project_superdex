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
render models. Mesh files resolve against the MJCF's <compiler meshdir>, relative
to the .xml file, so no preprocessing is required; an SDF collider is baked on the
fly when the bot is created.

The bundled model is the MuJoCo Menagerie Franka FR3 v2. Each of its bodies has one
collision mesh and several visual meshes, and every extra mesh geom becomes its own
Hard-jointed child link, so the 7-DoF arm loads as 30 links.

PITFALL: The runtime MJCF loader only loads *mesh* geoms. Primitive geoms (box,
capsule, cylinder, ellipsoid, sphere) are skipped with a warning and have no
collision geometry.

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
    """Resolve the path to the bundled FR3 v2 test MJCF."""
    return str(resolve_asset("test/mjcf/franka_fr3_v2/fr3v2.xml"))


def main() -> None:
    """Load a robot from an MJCF file and simulate it."""
    parser = argparse.ArgumentParser(description="MJCF bot loading example")
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=None,
        help="Path to an MJCF .xml file (default: franka_fr3_v2/fr3v2.xml)",
    )
    args = parser.parse_args()

    mjcf_path = args.path if args.path else get_default_mjcf_path()

    # Initialize the physics engine before creating scenes or actors.
    # num_worker_threads=0 runs single-threaded; pass -1 to auto-select.
    sdp.initialize(num_worker_threads=0)

    # Create an empty scene. MuJoCo and SuperDex both use Z-up, so gravity points
    # down the -Z axis.
    scene = sdp.create_scene("MJCF Bot Loading Example")
    scene.set_gravity([0, 0, -9.81])

    # Parse the MJCF into a bot prefab: a template describing links and joints.
    # A top-level body without a <freejoint> is welded to the world, so the arm's
    # base stays put. The "home" keyframe becomes the bot's default pose. Content
    # with no SuperDex equivalent (sites, actuator gains) is skipped with a warning.
    # The arm still sags under gravity since there is no controller.
    bot_prefab = sdr.load_bot_prefab_from_mjcf_file(mjcf_path)

    # Instantiate the prefab as a live Bot in the scene. This builds the robot's
    # articulated actor and bakes each link's SDF collider from its mesh. The
    # robotics context tracks every bot and controller you create.
    robotics_context = sdr.create_context()
    bot = sdr.create_bot(scene, bot_prefab, robotics_context)
    bot_actor = bot.get_articulated_actor()

    # Add a static ground plane for the robot to rest on (normal points up, +Z).
    plane_shape = sdp.create_plane_shape(normal=[0, 0, 1], distance=0)
    scene.create_rigid_actor(name="ground", shape=plane_shape, is_static=True)

    # Print a quick summary of the loaded robot.
    print(f"Robot: {bot_prefab.name}")
    print(f"  Links: {len(bot_prefab.links)}")
    print(f"  Joints: {len(bot_prefab.joints)}")
    print(f"  DOFs: {bot_actor.get_num_dofs()}")

    # Simulate at 60 Hz (each step advances 1/60 of a second).
    time_step = 1.0 / 60.0

    # Declare the scene's coordinate convention so the debugger renders it the
    # right way up: SuperDex is X-forward, Y-left, Z-up (FLU). Must come before
    # attach(), which starts the server.
    sdp.get_debug_server().set_coordinate_space(
        sdp.CoordinateSpace(axes=sdp.CoordinateSpaceAxes.FLU)
    )

    # Launch and connect the SuperDex Physics Debugger, a separate desktop app for
    # viewing and interacting with the simulation. The loop runs until you close
    # the debugger; attach() returns False if it can't connect.
    if sdp.debugger.attach():
        while sdp.debugger.is_attached():
            scene.step(time_step)

    # Tear down: destroy the bot, then shut the engine down cleanly.
    sdr.destroy_bot(scene, bot)
    sdp.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
