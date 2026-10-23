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

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import rerun as rr
import rerun.experimental as rr_experimental
import superdex.physics as sdp
import superdex.recorder as recorder
from superdex.physics.utils.scene_helpers import create_scene_from_prefab, find_actor
from superdex.recorder.reader import EpisodeReader

# The Rerun wheel does not expose its experimental API to static type checkers.
_rr_experimental: Any = rr_experimental

DT = 0.01
# Integration methods the restore tests run under.
METHODS = (
    sdp.IntegrationMethod.BACKWARD_EULER,
    sdp.IntegrationMethod.BDF2,
    sdp.IntegrationMethod.BDF3,
)
# The diagonally implicit Runge-Kutta methods.
DIRK_METHODS = (
    sdp.IntegrationMethod.DIRK22,
    sdp.IntegrationMethod.DIRK23,
    sdp.IntegrationMethod.DIRK33,
    sdp.IntegrationMethod.SYMPLECTIC_DIRK12,
    sdp.IntegrationMethod.SYMPLECTIC_DIRK22,
)
PICKUP_DT = 0.005
PICKUP_OPEN_POSE = (-0.43, -0.06, 0.06)
PICKUP_CLOSED_POSE = (-0.43, 0.0, 0.0)
PICKUP_LIFTED_POSE = (-0.1, 0.0, 0.0)
CUBE_HALF_SIZE = 0.05
# Off the actor origin, so the root transform is rounded from the simulated COM state.
CUBE_MESH_OFFSET = (0.0371, 0.0123, -0.0277)
CUBE_TRIANGLES = (
    (0, 2, 1),
    (1, 2, 3),
    (4, 5, 6),
    (5, 7, 6),
    (0, 1, 5),
    (0, 5, 4),
    (2, 6, 3),
    (3, 6, 7),
    (0, 4, 2),
    (2, 4, 6),
    (1, 3, 5),
    (3, 7, 5),
)


def pickup_target(frame: int) -> list[float]:
    if frame <= 120:
        start, end, alpha = PICKUP_OPEN_POSE, PICKUP_CLOSED_POSE, frame / 120
    else:
        start, end = PICKUP_CLOSED_POSE, PICKUP_LIFTED_POSE
        alpha = min(1.0, max(0.0, (frame - 180) / 150))
    return [(1 - alpha) * a + alpha * b for a, b in zip(start, end)]


class PickupScene:
    def __init__(self, scene: sdp.Scene) -> None:
        self.scene = scene
        self.robot = find_actor(scene, "RobotArmHand")
        self.block = find_actor(scene, "PickupBlock")

    def joint_positions(self) -> list[float]:
        positions = sdp.DynamicArrayReal(self.robot.get_num_dofs())
        self.robot.get_articulated_pose(positions)
        return list(positions)

    def target_pose(self) -> list[float]:
        pose = sdp.DynamicArrayReal(self.robot.get_num_dofs())
        self.robot.get_articulated_target_pose(pose)
        return list(pose)


class CubeRecording:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.forces: dict[int, list[float]] = {}
        self.translations: dict[int, list[float]] = {}


class PickupRecording:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.block_translations: dict[int, list[float]] = {}
        self.joint_positions: dict[int, list[float]] = {}


class EpisodeRestorerTest(unittest.TestCase):
    def setUp(self) -> None:
        sdp.initialize(num_worker_threads=0)
        self.addCleanup(sdp.shutdown)
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def make_cube_scene(
        self,
        method: sdp.IntegrationMethod,
        *,
        cube_name: str = "cube",
        ground_height: float = 0.0,
    ) -> tuple[sdp.Scene, sdp.Actor]:
        scene = sdp.create_scene("cube")
        self.addCleanup(sdp.destroy_scene, scene)
        solver_params = scene.get_solver_params()
        solver_params.integration_method = method
        scene.set_solver_params(solver_params)
        scene.set_gravity([0, -10, 0])
        contact = sdp.ContactParams(
            penalty_coefficient=1e8,
            penalty_threshold_default=0.005,
            viscous_friction_coefficient=2,
        )
        scene.create_rigid_actor(
            name="ground",
            shape=sdp.create_plane_shape(normal=[0, 1, 0], distance=0),
            world_from_local=sdp.TransformRT(translation=[0, ground_height, 0]),
            is_static=True,
            contact=contact,
        )
        h = CUBE_HALF_SIZE
        dx, dy, dz = CUBE_MESH_OFFSET
        shape = sdp.create_tri_mesh_shape(
            coordinates=[
                c
                for z in (-h, h)
                for y in (-h, h)
                for x in (-h, h)
                for c in (x + dx, y + dy, z + dz)
            ],
            connectivity=[i for triangle in CUBE_TRIANGLES for i in triangle],
        )
        cube = scene.create_rigid_actor(
            name=cube_name,
            shape=shape,
            density=1000,
            world_from_local=sdp.TransformRT(translation=[-dx, h + 0.003 - dy, -dz]),
            collider_type=sdp.ColliderType.BOX,
            contact=contact,
        )
        assert cube is not None
        cube.register_query(sdp.QueryType.TOTAL_CONTACT_FORCE)
        return scene, cube

    def record_sliding_cube(
        self,
        method: sdp.IntegrationMethod,
        num_frames: int,
    ) -> CubeRecording:
        scene, cube = self.make_cube_scene(method)
        cube.set_velocity([0.5, 0, 0], [0, 0.5, 2])
        config = recorder.RecorderConfig(self.directory)
        episode = recorder.Recorder(config).begin_episode()
        physics = recorder.PhysicsRecorder(episode, scene)
        physics.record(0, scene.get_total_simulation_time())
        recording = CubeRecording(episode.output_path)
        for frame in range(1, num_frames + 1):
            scene.step(DT)
            physics.record(frame, scene.get_total_simulation_time())
            recording.forces[frame] = cube.get_contact_force_world().tolist()
            recording.translations[frame] = (
                cube.get_root_transform().translation.tolist()
            )
        episode.commit()
        return recording

    def copy_with_cube_states(
        self, source: Path, chunks: list[list[tuple[int, int]]]
    ) -> EpisodeReader:
        """Copy a cube recording, writing the cube's states as ``chunks``.

        Each chunk lists (frame, recorded frame) rows. The recording must have every
        frame from 0.
        """
        entity_path = "/data/physics/actors/cube"
        path = Path(tempfile.mkdtemp(dir=self.directory)) / "copy.rrd"
        with rr.RecordingStream("superdex_restore_test") as stream:
            stream.save(path)
            for chunk in _rr_experimental.RrdReader(source).stream():
                if chunk.entity_path == entity_path and not chunk.is_static:
                    states = chunk.to_record_batch()
                else:
                    stream.send_chunks(chunk)
            for rows in chunks:
                values = states.take([recorded for _, recorded in rows])
                stream.send_columns(
                    entity_path,
                    indexes=[rr.TimeColumn("frame", sequence=[f for f, _ in rows])],
                    columns=[
                        rr.ComponentColumn(
                            name,
                            rr.AnyBatchValue(name, pc.list_flatten(values[name])),
                            lengths=pc.list_value_length(values[name]).to_numpy(),
                        )
                        for name in values.schema.names
                        if name.startswith("superdex.physics.")
                    ],
                )
        return EpisodeReader(path)

    def make_pickup_scene(self, method: sdp.IntegrationMethod) -> PickupScene:
        scene = create_scene_from_prefab(
            "bots/internal/tests/zq_robot_arm_pickup/zq_robot_arm_pickup.mochi_scene"
        )
        self.addCleanup(sdp.destroy_scene, scene)
        solver_params = scene.get_solver_params()
        solver_params.integration_method = method
        scene.set_solver_params(solver_params)
        return PickupScene(scene)

    def record_pickup(self, method: sdp.IntegrationMethod) -> PickupRecording:
        source = self.make_pickup_scene(method)
        source.robot.set_articulated_pose_from_joints(list(PICKUP_OPEN_POSE))
        source.robot.set_articulated_target_pose(list(PICKUP_OPEN_POSE))
        config = recorder.RecorderConfig(self.directory)
        episode = recorder.Recorder(config).begin_episode()
        physics = recorder.PhysicsRecorder(episode, source.scene)
        physics.record(0, source.scene.get_total_simulation_time())
        recording = PickupRecording(episode.output_path)
        for frame in range(1, 341):
            source.robot.set_articulated_target_pose(pickup_target(frame))
            source.scene.step(PICKUP_DT)
            physics.record(frame, source.scene.get_total_simulation_time())
            recording.block_translations[frame] = (
                source.block.get_root_transform().translation.tolist()
            )
            recording.joint_positions[frame] = source.joint_positions()
        episode.commit()
        return recording

    def assert_force_near(self, actual: sdp.Real3, expected: list[float]) -> None:
        self.assert_values_near(actual.tolist(), expected, 1e-6)

    def assert_values_near(
        self, actual: list[float], expected: list[float], tolerance: float
    ) -> None:
        self.assertEqual(len(actual), len(expected))
        scaled_tolerance = max(1.0, math.hypot(*expected)) * tolerance
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, delta=scaled_tolerance)

    def test_restored_frames_match_the_recording(self) -> None:
        for method in METHODS:
            with self.subTest(method=method.name):
                recording = self.record_sliding_cube(method, 30)
                scene, cube = self.make_cube_scene(method)
                restorer = recorder.EpisodeRestorer(
                    EpisodeReader(recording.path), scene
                )
                self.assertEqual(restorer.frames, list(range(31)))

                for frame in (1, 2, 3, 10, 30):
                    self.assertGreater(math.hypot(*recording.forces[frame]), 1.0)
                    restorer.restore_frame(frame)
                    self.assertAlmostEqual(
                        scene.get_total_simulation_time(), frame * DT, delta=1e-8
                    )
                    for actual, expected in zip(
                        cube.get_root_transform().translation.tolist(),
                        recording.translations[frame],
                    ):
                        self.assertAlmostEqual(actual, expected, delta=1e-6)

    def test_restores_frames_under_dirk_methods(self) -> None:
        for method in DIRK_METHODS:
            with self.subTest(method=method.name):
                recording = self.record_sliding_cube(method, 2)
                scene, cube = self.make_cube_scene(method)
                restorer = recorder.EpisodeRestorer(
                    EpisodeReader(recording.path), scene
                )
                restorer.restore_frame(2)
                for actual, expected in zip(
                    cube.get_root_transform().translation.tolist(),
                    recording.translations[2],
                ):
                    self.assertAlmostEqual(actual, expected, delta=1e-6)

    def test_rejects_scenes_that_do_not_match_the_recording(self) -> None:
        method = sdp.IntegrationMethod.BACKWARD_EULER
        reader = EpisodeReader(self.record_sliding_cube(method, 2).path)

        scene, _ = self.make_cube_scene(method, cube_name="block")
        with self.assertRaisesRegex(ValueError, "missing recorded actor 'cube'"):
            recorder.EpisodeRestorer(reader, scene)
        scene, _ = self.make_cube_scene(method, ground_height=0.01)
        with self.assertRaisesRegex(ValueError, "'ground' differs from its recorded"):
            recorder.EpisodeRestorer(reader, scene)
        scene, _ = self.make_cube_scene(method)
        with self.assertRaisesRegex(ValueError, "Frame 3 was not recorded"):
            recorder.EpisodeRestorer(reader, scene).restore_frame(3)

        scene.create_rigid_actor(
            name="wall",
            shape=sdp.create_plane_shape(normal=[1, 0, 0], distance=-1),
            is_static=True,
        )
        recorder.EpisodeRestorer(reader, scene)
        scene.create_rigid_actor(
            name="pebble",
            shape=sdp.create_tri_mesh_shape(
                coordinates=[0, 0, 0, 0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01],
                connectivity=[0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
            ),
            world_from_local=sdp.TransformRT(translation=[1, 1, 0]),
        )
        with self.assertRaisesRegex(
            ValueError, r"unrecorded dynamic actors \['pebble'\]"
        ):
            recorder.EpisodeRestorer(reader, scene)

    def test_rejects_recorded_actors_that_bind_to_one_scene_actor(self) -> None:
        method = sdp.IntegrationMethod.BACKWARD_EULER
        source, _ = self.make_cube_scene(method)
        # Recorded as ground_1, but with the runtime name of the other ground.
        source.create_rigid_actor(
            name="ground",
            shape=sdp.create_plane_shape(normal=[1, 0, 0], distance=-1),
            is_static=True,
        )
        config = recorder.RecorderConfig(self.directory)
        episode = recorder.Recorder(config).begin_episode()
        physics = recorder.PhysicsRecorder(episode, source)
        physics.record(0, source.get_total_simulation_time())
        episode.commit()

        scene, _ = self.make_cube_scene(method)
        with self.assertRaisesRegex(ValueError, "both bind to scene actor 'ground'"):
            recorder.EpisodeRestorer(EpisodeReader(episode.output_path), scene)

    def test_restores_states_written_out_of_frame_order(self) -> None:
        method = sdp.IntegrationMethod.BACKWARD_EULER
        recording = self.record_sliding_cube(method, 4)
        # Frame 2 is first written with frame 4's state, and then with its own.
        chunks = [[(4, 4), (2, 4), (3, 3)], [(2, 2), (0, 0)]]
        scene, cube = self.make_cube_scene(method)
        with self.assertRaisesRegex(
            ValueError, "'cube' has no 'root_quaternion_xyzw' at frame 1"
        ):
            recorder.EpisodeRestorer(
                self.copy_with_cube_states(recording.path, chunks), scene
            )

        restorer = recorder.EpisodeRestorer(
            self.copy_with_cube_states(recording.path, [*chunks, [(1, 1)]]), scene
        )
        for frame in range(1, 5):
            restorer.restore_frame(frame)
            self.assert_values_near(
                cube.get_root_transform().translation.tolist(),
                recording.translations[frame],
                1e-6,
            )
        with self.assertRaisesRegex(ValueError, "has a row without a frame"):
            EpisodeReader(recording.path)._component_columns(["superdex.physics.mass"])

    def test_restored_robot_frames_match_the_live_run(self) -> None:
        for method in METHODS:
            with self.subTest(method=method.name):
                recording = self.record_pickup(method)
                restored = self.make_pickup_scene(method)
                restorer = recorder.EpisodeRestorer(
                    EpisodeReader(recording.path), restored.scene
                )
                # The block's center of mass is off its origin, and it rests before the grasp.
                for frame in range(1, 341):
                    restorer.restore_frame(frame)
                    self.assert_values_near(
                        restored.target_pose(), pickup_target(frame), 1e-6
                    )
                    self.assert_values_near(
                        restored.block.get_root_transform().translation.tolist(),
                        recording.block_translations[frame],
                        1e-6,
                    )
                    self.assert_values_near(
                        restored.joint_positions(),
                        recording.joint_positions[frame],
                        1e-6,
                    )
