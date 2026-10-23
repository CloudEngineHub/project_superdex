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

import superdex.physics as sdp
import superdex.recorder as recorder
from superdex.recorder.reader import EpisodeReader

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


class CubeRecording:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.forces: dict[int, list[float]] = {}
        self.translations: dict[int, list[float]] = {}


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

    def assert_force_near(self, actual: sdp.Real3, expected: list[float]) -> None:
        tolerance = max(1.0, math.hypot(*expected)) * 1e-6
        for actual_value, expected_value in zip(actual.tolist(), expected):
            self.assertAlmostEqual(actual_value, expected_value, delta=tolerance)

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
