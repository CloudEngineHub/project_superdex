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

import gc
import tempfile
import unittest
from pathlib import Path

import superdex.physics as sdp
import superdex.recorder as recorder
from superdex_recorder_test_helpers.test_python.rrd_reader import EpisodeReader


class RecorderTest(unittest.TestCase):
    def make_recorder(self, directory: Path) -> recorder.Recorder:
        return recorder.Recorder(recorder.RecorderConfig(directory))

    def make_metadata(self) -> recorder.EpisodeMetadata:
        return recorder.EpisodeMetadata(
            properties={
                "/": "slash",
                "\\/": "escaped slash",
                "build revision": "abc123",
            },
        )

    def test_committed_rrd_contains_episode_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            episode = self.make_recorder(directory).begin_episode(self.make_metadata())
            episode.commit()

            self.assertTrue(episode.output_path.exists())
            recording = EpisodeReader(episode.output_path)
            self.assertEqual(
                recording.read_static_value("superdex.episode.coordinate_space"),
                "FLU",
            )
            self.assertEqual(
                recording.read_static_value("superdex.episode.units_per_meter"),
                1.0,
            )
            property_keys = recording.component_samples("superdex.episode.property_key")
            self.assertCountEqual(
                [sample.value[0] for sample in property_keys],
                ["/", "\\/", "build revision"],
            )
            # `/` escapes to `\/`, so the key `\/` must still get its own path.
            self.assertCountEqual(
                {sample.entity_path for sample in property_keys},
                {
                    r"/episode/properties/\/",
                    r"/episode/properties/\\\/",
                    r"/episode/properties/build\ revision",
                },
            )

    def test_context_exit_without_commit_discards_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            with self.make_recorder(directory).begin_episode(self.make_metadata()):
                pass
            self.assertEqual(list(directory.iterdir()), [])

    def test_exception_in_context_propagates_and_discards_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            error = RuntimeError("controller failed")
            with self.assertRaises(RuntimeError) as raised:
                with self.make_recorder(directory).begin_episode() as episode:
                    raise error
            self.assertIs(raised.exception, error)
            self.assertFalse(episode.is_active)
            self.assertEqual(list(directory.iterdir()), [])

    def test_physics_recorder_owns_episode_for_its_lifetime(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            sdp.initialize(num_worker_threads=0)
            try:
                scene = sdp.create_scene("recorder lifetime test")
                episode = self.make_recorder(directory).begin_episode()
                physics = recorder.PhysicsRecorder(episode, scene)
                partial_files = list(directory.glob("*.rrd.partial"))
                self.assertEqual(len(partial_files), 1)

                del episode
                gc.collect()
                self.assertTrue(partial_files[0].exists())

                del physics
                gc.collect()
                self.assertEqual(list(directory.iterdir()), [])
            finally:
                sdp.shutdown()

    def test_duplicate_actor_names_get_distinct_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            sdp.initialize(num_worker_threads=0)
            try:
                scene = sdp.create_scene("duplicate actor names test")
                dynamic_shape = sdp.create_tri_mesh_shape(
                    coordinates=[
                        0,
                        0,
                        0,
                        0.1,
                        0,
                        0,
                        0,
                        0.1,
                        0,
                        0,
                        0,
                        0.1,
                    ],
                    connectivity=[0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
                )
                dynamic_actor = scene.create_rigid_actor(
                    name="cube",
                    shape=dynamic_shape,
                    density=1000,
                )
                static_shape = sdp.create_plane_shape(normal=[0, 0, 1], distance=0)
                static_actor = scene.create_rigid_actor(
                    name="cube",
                    shape=static_shape,
                    is_static=True,
                )
                self.assertIsNotNone(dynamic_actor)
                self.assertIsNotNone(static_actor)

                episode = self.make_recorder(directory).begin_episode()
                physics = recorder.PhysicsRecorder(episode, scene)
                physics.record(0, 0.0)
                scene.step(0.01)
                physics.record(1, scene.get_total_simulation_time())
                episode.commit()

                recording = EpisodeReader(episode.output_path)
                actor_ids = recording.component_samples("superdex.physics.actor_id")
                self.assertCountEqual(
                    [sample.value[0] for sample in actor_ids],
                    ["cube", "cube_1"],
                )
                self.assertEqual(len({sample.entity_path for sample in actor_ids}), 2)
                static_by_path = {
                    sample.entity_path: bool(sample.value[0])
                    for sample in recording.component_samples(
                        "superdex.physics.is_static"
                    )
                }
                self.assertCountEqual(static_by_path.values(), [False, True])
                root_poses = recording.component_samples(
                    "superdex.physics.root_translation_xyz"
                )
                for actor_path, is_static in static_by_path.items():
                    sample_frames = [
                        sample.timelines["frame"]
                        for sample in root_poses
                        if sample.entity_path == actor_path
                    ]
                    self.assertEqual(sample_frames, [0] if is_static else [0, 1])
            finally:
                sdp.shutdown()
