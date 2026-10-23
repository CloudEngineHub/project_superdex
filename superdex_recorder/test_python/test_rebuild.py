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
import json
import os
import tempfile
import unittest
from pathlib import Path

import superdex.physics as sdp
import superdex.recorder as recorder
import superdex.robotics as sdr
from superdex.physics.paths import resolve_asset
from superdex.physics.utils.scene_helpers import load_bot_scene
from superdex.recorder.reader import EpisodeReader

BOT_SCENE_PATH = resolve_asset(
    "bots/internal/tests/bot_scene/test_bot_scene.mochi_bot_scene"
)
SKINNED_HAND_PATH = resolve_asset(
    "bots/hands/wuji_hand2_beta1/left/wuji_hand2_beta1_skinned_left.superdex_bot"
)
SPAWN_NAME = "shape_filter"
SPAWN_TRANSLATION = [0.05, 0.0, 0.0]
RANDOMIZED_SPAWN_TRANSLATION = [0.06, 0.01, 0.0]
FRICTION_SCALE = 1.4
DENSITY_SCALE = 0.8
NUM_FRAMES = 100
TIME_STEP = 0.005


def physics_scene(bot_scene: sdr.BotScene) -> sdp.Scene:
    scene = bot_scene.get_scene()
    assert scene is not None
    return scene


def scene_actors(scene: sdp.Scene) -> dict[str, sdp.Actor]:
    actors: dict[str, sdp.Actor] = {}
    scene.for_each_actor(lambda actor: actors.__setitem__(actor.get_name(), actor))
    return actors


def colliding_bodies(scene: sdp.Scene) -> dict[str, sdp.Actor]:
    return {
        name: actor
        for name, actor in scene_actors(scene).items()
        if actor.get_type() == sdp.ActorType.RIGID
        and actor.get_collider_type() != sdp.ColliderType.NONE
    }


def randomize_physics(scene: sdp.Scene) -> None:
    for actor in colliding_bodies(scene).values():
        contact = actor.get_contact_params()
        contact.coulomb_friction_coefficient *= FRICTION_SCALE
        contact.viscous_friction_coefficient *= FRICTION_SCALE
        actor.set_contact_params(contact)
        if not actor.is_static():
            actor.set_density(DENSITY_SCALE * actor.get_density())


def physical_parameters(
    actor: sdp.Actor, contact_param_names: list[str]
) -> dict[str, list[float]]:
    contact = actor.get_contact_params()
    parameters = {"contact": [getattr(contact, name) for name in contact_param_names]}
    if actor.is_static():
        pose = actor.get_root_transform()
        parameters["pose"] = pose.translation.tolist() + pose.rotation.tolist()
    else:
        parameters["mass_properties"] = [actor.get_density(), actor.get_mass()]
    return parameters


class RebuildSceneTest(unittest.TestCase):
    def setUp(self) -> None:
        sdp.initialize(num_worker_threads=0)
        self.addCleanup(sdp.shutdown)
        self.addCleanup(gc.collect)
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.context = sdr.create_context()
        self.task_path = self.directory / "shape_filter.mochi_bot_task"
        self.task_path.write_text(
            json.dumps(
                {
                    "metadata": {"name": "shape_filter"},
                    "spawns": [
                        {
                            "name": SPAWN_NAME,
                            "prefabName": SPAWN_NAME,
                            "parent": "task_root",
                            "parentFromSpawn": {
                                "rotation": [0, 0, 0, 1],
                                "translation": SPAWN_TRANSLATION,
                            },
                        }
                    ],
                }
            )
        )

    def load_live_scene(
        self, spawn_translation: list[float] = SPAWN_TRANSLATION
    ) -> sdr.BotScene:
        """Load the bot scene and spawn the task, like Unreal does, which applies
        randomized spawn poses as it spawns."""
        bot_scene = load_bot_scene(str(BOT_SCENE_PATH), self.context)
        prefab_path = next(
            str(entry.path)
            for entry in bot_scene.get_bot_scene_prefab().scene.spawnable_prefabs
            if str(entry.name) == SPAWN_NAME
        )
        params = sdp.prefab.PrefabParams()
        params.name = SPAWN_NAME
        params.translation = spawn_translation
        params.apply_scene_settings = False
        sdp.prefab.add_to_scene(
            prefab_path, os.path.dirname(prefab_path), physics_scene(bot_scene), params
        )
        return bot_scene

    def record(
        self,
        scene: sdp.Scene,
        num_frames: int,
        metadata: recorder.EpisodeMetadata | None = None,
    ) -> tuple[Path, Path]:
        episode_recorder = recorder.Recorder(recorder.RecorderConfig(self.directory))
        episodes = [
            episode_recorder.begin_episode(metadata or recorder.EpisodeMetadata())
            for _ in range(2)
        ]
        config = recorder.PhysicsRecorderConfig()
        config.record_contact_forces = True
        recorders = [
            recorder.PhysicsRecorder(episodes[0], scene),
            recorder.PhysicsRecorder(episodes[1], scene, config),
        ]
        for frame in range(num_frames + 1):
            if frame > 0:
                scene.step(TIME_STEP)
            for physics in recorders:
                physics.record(frame, scene.get_total_simulation_time())
        for episode in episodes:
            episode.commit()
        return episodes[0].output_path, episodes[1].output_path

    def test_rebuilt_scene_has_randomized_physical_parameters(self) -> None:
        live = self.load_live_scene(RANDOMIZED_SPAWN_TRANSLATION)
        live_scene = physics_scene(live)
        randomize_physics(live_scene)
        kinematic_path, _ = self.record(live_scene, NUM_FRAMES)
        live_bodies = colliding_bodies(live_scene)

        recording = EpisodeReader(kinematic_path)
        rebuilt = recorder.rebuild_scene(
            recording, BOT_SCENE_PATH, self.context, bot_task_path=self.task_path
        )
        scene = physics_scene(rebuilt.bot_scene)
        bodies = colliding_bodies(scene)
        self.assertCountEqual(
            rebuilt.spawned_actors[SPAWN_NAME],
            [name for name in live_bodies if name.startswith(f"{SPAWN_NAME}/")],
        )
        contact_param_names = recording.read_static_value(
            "superdex.physics.contact_param_names"
        )
        for name, live_body in live_bodies.items():
            with self.subTest(body=name):
                self.assertEqual(
                    physical_parameters(bodies[name], contact_param_names),
                    physical_parameters(live_body, contact_param_names),
                )

    def test_applies_recorded_contact_parameters_of_a_skinned_hand(self) -> None:
        """A skinned articulation keeps its contact skin's parameters on the root."""
        prefab = sdr.load_bot_prefab_from_file(str(SKINNED_HAND_PATH))
        live_scene = sdp.create_scene("live")
        live_hand = sdr.create_bot(
            live_scene, prefab, self.context
        ).get_articulated_actor()
        contact = live_hand.get_contact_params()
        contact.coulomb_friction_coefficient *= FRICTION_SCALE
        live_hand.set_contact_params(contact)
        path, _ = self.record(live_scene, num_frames=1)

        rebuilt_scene = sdp.create_scene("rebuilt")
        rebuilt_hand = sdr.create_bot(
            rebuilt_scene, prefab, self.context
        ).get_articulated_actor()
        self.assertNotEqual(
            rebuilt_hand.get_contact_params().coulomb_friction_coefficient,
            contact.coulomb_friction_coefficient,
        )
        recording = EpisodeReader(path)
        recorder.apply_recorded_physics(recording, rebuilt_scene)

        names = recording.read_static_value("superdex.physics.contact_param_names")
        self.assertEqual(
            [getattr(rebuilt_hand.get_contact_params(), name) for name in names],
            [getattr(live_hand.get_contact_params(), name) for name in names],
        )

    def test_rejects_bot_scene_that_differs_from_recording(self) -> None:
        live = self.load_live_scene()
        kinematic_path, _ = self.record(
            physics_scene(live),
            1,
            recorder.EpisodeMetadata(
                {
                    "scene.archive.filename": BOT_SCENE_PATH.name,
                    "scene.archive.hash": "0000000000000000",
                    "scene.archive.hash_algorithm": "xxh3_64",
                }
            ),
        )
        with self.assertRaisesRegex(ValueError, "differs from the scene.archive"):
            recorder.rebuild_scene(
                EpisodeReader(kinematic_path),
                BOT_SCENE_PATH,
                self.context,
                bot_task_path=self.task_path,
            )

    def test_requires_recorded_bot_task(self) -> None:
        live = self.load_live_scene()
        kinematic_path, _ = self.record(physics_scene(live), 1)
        with self.assertRaisesRegex(ValueError, "missing recorded actor"):
            recorder.rebuild_scene(
                EpisodeReader(kinematic_path), BOT_SCENE_PATH, self.context
            )

    def test_rejects_static_actor_without_recorded_pose(self) -> None:
        live = load_bot_scene(str(BOT_SCENE_PATH), self.context)
        episode = recorder.Recorder(
            recorder.RecorderConfig(self.directory)
        ).begin_episode()
        # Static poses are recorded with the first frame, which this episode never has.
        recorder.PhysicsRecorder(episode, physics_scene(live))
        episode.commit()
        with self.assertRaisesRegex(
            ValueError, r"Static actor '.+' has no recorded pose"
        ):
            recorder.rebuild_scene(
                EpisodeReader(episode.output_path), BOT_SCENE_PATH, self.context
            )
