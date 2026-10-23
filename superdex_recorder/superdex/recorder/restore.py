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

"""Restore a Mochi scene to the recorded frames of a SuperDex episode."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow.compute as pc
import superdex.physics as sdp

from . import _extension
from .reader import EpisodeReader

_FRAME_PATH = "data/physics/frames"
_STATIC_POSE_TOLERANCE = 1e-5
_STATE_COMPONENTS = (
    "root_translation_xyz",
    "root_quaternion_xyzw",
    "linear_velocity_xyz",
    "angular_velocity_xyz",
    "com_translation_xyz",
    "joint_positions",
    "joint_velocities",
    "controller_target_positions",
)


@dataclass(frozen=True)
class RecordedActor:
    """A recorded actor and the scene actor that it is restored to."""

    path: str
    """Recorded data path, such as ``data/physics/actors/cube``."""
    actor: sdp.Actor
    is_articulated: bool
    is_static: bool


def _normalized(path: str) -> str:
    return path.lstrip("/")


def _static_values(reader: EpisodeReader, component: str) -> dict[str, Any]:
    result = {}
    for sample in reader.component_samples(component):
        if not isinstance(sample.value, list) or len(sample.value) != 1:
            raise ValueError(f"Expected one value for {component!r}.")
        result[_normalized(sample.entity_path)] = sample.value[0]
    return result


def _scene_actors_by_name(scene: sdp.Scene) -> dict[str, sdp.Actor]:
    actors: dict[str, sdp.Actor] = {}

    def collect(actor: sdp.Actor) -> None:
        if actor.is_nested_link_actor() or actor.is_nested_soft_actor():
            return
        name = actor.get_name()
        if name in actors:
            raise ValueError(f"Restoring requires unique actor names: {name!r}")
        actors[name] = actor

    scene.for_each_actor(collect)
    return actors


def _bind_actors(reader: EpisodeReader, scene: sdp.Scene) -> list[RecordedActor]:
    runtime_names = _static_values(reader, "superdex.physics.runtime_name")
    spawn_names = _static_values(reader, "superdex.physics.spawn_name")
    actor_types = _static_values(reader, "superdex.physics.actor_type")
    static_flags = _static_values(reader, "superdex.physics.is_static")
    if not runtime_names.keys() == actor_types.keys() == static_flags.keys():
        raise ValueError("Recorded actor metadata is incomplete.")

    scene_actors = _scene_actors_by_name(scene)
    bound_paths: dict[sdp.ActorHandle, str] = {}
    actors = []
    for path, runtime_name in sorted(runtime_names.items()):
        spawn_name = spawn_names.get(path)
        # Loading a task names a spawn's bodies "<spawn name>/<name>", and a spawned bot after
        # its spawn.
        names = (
            [runtime_name]
            if spawn_name is None
            else [f"{spawn_name}/{runtime_name}", spawn_name, runtime_name]
        )
        actor = next(
            (scene_actors[name] for name in names if name in scene_actors), None
        )
        if actor is None:
            raise ValueError(f"Scene is missing recorded actor {names[0]!r}.")
        handle = actor.get_handle()
        if handle in bound_paths:
            raise ValueError(
                f"Recorded actors {bound_paths[handle]!r} and {path!r} both bind to "
                f"scene actor {actor.get_name()!r}."
            )
        bound_paths[handle] = path
        expected_type = {
            "rigid": sdp.ActorType.RIGID,
            "articulated": sdp.ActorType.ARTICULATED,
        }.get(actor_types[path])
        if actor.get_type() != expected_type:
            raise ValueError(f"Actor type mismatch for {runtime_name!r}.")
        actors.append(
            RecordedActor(
                path,
                actor,
                expected_type == sdp.ActorType.ARTICULATED,
                bool(static_flags[path]),
            )
        )
    return actors


class EpisodeRestorer:
    """Restores a scene to the state recorded at any frame of an episode.

    The scene must contain the recorded actors, with static actors at their recorded
    poses. An actor recorded with a spawn name binds to the scene actor named
    ``<spawn name>/<runtime name>``, or else ``<spawn name>``, or else its runtime
    name, and other actors bind by runtime name. No two recorded actors may bind to
    the same scene actor, and scene actors that were not recorded must be static or
    nested, such as robot links.

    Restoring a frame sets the scene clock and each dynamic actor's pose, velocity
    and pose-controller target through the public setters, then takes a zero step to
    refresh actor queries. A positive step then continues the episode. The integrator
    starts afresh from the restored state, as at the start of a simulation, so a
    resumed run stays close to the original but doesn't match it bit for bit.

    Restoring a frame also sets each recorded robot's external forces to the joint
    effort recorded for the step after the frame, and clears other dynamic actors'
    external forces. At frames without a recorded effort, such as the last frame, a
    robot keeps its latest earlier effort, because Mochi applies an external force
    until it is replaced. Before its first recorded effort, a robot has none.

    Actors that were spawned or destroyed during the episode are not
    supported.
    """

    def __init__(self, recording: EpisodeReader, scene: sdp.Scene) -> None:
        self._actors = _bind_actors(recording, scene)
        self._validate_unrecorded_actors(scene)
        self._read_frames(recording)
        self._read_efforts(recording)
        self._states = {
            (_normalized(path), component): rows
            for (path, component), rows in recording._component_columns(
                [f"superdex.physics.{component}" for component in _STATE_COMPONENTS]
            ).items()
        }
        self._validate_static_poses()
        self._native = _extension._EpisodeRestorer(
            scene,
            np.asarray(self._times_sec, dtype=np.float64),
            np.asarray(self._step_dts_sec, dtype=np.float64),
            [
                self._native_actor(recorded)
                for recorded in self._actors
                if not recorded.is_static
            ],
        )

    @property
    def frames(self) -> list[int]:
        """Recorded frame numbers, in recording order."""
        return list(self._frames)

    @property
    def actors(self) -> list[RecordedActor]:
        """Recorded actors, sorted by path."""
        return list(self._actors)

    def restore_frame(self, frame: int) -> None:
        """Restore the scene to the end of the step recorded at ``frame``.

        Recorded robots get the joint efforts recorded for the next step.
        """
        index = self._frame_indices.get(frame)
        if index is None:
            raise ValueError(f"Frame {frame} was not recorded.")
        self._native.restore_frame(index)

    def _read_frames(self, recording: EpisodeReader) -> None:
        self._frames: list[int] = []
        self._times_sec: list[float] = []
        self._step_dts_sec: list[float] = []
        for sample in recording.component_samples(
            "superdex.physics.step_dt_sec", entity_path=_FRAME_PATH
        ):
            frame = sample.timelines.get("frame")
            time_ns = sample.timelines.get("sim_time")
            if not isinstance(frame, int) or not isinstance(time_ns, int):
                raise ValueError("Recorded frame timing is missing its timelines.")
            self._frames.append(frame)
            self._times_sec.append(time_ns / 1e9)
            self._step_dts_sec.append(float(sample.value[0]))
        if not self._frames:
            raise ValueError("The recording has no physics frames.")
        self._frame_indices = {frame: i for i, frame in enumerate(self._frames)}

    def _read_efforts(self, recording: EpisodeReader) -> None:
        robot_ids = _static_values(recording, "superdex.robotics.robot_id")
        runtime_names = _static_values(recording, "superdex.physics.runtime_name")
        robots = {
            path: [
                recorded
                for recorded in self._actors
                if recorded.is_articulated and runtime_names[recorded.path] == robot_id
            ]
            for path, robot_id in robot_ids.items()
        }
        self._efforts: dict[str, np.ndarray] = {}
        for (entity_path, _), rows in recording._component_columns(
            ("superdex.robotics.applied_joint_effort",)
        ).items():
            matches = robots.get(_normalized(entity_path).rsplit("/", 1)[0], [])
            if len(matches) != 1:
                raise ValueError(f"{entity_path!r} has no unique recorded robot.")
            frames = rows["frame"].to_numpy()
            unrecorded = ~np.isin(frames, self._frames)
            if unrecorded.any():
                raise ValueError(
                    f"{entity_path!r} has an effort at unrecorded frame "
                    f"{frames[unrecorded][0]}."
                )
            values = pc.list_flatten(rows["value"]).to_numpy().reshape(len(frames), -1)
            # Mochi keeps an external force until it is replaced, so each effort holds until
            # the next one.
            latest = np.searchsorted(frames, self._frames, side="right") - 1
            held = latest >= 0
            efforts = np.zeros((len(self._frames), values.shape[1]))
            efforts[held] = values[latest[held]]
            self._efforts[matches[0].path] = efforts

    def _values(
        self, component: str, recorded: RecordedActor, frames: np.ndarray
    ) -> np.ndarray:
        rows = self._states.get((recorded.path, f"superdex.physics.{component}"))
        recorded_frames = np.zeros(0) if rows is None else rows["frame"].to_numpy()
        missing = ~np.isin(frames, recorded_frames)
        if rows is None or missing.any():
            raise ValueError(
                f"Actor {recorded.actor.get_name()!r} has no {component!r} at "
                f"frame {frames[missing][0]}."
            )
        # Use the last row at each frame.
        indices = np.searchsorted(recorded_frames, frames, side="right") - 1
        return np.asarray(
            pc.list_flatten(rows["value"].take(indices)).to_numpy(), dtype=np.float64
        )

    def _value(self, component: str, recorded: RecordedActor, frame: int) -> Any:
        return self._values(component, recorded, np.array([frame])).tolist()

    def _column(self, component: str, recorded: RecordedActor) -> np.ndarray:
        return self._values(component, recorded, np.asarray(self._frames))

    def _native_actor(self, recorded: RecordedActor) -> Any:
        empty = np.zeros(0)
        rigid = not recorded.is_articulated
        articulated = recorded.is_articulated
        # Targets are recorded at every frame or at none; _column rejects gaps.
        has_targets = articulated and (
            (recorded.path, "superdex.physics.controller_target_positions")
            in self._states
        )
        return _extension._RestoredActor(
            handle=recorded.actor.get_handle(),
            is_articulated=articulated,
            root_rotations=self._column("root_quaternion_xyzw", recorded),
            com_translations=(
                self._column("com_translation_xyz", recorded) if rigid else empty
            ),
            linear_velocities=(
                self._column("linear_velocity_xyz", recorded) if rigid else empty
            ),
            angular_velocities=(
                self._column("angular_velocity_xyz", recorded) if rigid else empty
            ),
            root_translations=(
                self._column("root_translation_xyz", recorded) if articulated else empty
            ),
            joint_positions=(
                self._column("joint_positions", recorded) if articulated else empty
            ),
            joint_velocities=(
                self._column("joint_velocities", recorded) if articulated else empty
            ),
            controller_targets=(
                self._column("controller_target_positions", recorded)
                if has_targets
                else empty
            ),
            joint_efforts=self._efforts.get(recorded.path, empty).reshape(-1),
        )

    def _transform(self, recorded: RecordedActor, frame: int) -> sdp.TransformRT:
        return sdp.TransformRT(
            rotation=self._value("root_quaternion_xyzw", recorded, frame),
            translation=self._value("root_translation_xyz", recorded, frame),
        )

    def _validate_unrecorded_actors(self, scene: sdp.Scene) -> None:
        bound = {recorded.actor.get_handle() for recorded in self._actors}
        unrecorded = sorted(
            name
            for name, actor in _scene_actors_by_name(scene).items()
            if not actor.is_static() and actor.get_handle() not in bound
        )
        if unrecorded:
            raise ValueError(f"Scene has unrecorded dynamic actors {unrecorded}.")

    def _validate_static_poses(self) -> None:
        for recorded in self._actors:
            if not recorded.is_static:
                continue
            pose = recorded.actor.get_root_transform()
            recorded_pose = self._transform(recorded, self._frames[0])
            rotation = pose.rotation.tolist()
            recorded_rotation = recorded_pose.rotation.tolist()
            # q and -q are the same rotation.
            rotation_error = min(
                math.dist(rotation, recorded_rotation),
                math.dist(rotation, [-value for value in recorded_rotation]),
            )
            translation_error = math.dist(
                pose.translation.tolist(), recorded_pose.translation.tolist()
            )
            if max(rotation_error, translation_error) > _STATIC_POSE_TOLERANCE:
                raise ValueError(
                    f"Static actor {recorded.actor.get_name()!r} differs from its "
                    "recorded pose."
                )
