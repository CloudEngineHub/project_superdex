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
import math
import tempfile
import unittest
from collections.abc import Callable, Container
from pathlib import Path
from typing import NamedTuple

import numpy as np
import superdex.physics as sdp
import superdex.recorder as recorder
import superdex.robotics as sdr
from superdex.physics.paths import resolve_asset
from superdex.recorder.reader import EpisodeReader

from . import rrd_verify

BOT = "bots/arm_hand_combos/fr3_dg5f_short/right/fr3_dg5f_short_right.superdex_bot"
METHODS = (
    sdp.IntegrationMethod.BACKWARD_EULER,
    sdp.IntegrationMethod.BDF2,
    sdp.IntegrationMethod.BDF3,
)
NP_REAL = np.float64 if sdp.uses_double_precision() else np.float32
NUM_STEPS = 450
# Variable steps of 1.4 to 2.6 ms in whole 100 ns, like Unreal's.
STEP_DURATIONS = [
    round(2e4 + 6e3 * math.sin(0.37 * step)) / 1e7 for step in range(NUM_STEPS)
]
TABLE_CENTER = (0.59, 0.0, 0.225)
TABLE_HALF_EXTENTS = (0.21, 0.2, 0.025)
CUBE_HALF_SIZE = 0.035
# On the table top.
CUBE_POSITION = (0.52, 0.0, 0.25 + CUBE_HALF_SIZE)
BOX_TRIANGLES = (
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
# The hand flange's target moves through these points [m], reaching each at the given
# time [s]. The hand reaches the cube at about frame 310 and pushes it along x.
EE_TARGET_TIMES = (0.0, 0.25, 0.85)
EE_TARGET_POINTS = ((0.307, 0.0, 0.59), (0.36, 0.0, 0.47), (0.6, 0.0, 0.47))
NUM_ARM_DOFS = 7
# The cube rests on the table, away from the hand, during these steps.
RESTING_STEPS = slice(150, 250)
PUSHING_FRAME = 330


class Recording(NamedTuple):
    path: Path
    efforts: list[np.ndarray]
    """The effort applied during each step."""


def add_box(
    scene: sdp.Scene,
    name: str,
    center: tuple[float, float, float],
    half_extents: tuple[float, float, float],
    is_static: bool = False,
) -> sdp.Actor:
    hx, hy, hz = half_extents
    shape = sdp.create_tri_mesh_shape(
        coordinates=[
            c
            for z in (-hz, hz)
            for y in (-hy, hy)
            for x in (-hx, hx)
            for c in (x, y, z)
        ],
        connectivity=[i for triangle in BOX_TRIANGLES for i in triangle],
    )
    # Without normal damping, BDF2 and BDF3 barely damp the cube's bouncing.
    contact = sdp.ContactParams(
        penalty_coefficient=1e7,
        penalty_threshold_default=0.002,
        viscous_friction_coefficient=2,
        coulomb_friction_coefficient=1,
        normal_viscous_damping_coefficient=10,
    )
    actor = scene.create_rigid_actor(
        name=name,
        shape=shape,
        world_from_local=sdp.TransformRT(translation=center),
        collider_type=sdp.ColliderType.BOX,
        is_static=is_static,
        contact=contact,
        density=500,
    )
    assert actor is not None
    return actor


class PushScene:
    """An effort-driven FR3 arm with a DG5F hand pushes a cube across a table."""

    def __init__(
        self,
        test: unittest.TestCase,
        method: sdp.IntegrationMethod,
        context: sdr.RoboticsContext,
    ) -> None:
        self.scene = sdp.create_scene("push")
        test.addCleanup(sdp.destroy_scene, self.scene)
        solver_params = self.scene.get_solver_params()
        solver_params.integration_method = method
        self.scene.set_solver_params(solver_params)
        self.scene.set_gravity([0, 0, -9.81])
        add_box(self.scene, "table", TABLE_CENTER, TABLE_HALF_EXTENTS, is_static=True)
        self.cube = add_box(self.scene, "cube", CUBE_POSITION, (CUBE_HALF_SIZE,) * 3)

        prefab = sdr.load_bot_prefab_from_file(str(resolve_asset(BOT)))
        # Neither controller compensates for gravity.
        for index in range(len(prefab.links)):
            prefab.links[index].has_gravity = False
        bot = sdr.create_bot(self.scene, prefab, context)
        assert bot is not None
        test.addCleanup(sdr.destroy_bot, self.scene, bot)
        robot = bot.get_articulated_actor()
        osc = bot.create_controller("BASIC_OSC_PD")
        jsc = bot.create_controller("BASIC_JSC_PD")
        assert robot is not None
        assert isinstance(osc, sdr.ControllerBasicOscPd)
        assert isinstance(jsc, sdr.ControllerBasicJscPd)
        self.bot, self.robot, self.osc, self.jsc = bot, robot, osc, jsc
        num_dofs = robot.get_num_dofs()
        self.dofs = list(range(num_dofs))

        osc.initialize(f"{bot.get_name()}/fr3_link0", f"{bot.get_name()}/fr3_link8")
        osc_params = osc.get_params()
        osc_params.kp_p, osc_params.kd_p = 900.0, 75.0
        osc_params.kp_r, osc_params.kd_r = 30.0, 3.0
        osc_params.max_translation_error = 0.05
        osc_params.max_rotation_error = 0.4
        osc_params.b_apply_max_osc_torque_normalization = True
        osc.set_params(osc_params)
        jsc_params = sdr.ControllerBasicJscPdParams()
        jsc_params.kp = np.full(num_dofs, 3.0, dtype=NP_REAL)
        jsc_params.kd = np.full(num_dofs, 0.2, dtype=NP_REAL)
        jsc_params.saturation = np.full(num_dofs, 2.0, dtype=NP_REAL)
        jsc_params.deadband = np.zeros(num_dofs, dtype=NP_REAL)
        jsc.set_params(jsc_params)
        self.hold_pose = self.dof_values(robot.get_articulated_pose)
        observation = osc.get_current_observations_from_mochi()
        self.root_from_world = observation.world_from_root.inverse()
        self.ee_rotation = observation.world_from_ee_link.rotation

    def dof_values(self, get: Callable[[np.ndarray], None]) -> np.ndarray:
        values = np.zeros(len(self.dofs), dtype=NP_REAL)
        get(values)
        return values

    def effort(self, duration: float) -> np.ndarray:
        """The OSC effort on the arm plus the JSC effort on the hand."""
        time_sec = self.scene.get_total_simulation_time()
        world_from_target = sdp.TransformRT(
            rotation=self.ee_rotation,
            translation=[
                float(np.interp(time_sec, EE_TARGET_TIMES, axis))
                for axis in zip(*EE_TARGET_POINTS)
            ],
        )
        arm = self.osc.compute_output(
            self.osc.get_current_observations_from_mochi(),
            sdr.ControllerBasicOscPdTarget(
                root_from_target_ee=self.root_from_world * world_from_target
            ),
        )
        observation = self.jsc.get_current_observations_from_mochi()
        observation.dt = duration
        hand = np.array(
            self.jsc.compute_output(
                observation, sdr.ControllerBasicJscPdTarget(target_pose=self.hold_pose)
            ),
            dtype=NP_REAL,
        )
        hand[:NUM_ARM_DOFS] = 0.0
        return np.array(arm, dtype=NP_REAL) + hand


class ResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        sdp.initialize(num_worker_threads=0)
        self.addCleanup(sdp.shutdown)
        self.addCleanup(gc.collect)
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.context = sdr.create_context()

    def record(
        self,
        method: sdp.IntegrationMethod,
        num_steps: int = NUM_STEPS,
        effort_steps: Container[int] | None = None,
    ) -> Recording:
        source = PushScene(self, method, self.context)
        config = recorder.RecorderConfig(self.directory)
        episode = recorder.Recorder(config).begin_episode()
        physics = recorder.PhysicsRecorder(episode, source.scene)
        controls = recorder.RoboticsRecorder(episode, source.bot)
        physics.record(0, source.scene.get_total_simulation_time())
        recording = Recording(episode.output_path, [])
        applied = np.zeros(len(source.dofs), dtype=NP_REAL)
        for step, duration in enumerate(STEP_DURATIONS[:num_steps]):
            time_sec = source.scene.get_total_simulation_time()
            effort = source.effort(duration)
            if effort_steps is None or step in effort_steps:
                source.robot.set_external_forces_on_dofs(source.dofs, effort)
                applied = effort
            # Like Unreal, record the effort applied for each step at the frame the step
            # starts from.
            controls.record(step, time_sec)
            source.scene.step(duration)
            physics.record(step + 1, source.scene.get_total_simulation_time())
            recording.efforts.append(applied)
        episode.commit()
        rrd_verify.assert_rerun_accepts(episode.output_path)
        return recording

    def restore(
        self, method: sdp.IntegrationMethod, recording: Recording
    ) -> tuple[PushScene, recorder.EpisodeRestorer]:
        restored = PushScene(self, method, self.context)
        reader = EpisodeReader(recording.path)
        return restored, recorder.EpisodeRestorer(reader, restored.scene)

    def test_restoring_applies_the_effort_recorded_for_the_next_step(self) -> None:
        for method in METHODS:
            with self.subTest(method=method.name):
                recording = self.record(method)
                restored, restorer = self.restore(method, recording)
                for frame in (0, RESTING_STEPS.start, PUSHING_FRAME, NUM_STEPS):
                    # A force left from before the restore.
                    restored.cube.set_external_forces_on_dofs([2], [5.0])
                    restorer.restore_frame(frame)
                    # The last frame has no effort row, so the last effort holds.
                    np.testing.assert_array_equal(
                        restored.dof_values(restored.robot.get_external_forces),
                        recording.efforts[min(frame, NUM_STEPS - 1)],
                    )
                    cube_forces = np.ones(6, dtype=NP_REAL)
                    restored.cube.get_external_forces(cube_forces)
                    np.testing.assert_array_equal(cube_forces, 0.0)

    def test_records_and_restores_the_effort_mochi_holds_between_steps(self) -> None:
        method = sdp.IntegrationMethod.BACKWARD_EULER
        recording = self.record(method, num_steps=12, effort_steps=(3, 4, 8))
        restored, restorer = self.restore(method, recording)
        zero = np.zeros_like(recording.efforts[0])
        e3, e4, e8 = (recording.efforts[step] for step in (3, 4, 8))
        expected = [zero, zero, zero, e3, e4, e4, e4, e4, e8, e8, e8, e8, e8]
        for frame, effort in enumerate(expected):
            # A force left from before the restore.
            restored.robot.set_external_forces_on_dofs(
                restored.dofs, [5.0] * len(restored.dofs)
            )
            restorer.restore_frame(frame)
            np.testing.assert_array_equal(
                restored.dof_values(restored.robot.get_external_forces), effort
            )
