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

"""Tests for explicit public Gymnasium registration."""

from __future__ import annotations

import copy
import dataclasses
import importlib
import sys
import unittest
from typing import Any

import gymnasium as gym
import superdex.lab.gym as superdex_gym
import superdex.lab.gym.registration as registration_module
from gymnasium.envs.registration import EnvSpec, namespace, WrapperSpec
from superdex.lab.gym.registration import (
    get_env_specs,
    register_env_spec,
    register_envs,
)
from superdex.lab.gym.utils.env_discovery import register_all_envs
from test.envs.registry_expectations import (
    EXPECTED_PUBLIC_REGISTRATIONS,
    restore_gym_registry,
    serialize_snapshot,
    snapshot_env_spec,
    snapshot_gym_registry,
)

_EXPECTED_PUBLIC_IDS = tuple(
    expected["entry"]["env_id"] for expected in EXPECTED_PUBLIC_REGISTRATIONS
)
_CONCRETE_ENV_MODULES = (
    "superdex.lab.gym.envs.benchmarks.ant_env",
    "superdex.lab.gym.envs.benchmarks.cartpole_env",
    "superdex.lab.gym.envs.benchmarks.halfcheetah_env",
)
_EXPECTED_EXPORTS = (
    "VALID_RENDER_MODES",
    "MochiEnv",
    "MochiEnvCfg",
    "RenderMode",
    "Action",
    "ActionSpace",
    "ActionSpaceStructure",
    "Info",
    "Observation",
    "ObservationSpace",
    "ObservationSpaceStructure",
    "ResetResult",
    "RewardTerms",
    "StepResult",
    "StructuredAction",
    "StructuredObservation",
    "StructuredStepResult",
    "envs",
)


@dataclasses.dataclass
class _TypedCartPoleCfg:
    control_frequency: int = 25
    simulation_frequency: int = 50
    actuate_on_pole: bool = True
    render_mode: str | None = "human"


class RegistrationTest(unittest.TestCase):
    def setUp(self) -> None:
        registry_snapshot = snapshot_gym_registry()
        self.addCleanup(restore_gym_registry, registry_snapshot)

    def _clear_registry(self) -> None:
        gym.registry.clear()

    def test_public_specs_match_frozen_snapshot(self) -> None:
        self._clear_registry()
        register_envs()

        actual = tuple(
            snapshot_env_spec(gym.spec(env_id)) for env_id in _EXPECTED_PUBLIC_IDS
        )
        expected = tuple(
            registration["spec"] for registration in EXPECTED_PUBLIC_REGISTRATIONS
        )
        self.assertEqual(serialize_snapshot(expected), serialize_snapshot(actual))
        self.assertEqual(_EXPECTED_PUBLIC_IDS, tuple(gym.registry))

    def test_package_import_preserves_exports_and_registers_idempotently(self) -> None:
        self._clear_registry()

        first_import = importlib.reload(superdex_gym)
        first_specs = {env_id: gym.spec(env_id) for env_id in _EXPECTED_PUBLIC_IDS}
        second_import = importlib.reload(first_import)

        self.assertIs(first_import, second_import)
        self.assertEqual(list(_EXPECTED_EXPORTS), second_import.__all__)
        self.assertEqual(_EXPECTED_PUBLIC_IDS, tuple(gym.registry))
        for env_id, first_spec in first_specs.items():
            with self.subTest(env_id=env_id):
                self.assertIs(first_spec, gym.spec(env_id))

    def test_package_import_keeps_concrete_entry_points_lazy(self) -> None:
        self._clear_registry()
        removed_modules = {
            module_name: sys.modules.pop(module_name)
            for module_name in _CONCRETE_ENV_MODULES
            if module_name in sys.modules
        }
        self.addCleanup(sys.modules.update, removed_modules)

        importlib.reload(registration_module)
        importlib.reload(superdex_gym)

        for module_name in _CONCRETE_ENV_MODULES:
            with self.subTest(module_name=module_name):
                self.assertNotIn(module_name, sys.modules)
        for spec in get_env_specs():
            with self.subTest(env_id=spec.id):
                self.assertEqual(
                    "superdex.lab.gym.registration:make_superdex_env",
                    spec.entry_point,
                )
                self.assertIsInstance(spec.kwargs["env_cls"], str)

    def test_gym_make_routes_environment_kwargs_through_factory(self) -> None:
        self._clear_registry()
        register_envs()

        env = gym.make(
            "superdex_gym/CartPoleActuateOnPole-v0",
            render_mode=None,
        )
        try:
            self.assertTrue(env.unwrapped.actuate_on_pole)
            self.assertIsNone(env.unwrapped.render_mode)
        finally:
            env.close()

        env = gym.make(
            "superdex_gym/CartPoleActuateOnPole-v0",
            actuate_on_pole=False,
        )
        try:
            self.assertFalse(env.unwrapped.actuate_on_pole)
        finally:
            env.close()

    def test_gym_make_merges_environment_kwargs_into_typed_cfg(self) -> None:
        self._clear_registry()
        register_envs()
        cfg = _TypedCartPoleCfg()

        env = gym.make(
            "superdex_gym/CartPole-v0",
            cfg=cfg,
            render_mode=None,
        )
        try:
            self.assertTrue(env.unwrapped.actuate_on_pole)
            self.assertIsNone(env.unwrapped.render_mode)
            self.assertEqual("human", cfg.render_mode)
        finally:
            env.close()

    def test_factory_specs_support_gymnasium_json_round_trip(self) -> None:
        self._clear_registry()
        register_envs()

        for spec in get_env_specs():
            with self.subTest(env_id=spec.id):
                restored = EnvSpec.from_json(spec.to_json())
                self.assertEqual(spec, restored)

    def test_registration_ignores_and_restores_ambient_namespace(self) -> None:
        self._clear_registry()

        with namespace("ambient"):
            register_envs()
            gym.register(
                id="Probe-v0",
                entry_point="example.module:Environment",
            )

        self.assertEqual(
            (*_EXPECTED_PUBLIC_IDS, "ambient/Probe-v0"),
            tuple(gym.registry),
        )

    def test_registration_restores_ambient_namespace_after_failure(self) -> None:
        self._clear_registry()
        gym.registry["superdex_gym/Conflict"] = EnvSpec(
            id="superdex_gym/Conflict",
            entry_point="example.module:Environment",
        )
        versioned_spec = EnvSpec(
            id="superdex_gym/Conflict-v0",
            entry_point="example.module:Environment",
        )

        with namespace("ambient"):
            with self.assertRaises(gym.error.RegistrationError):
                register_env_spec(versioned_spec)
            gym.register(
                id="Probe-v0",
                entry_point="example.module:Environment",
            )

        self.assertIn("ambient/Probe-v0", gym.registry)
        self.assertNotIn("Probe-v0", gym.registry)

    def test_equivalent_pre_registration_is_preserved(self) -> None:
        desired_specs = {
            env_id: copy.deepcopy(gym.spec(env_id)) for env_id in _EXPECTED_PUBLIC_IDS
        }
        self._clear_registry()
        gym.registry.update(desired_specs)

        register_envs()

        for env_id, desired_spec in desired_specs.items():
            with self.subTest(env_id=env_id):
                self.assertIs(desired_spec, gym.spec(env_id))

    def test_every_env_spec_field_participates_in_collision_check(self) -> None:
        desired_specs = {
            env_id: copy.deepcopy(gym.spec(env_id)) for env_id in _EXPECTED_PUBLIC_IDS
        }
        target_id = _EXPECTED_PUBLIC_IDS[0]
        conflicting_values: dict[str, Any] = {
            "id": "different_namespace/Different-v9",
            "entry_point": "different.module:Environment",
            "reward_threshold": 1.0,
            "nondeterministic": True,
            "max_episode_steps": 7,
            "order_enforce": False,
            "disable_env_checker": True,
            "kwargs": {"cfg": {"different": True}},
            "namespace": "different_namespace",
            "name": "Different",
            "version": 9,
            "additional_wrappers": (
                WrapperSpec(
                    name="DifferentWrapper",
                    entry_point="different.module:Wrapper",
                    kwargs={"nested": {"value": 1}},
                ),
            ),
            "vector_entry_point": "different.module:VectorEnvironment",
        }
        self.assertEqual(
            {field.name for field in dataclasses.fields(EnvSpec)},
            set(conflicting_values),
        )

        for field_name, conflicting_value in conflicting_values.items():
            with self.subTest(field_name=field_name):
                self._clear_registry()
                gym.registry.update(copy.deepcopy(desired_specs))
                conflicting_spec = gym.registry[target_id]
                setattr(conflicting_spec, field_name, conflicting_value)

                with self.assertRaisesRegex(
                    ValueError,
                    rf"{target_id!r}.*{field_name}",
                ):
                    register_envs()
                self.assertIs(conflicting_spec, gym.registry[target_id])

    def test_registration_deep_copies_nested_kwargs(self) -> None:
        source_cfg = {"nested": {"values": [1, 2]}}
        spec = EnvSpec(
            id="superdex_gym/NestedCopyTest-v0",
            entry_point="example.module:Environment",
            kwargs={"cfg": source_cfg},
        )
        self._clear_registry()

        register_env_spec(spec)
        registered_cfg = gym.spec(spec.id).kwargs["cfg"]
        source_cfg["nested"]["values"].append(3)
        registered_cfg["nested"]["values"].append(4)

        self.assertEqual({"nested": {"values": [1, 2, 3]}}, source_cfg)
        self.assertEqual({"nested": {"values": [1, 2, 4]}}, registered_cfg)
        self.assertEqual({"nested": {"values": [1, 2, 3]}}, spec.kwargs["cfg"])

    def test_all_version_enumeration_is_exact_and_deterministic(self) -> None:
        specs = (
            EnvSpec("superdex_gym/Example-v2", "example.module:Environment"),
            EnvSpec("other/Example-v0", "example.module:Environment"),
            EnvSpec("superdex_gym/Another-v0", "example.module:Environment"),
            EnvSpec("superdex_gym/Example-v0", "example.module:Environment"),
            EnvSpec("superdex_gym/Example-v1", "example.module:Environment"),
        )
        self._clear_registry()
        for spec in specs:
            gym.registry[spec.id] = spec

        actual = get_env_specs()

        self.assertEqual(
            (
                "superdex_gym/Another-v0",
                "superdex_gym/Example-v0",
                "superdex_gym/Example-v1",
                "superdex_gym/Example-v2",
            ),
            tuple(spec.id for spec in actual),
        )
        self.assertEqual(
            (specs[2], specs[3], specs[4], specs[0]),
            actual,
        )

    def test_legacy_discovery_keeps_explicit_specs(self) -> None:
        self._clear_registry()
        register_envs()
        explicit_specs = {env_id: gym.spec(env_id) for env_id in _EXPECTED_PUBLIC_IDS}

        entries = register_all_envs()

        discovered_ids = {entry.env_id for entry in entries if not entry.test_only}
        self.assertTrue(set(_EXPECTED_PUBLIC_IDS).issubset(discovered_ids))
        for env_id, explicit_spec in explicit_specs.items():
            with self.subTest(env_id=env_id):
                self.assertIs(explicit_spec, gym.spec(env_id))
                self.assertIsInstance(gym.spec(env_id).entry_point, str)

    def test_all_public_ids_support_spec_and_make(self) -> None:
        self._clear_registry()
        register_envs()

        for env_id in _EXPECTED_PUBLIC_IDS:
            with self.subTest(env_id=env_id):
                self.assertEqual(env_id, gym.spec(env_id).id)
                env = gym.make(env_id)
                try:
                    self.assertEqual(env_id, env.spec.id)
                finally:
                    env.close()
