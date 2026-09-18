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

"""Characterization tests for the public environment registry baseline."""

import unittest

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
from superdex.lab.gym.utils.env_discovery import (
    discover_envs,
    load_entry_config,
    register_all_envs,
)
from test.envs.registry_expectations import (
    EXPECTED_PUBLIC_RECIPE_ASSOCIATIONS,
    EXPECTED_PUBLIC_REGISTRATIONS,
    EXPECTED_PUBLIC_TEST_ONLY_CONFIGURATIONS,
    PUBLIC_ENV_PACKAGE,
    restore_gym_registry,
    serialize_snapshot,
    snapshot_entry,
    snapshot_env_spec,
    snapshot_gym_registry,
    snapshot_test_only_configuration,
)


class TestRegistryBaseline(unittest.TestCase):
    def _register_envs_for_test(self) -> None:
        registry_snapshot = snapshot_gym_registry()
        self.addCleanup(restore_gym_registry, registry_snapshot)
        gym.registry.clear()
        register_all_envs()

    def test_public_registrations_match_baseline(self) -> None:
        entries = discover_envs((PUBLIC_ENV_PACKAGE,))
        actual = tuple(
            snapshot_entry(entry) for entry in entries if not entry.test_only
        )

        self.assertEqual(
            serialize_snapshot(
                tuple(expected["entry"] for expected in EXPECTED_PUBLIC_REGISTRATIONS)
            ),
            serialize_snapshot(actual),
        )

    def test_public_env_specs_match_baseline(self) -> None:
        self._register_envs_for_test()
        expected_ids = {
            expected["entry"]["env_id"] for expected in EXPECTED_PUBLIC_REGISTRATIONS
        }
        actual = tuple(
            snapshot_env_spec(spec)
            for env_id, spec in gym.registry.items()
            if env_id in expected_ids
        )

        self.assertEqual(
            serialize_snapshot(
                tuple(expected["spec"] for expected in EXPECTED_PUBLIC_REGISTRATIONS)
            ),
            serialize_snapshot(actual),
        )

    def test_public_test_only_configurations_match_baseline(self) -> None:
        self._register_envs_for_test()
        entries = discover_envs((PUBLIC_ENV_PACKAGE,))
        test_only = tuple(entry for entry in entries if entry.test_only)

        self.assertEqual(
            serialize_snapshot(EXPECTED_PUBLIC_TEST_ONLY_CONFIGURATIONS),
            serialize_snapshot(
                tuple(snapshot_test_only_configuration(entry) for entry in test_only)
            ),
        )
        for entry in test_only:
            with self.subTest(env_id=entry.env_id):
                self.assertNotIn(entry.env_id, gym.registry)

    def test_public_recipe_associations_match_baseline(self) -> None:
        entries = discover_envs((PUBLIC_ENV_PACKAGE,))
        actual = tuple(
            (entry.env_id, kind)
            for entry in entries
            for kind in ("train", "benchmark")
            if load_entry_config(entry, kind)
        )

        self.assertEqual(EXPECTED_PUBLIC_RECIPE_ASSOCIATIONS, actual)

    def test_incompatible_existing_registration_is_rejected(self) -> None:
        env_id = EXPECTED_PUBLIC_REGISTRATIONS[0]["entry"]["env_id"]
        self._register_envs_for_test()
        conflicting_spec = EnvSpec(
            id=env_id,
            entry_point="deliberately.incompatible:Environment",
            kwargs={"unexpected": True},
            max_episode_steps=7,
        )
        gym.registry[env_id] = conflicting_spec

        with self.assertRaisesRegex(ValueError, rf"{env_id!r}.*entry_point"):
            register_all_envs()

        self.assertIs(conflicting_spec, gym.registry[env_id])


if __name__ == "__main__":
    unittest.main()
