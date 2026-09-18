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

"""Test-only snapshots of the public environment registry baseline."""

import json
from typing import Any

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
from superdex.lab.gym.utils.env_discovery import EnvEntry

PUBLIC_ENV_PACKAGE = "superdex.lab.gym.envs.benchmarks"


def _registration(
    env_id: str,
    short_name: str,
    entry_point: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    name = env_id.removeprefix("superdex_gym/").removesuffix("-v0")
    return {
        "entry": {
            "env_id": env_id,
            "short_name": short_name,
            "entry_point": entry_point,
            "cfg_kwargs": cfg,
        },
        "spec": {
            "id": env_id,
            "entry_point": entry_point,
            "reward_threshold": None,
            "nondeterministic": False,
            "max_episode_steps": None,
            "order_enforce": True,
            "disable_env_checker": False,
            "kwargs": {"cfg": cfg},
            "namespace": "superdex_gym",
            "name": name,
            "version": 0,
            "additional_wrappers": (),
            "vector_entry_point": None,
            "autoreset": "<absent>",
        },
    }


EXPECTED_PUBLIC_REGISTRATIONS = (
    _registration(
        "superdex_gym/Ant-v0",
        "ant",
        "superdex.lab.gym.envs.benchmarks.ant_env:AntEnv",
        {},
    ),
    _registration(
        "superdex_gym/AntFullObservation-v0",
        "ant_full_observation",
        "superdex.lab.gym.envs.benchmarks.ant_env:AntEnv",
        {"exclude_current_positions_from_observation": False},
    ),
    _registration(
        "superdex_gym/AntNoContact-v0",
        "ant_no_contact",
        "superdex.lab.gym.envs.benchmarks.ant_env:AntEnv",
        {"include_contact_in_observation": False},
    ),
    _registration(
        "superdex_gym/AntRotationVector-v0",
        "ant_rotation_vector",
        "superdex.lab.gym.envs.benchmarks.ant_env:AntEnv",
        {"use_rotation_vector": True},
    ),
    _registration(
        "superdex_gym/CartPole-v0",
        "cart_pole",
        "superdex.lab.gym.envs.benchmarks.cartpole_env:CartPoleEnv",
        {},
    ),
    _registration(
        "superdex_gym/CartPoleActuateOnPole-v0",
        "cart_pole_actuate_on_pole",
        "superdex.lab.gym.envs.benchmarks.cartpole_env:CartPoleEnv",
        {"actuate_on_pole": True},
    ),
    _registration(
        "superdex_gym/HalfCheetah-v0",
        "half_cheetah",
        "superdex.lab.gym.envs.benchmarks.halfcheetah_env:HalfCheetahEnv",
        {},
    ),
    _registration(
        "superdex_gym/HalfCheetahFullObservation-v0",
        "half_cheetah_full_observation",
        "superdex.lab.gym.envs.benchmarks.halfcheetah_env:HalfCheetahEnv",
        {"exclude_current_position_from_observation": False},
    ),
)

EXPECTED_PUBLIC_TEST_ONLY_CONFIGURATIONS = (
    {
        "env_id": "superdex_gym/AntTestNoDynamics-v0",
        "short_name": "ant_test_no_dynamics",
        "entry_point": "superdex.lab.gym.envs.benchmarks.ant_env:AntEnv",
        "description": (
            "Test-only: Ant stripped of damping and gravity on a low-friction ground. "
            "A degenerate setup that is not a trainable task, kept as a crash check for "
            "the non-default physics paths."
        ),
        "env_cfg": {
            "use_damping": False,
            "use_gravity": False,
            "use_low_friction": True,
        },
    },
    {
        "env_id": "superdex_gym/CartPoleTestDampedFreePole-v0",
        "short_name": "cart_pole_test_damped_free_pole",
        "entry_point": "superdex.lab.gym.envs.benchmarks.cartpole_env:CartPoleEnv",
        "description": (
            "Test-only: CartPole without gravity and with the pole joint limits removed. "
            "A degenerate setup that is not a trainable task, kept as a crash check for "
            "the non-default physics paths."
        ),
        "env_cfg": {
            "use_damping": True,
            "use_gravity": False,
            "free_pole": True,
        },
    },
    {
        "env_id": "superdex_gym/HalfCheetahTestNoGravityNoSprings-v0",
        "short_name": "half_cheetah_test_no_gravity_no_springs",
        "entry_point": "superdex.lab.gym.envs.benchmarks.halfcheetah_env:HalfCheetahEnv",
        "description": (
            "Test-only: HalfCheetah without gravity and without joint rest springs. A "
            "degenerate setup that is not a trainable task, kept as a crash check for "
            "the non-default physics paths."
        ),
        "env_cfg": {"use_gravity": False, "use_rest_springs": False},
    },
)

EXPECTED_PUBLIC_RECIPE_ASSOCIATIONS = (
    ("superdex_gym/AntNoContact-v0", "train"),
    ("superdex_gym/CartPole-v0", "train"),
    ("superdex_gym/HalfCheetah-v0", "train"),
)


def snapshot_gym_registry() -> tuple[tuple[str, EnvSpec], ...]:
    return tuple(gym.registry.items())


def restore_gym_registry(snapshot: tuple[tuple[str, EnvSpec], ...]) -> None:
    gym.registry.clear()
    gym.registry.update(snapshot)


def serialize_snapshot(snapshot: Any) -> str:
    return json.dumps(snapshot, sort_keys=True, separators=(",", ":"))


def normalize_entry_point(entry_point: Any) -> str | None:
    if entry_point is None or isinstance(entry_point, str):
        return entry_point
    return f"{entry_point.__module__}:{entry_point.__name__}"


def snapshot_entry(entry: EnvEntry) -> dict[str, Any]:
    return {
        "env_id": entry.env_id,
        "short_name": entry.short_name,
        "entry_point": normalize_entry_point(entry.env_cls),
        "cfg_kwargs": entry.cfg_kwargs,
    }


def snapshot_test_only_configuration(entry: EnvEntry) -> dict[str, Any]:
    return {
        "env_id": entry.env_id,
        "short_name": entry.short_name,
        "entry_point": normalize_entry_point(entry.env_cls),
        "description": entry.description,
        "env_cfg": entry.cfg_kwargs,
    }


def snapshot_env_spec(spec: EnvSpec) -> dict[str, Any]:
    return {
        "id": spec.id,
        "entry_point": normalize_entry_point(spec.entry_point),
        "reward_threshold": spec.reward_threshold,
        "nondeterministic": spec.nondeterministic,
        "max_episode_steps": spec.max_episode_steps,
        "order_enforce": spec.order_enforce,
        "disable_env_checker": spec.disable_env_checker,
        "kwargs": spec.kwargs,
        "namespace": spec.namespace,
        "name": spec.name,
        "version": spec.version,
        "additional_wrappers": spec.additional_wrappers,
        "vector_entry_point": spec.vector_entry_point,
        "autoreset": getattr(spec, "autoreset", "<absent>"),
    }
