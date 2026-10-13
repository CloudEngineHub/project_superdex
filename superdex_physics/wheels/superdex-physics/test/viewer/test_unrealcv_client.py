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

import json
import queue
import threading
import unittest
from collections.abc import Callable
from socket import SHUT_RDWR
from unittest.mock import call, Mock

import numpy as np
from superdex.physics.viewer.unrealcv.unrealcv_client import (
    LitMultiLayout,
    UnrealCVClient,
)


class RecordingUnrealCVClient(UnrealCVClient):
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.batch_commands: list[list[str]] = []
        self.responses: dict[str, str | None] = {}
        self.batch_responses: list[str | None] = []

    def _request(self, command: str, use_async: bool = False) -> str | None:
        self.commands.append(command)
        return self.responses.get(command)

    def _request_batch(
        self, commands: list[str], use_async: bool = False, profile: bool = False
    ) -> list[str | None] | tuple[list[str | None], dict]:
        self.batch_commands.append(commands)
        if profile:
            return self.batch_responses, {}
        return self.batch_responses


class _FailingBatchClient:
    """A connected unrealcv ``Client`` stand-in whose batch write fails mid-send."""

    def __init__(self) -> None:
        self.sock: Mock | None = Mock()
        self.batches: list[list[str]] = []
        self.recv_num_q: queue.SimpleQueue[int | None] = queue.SimpleQueue()
        self.before_first_failure: Callable[[], None] | None = None

    def isconnected(self) -> bool:
        return self.sock is not None

    def request_batch(self, commands: list[str]) -> list[str]:
        self.batches.append(commands)
        hook, self.before_first_failure = self.before_first_failure, None
        if hook is not None:
            hook()
        raise ConnectionResetError("Connection reset by peer")

    def request_async(self, command: str) -> None:
        self.batches.append([command])


class _ContendedLock:
    """A request lock that flags when a second caller starts waiting for it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.contended = threading.Event()

    def __enter__(self) -> None:
        if not self._lock.acquire(blocking=False):
            self.contended.set()
            self._lock.acquire()

    def __exit__(self, *exc_info: object) -> None:
        self._lock.release()


class TestUnrealCVClientFailedBatch(unittest.TestCase):
    def test_a_failed_batch_retires_the_connection_before_the_next_request(
        self,
    ) -> None:
        client = UnrealCVClient(request_timeout=0)
        failing = _FailingBatchClient()
        socket = failing.sock
        client._client = failing  # pyre-ignore[8]
        client._connected = True

        self.assertEqual([None, None], client._request_batch(["vget /a", "vget /b"]))
        self.assertFalse(client.is_connected())
        # Shut down before closing, so a receive thread blocked in recv() wakes now.
        self.assertEqual([call.shutdown(SHUT_RDWR), call.close()], socket.method_calls)
        # The receive thread gets its quit signal, as on a disconnect.
        self.assertIsNone(failing.recv_num_q.get_nowait())
        # The next request short-circuits instead of writing to the broken socket.
        self.assertEqual([None], client._request_batch(["vget /c"]))
        self.assertEqual([["vget /a", "vget /b"]], failing.batches)

    def _fail_a_batch_while_another_request_waits(
        self, waiting_request: Callable[[UnrealCVClient], object]
    ) -> tuple[list[object], list[list[str]]]:
        client = UnrealCVClient(request_timeout=0)
        failing = _FailingBatchClient()
        lock = _ContendedLock()
        client._client = failing  # pyre-ignore[8]
        client._connected = True
        client._request_lock = lock  # pyre-ignore[8]
        waited: list[object] = []
        waiter = threading.Thread(
            target=lambda: waited.append(waiting_request(client)), daemon=True
        )

        def start_the_waiter() -> None:
            waiter.start()
            lock.contended.wait(timeout=10)

        failing.before_first_failure = start_the_waiter
        self.assertEqual([None], client._request_batch(["vget /a"]))
        waiter.join(timeout=10)
        # The second request was queued on the lock while the batch failed.
        self.assertTrue(lock.contended.is_set())
        return waited, failing.batches

    def test_a_batch_waiting_on_a_failing_batch_never_reaches_the_socket(
        self,
    ) -> None:
        waited, sent = self._fail_a_batch_while_another_request_waits(
            lambda client: client._request_batch(["vget /b"])
        )

        self.assertEqual([[None]], waited)
        self.assertEqual([["vget /a"]], sent)

    def test_an_async_request_waits_for_a_batch_and_then_short_circuits(
        self,
    ) -> None:
        waited, sent = self._fail_a_batch_while_another_request_waits(
            lambda client: client._request("vset /action/scene/reset", use_async=True)
        )

        self.assertEqual([None], waited)
        self.assertEqual([["vget /a"]], sent)


class TestUnrealCVClientCameraCalibration(unittest.TestCase):
    def test_builds_exact_camera_calibration_commands(self) -> None:
        self.assertEqual(
            UnrealCVClient.build_camera_local_location_command(
                3, np.array([1.25, -2.5, 0.0])
            ),
            "vset /camera/3/local_location 1.25 -2.5 0",
        )
        self.assertEqual(
            UnrealCVClient.build_camera_local_rotation_command(
                3, np.array([10.0, -20.5, 30.25])
            ),
            "vset /camera/3/local_rotation 10 -20.5 30.25",
        )
        self.assertEqual(
            UnrealCVClient.build_camera_filmback_command(3, np.array([6.4, 4.8])),
            "vset /camera/3/filmback 6.4 4.8",
        )
        self.assertEqual(
            UnrealCVClient.build_camera_focal_length_command(3, 1.88),
            "vset /camera/3/focal_length 1.88",
        )

    def test_gets_typed_camera_calibration_values(self) -> None:
        client = RecordingUnrealCVClient()
        client.responses = {
            "vget /camera/4/local_location": "1.0 -2.5 3",
            "vget /camera/4/local_rotation": "10 20.5 -30",
            "vget /camera/4/filmback": "6.4 4.8",
            "vget /camera/4/focal_length": "1.88",
        }

        np.testing.assert_allclose(
            client.get_camera_local_location(4), np.array([1.0, -2.5, 3.0])
        )
        np.testing.assert_allclose(
            client.get_camera_local_rotation(4), np.array([10.0, 20.5, -30.0])
        )
        np.testing.assert_allclose(client.get_camera_filmback(4), np.array([6.4, 4.8]))
        self.assertEqual(client.get_camera_focal_length(4), 1.88)

    def test_gets_return_none_for_failure_responses(self) -> None:
        client = RecordingUnrealCVClient()
        client.responses = {
            "vget /camera/5/local_location": "error missing camera",
            "vget /camera/5/local_rotation": "10 20",
            "vget /camera/5/filmback": None,
            "vget /camera/5/focal_length": "not-a-float",
        }

        self.assertIsNone(client.get_camera_local_location(5))
        self.assertIsNone(client.get_camera_local_rotation(5))
        self.assertIsNone(client.get_camera_filmback(5))
        self.assertIsNone(client.get_camera_focal_length(5))

    def test_sets_return_false_for_failure_responses(self) -> None:
        client = RecordingUnrealCVClient()
        client.responses = {
            "vset /camera/6/local_location 1 2 3": "ok",
            "vset /camera/6/local_rotation 4 5 6": "error invalid rotation",
            "vset /camera/6/filmback 6.4 4.8": None,
            "vset /camera/6/focal_length 1.88": "OK",
        }

        self.assertTrue(client.set_camera_local_location(6, np.array([1.0, 2.0, 3.0])))
        self.assertFalse(client.set_camera_local_rotation(6, np.array([4.0, 5.0, 6.0])))
        self.assertFalse(client.set_camera_filmback(6, np.array([6.4, 4.8])))
        self.assertTrue(client.set_camera_focal_length(6, 1.88))

    def test_apply_camera_calibration_skips_omitted_fields(self) -> None:
        client = RecordingUnrealCVClient()
        client.batch_responses = ["ok", "ok"]

        result = client.apply_camera_calibration(
            7,
            local_location=np.array([1.0, 2.0, 3.0]),
            focal_length=1.88,
        )

        self.assertEqual(result, [True, True])
        self.assertEqual(
            client.batch_commands,
            [
                [
                    "vset /camera/7/local_location 1 2 3",
                    "vset /camera/7/focal_length 1.88",
                ]
            ],
        )

    def test_apply_camera_calibration_uses_deterministic_full_order(self) -> None:
        client = RecordingUnrealCVClient()
        client.batch_responses = ["ok", "error bad rotation", None, "OK"]

        result = client.apply_camera_calibration(
            8,
            focal_length=1.88,
            filmback=np.array([6.4, 4.8]),
            local_rotation=np.array([4.0, 5.0, 6.0]),
            local_location=np.array([1.0, 2.0, 3.0]),
        )

        self.assertEqual(result, [True, False, False, True])
        self.assertEqual(
            client.batch_commands,
            [
                [
                    "vset /camera/8/local_location 1 2 3",
                    "vset /camera/8/local_rotation 4 5 6",
                    "vset /camera/8/filmback 6.4 4.8",
                    "vset /camera/8/focal_length 1.88",
                ]
            ],
        )

    def test_apply_camera_calibration_without_fields_does_not_send_batch(self) -> None:
        client = RecordingUnrealCVClient()

        self.assertEqual(client.apply_camera_calibration(9), [])
        self.assertEqual(client.batch_commands, [])


_LAYOUT_COMMAND = "vget /cameras/lit_layout_multi 2,0"


def _layout_reply(**overrides: object) -> str:
    layout = {
        "width": 721,
        "height": 480,
        "format": "BGRA8",
        "ids": [2, 0],
        "tiles": [[0, 0, 640, 480], [640, 0, 81, 64]],
    }
    layout.update(overrides)
    return json.dumps(layout)


class TestUnrealCVClientMultiViewCapture(unittest.TestCase):
    def test_builds_exact_multi_view_capture_commands(self) -> None:
        self.assertEqual(
            UnrealCVClient.build_start_lit_async_multi_command([2, 0, 5]),
            "vget /cameras/lit_async_multi 2,0,5",
        )
        self.assertEqual(
            UnrealCVClient.build_get_lit_latest_multi_command((2, 0, 5)),
            "vget /cameras/lit_get_latest_multi 2,0,5 npy",
        )

    def test_parses_the_layout_reply(self) -> None:
        client = RecordingUnrealCVClient()
        client.responses = {_LAYOUT_COMMAND: _layout_reply()}

        self.assertEqual(
            client.get_lit_layout_multi([2, 0]),
            LitMultiLayout(
                width=721,
                height=480,
                ids=(2, 0),
                tiles=((0, 0, 640, 480), (640, 0, 81, 64)),
            ),
        )

    def test_rejects_failed_and_unexpected_layout_replies(self) -> None:
        missing_tiles = json.loads(_layout_reply())
        del missing_tiles["tiles"]
        replies: dict[str, str | None] = {
            "error reply": "error Camera 9 not found",
            "no reply": None,
            "bad json": "{width: 721",
            "missing key": json.dumps(missing_tiles),
            "not bgra8": _layout_reply(format="RGBA8"),
            "reordered ids": _layout_reply(ids=[0, 2]),
            "other ids": _layout_reply(ids=[2, 1]),
            "missing tile": _layout_reply(tiles=[[0, 0, 640, 480]]),
            "malformed tile": _layout_reply(tiles=[[0, 0, 640], [640, 0, 81, 64]]),
            "tile past the atlas": _layout_reply(
                tiles=[[0, 0, 640, 480], [641, 0, 81, 64]]
            ),
            "negative origin": _layout_reply(
                tiles=[[0, -1, 640, 480], [640, 0, 81, 64]]
            ),
            "empty tile": _layout_reply(tiles=[[0, 0, 640, 480], [640, 0, 0, 64]]),
        }
        for case, reply in replies.items():
            with self.subTest(case=case):
                client = RecordingUnrealCVClient()
                client.responses = {_LAYOUT_COMMAND: reply}

                with self.assertRaisesRegex(RuntimeError, "lit_layout_multi 2,0"):
                    client.get_lit_layout_multi([2, 0])
                with self.assertRaises(RuntimeError):
                    client.get_lit_layout_multi([2, 0])
                self.assertEqual(client.commands, [_LAYOUT_COMMAND] * 2)

    def test_rejects_a_repeated_camera_before_querying_ue(self) -> None:
        client = RecordingUnrealCVClient()

        with self.assertRaisesRegex(ValueError, "repeats a camera"):
            client.get_lit_layout_multi([2, 0, 2])
        self.assertEqual(client.commands, [])

    def test_caches_the_layout_until_a_camera_size_or_crop_changes(self) -> None:
        client = RecordingUnrealCVClient()
        client.responses = {_LAYOUT_COMMAND: _layout_reply()}

        def layout_queries() -> int:
            return client.commands.count(_LAYOUT_COMMAND)

        for _ in range(3):
            client.get_lit_layout_multi((2, 0))
        self.assertEqual(layout_queries(), 1)

        client.set_camera_size(2, 640, 480)
        client.get_lit_layout_multi((2, 0))
        client.get_lit_layout_multi((2, 0))
        self.assertEqual(layout_queries(), 2)

        client.set_all_camera_sizes([0], 81, 64)
        client.get_lit_layout_multi((2, 0))
        self.assertEqual(layout_queries(), 3)

        client.set_camera_crop(2, 0.0, 0.0, 0.5, 0.5)
        client.get_lit_layout_multi((2, 0))
        self.assertEqual(layout_queries(), 4)


if __name__ == "__main__":
    unittest.main()
