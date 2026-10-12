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

"""Minimal RRD inspection helpers for recorder tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rerun.experimental import RrdReader


@dataclass(frozen=True)
class ComponentSample:
    entity_path: str
    timelines: dict[str, int | float]
    value: Any
    is_static: bool


class EpisodeReader:
    def __init__(self, path: Path) -> None:
        self._reader = RrdReader(path)
        recordings = self._reader.recordings()
        if len(recordings) != 1:
            raise ValueError(f"Expected one recording store, found {len(recordings)}")

    def component_samples(
        self, component: str, *, entity_path: str | None = None
    ) -> list[ComponentSample]:
        stream = self._reader.stream().filter(
            content=entity_path,
            components=component,
        )
        samples: list[ComponentSample] = []
        for chunk in stream:
            batch = chunk.to_record_batch()
            component_index = batch.schema.get_field_index(component)
            if component_index < 0:
                raise ValueError(
                    f"Chunk for {chunk.entity_path!r} is missing {component!r}"
                )
            component_values = batch.column(component_index).to_pylist()
            timeline_values = {
                timeline: batch.column(
                    batch.schema.get_field_index(timeline)
                ).to_pylist()
                for timeline in chunk.timeline_names
                if timeline in {"frame", "sim_time"}
            }
            samples.extend(
                ComponentSample(
                    entity_path=chunk.entity_path,
                    timelines={
                        timeline: values[row]
                        for timeline, values in timeline_values.items()
                    },
                    value=value,
                    is_static=chunk.is_static,
                )
                for row, value in enumerate(component_values)
                if value is not None
            )
        return sorted(
            samples,
            key=lambda sample: (
                sample.timelines.get("frame", -1),
                sample.entity_path,
            ),
        )

    def read_static_value(self, component: str) -> Any:
        samples = self.component_samples(component)
        if len(samples) != 1 or not samples[0].is_static:
            raise ValueError(
                f"Expected exactly one static value for {component!r}; got {samples!r}"
            )
        value = samples[0].value
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        return value
