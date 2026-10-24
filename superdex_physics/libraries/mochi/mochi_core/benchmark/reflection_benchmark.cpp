/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "config.h"

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/reflection.h>

#include <cstdint>
#include <vector>

#if MOCHI_USE_REFLECTION

namespace mochi_benchmark {

// Binary serialization of arrays through reflection, as used to capture and restore scene state.
// The argument is the payload size in bytes, so that arrays of real and Real3 hold the same data.
// std::vector uses the generic per-element path of SReflect::ArrayTypeInfo.

template <class ArrayT>
static ArrayT MakeArray(benchmark::State const& state) {
  using T = typename ArrayT::value_type;
  return ArrayT(static_cast<size_t>(state.range(0)) / sizeof(T), T{});
}

template <class ArrayT>
static void SerializeBytes(benchmark::State& state) {
  auto const arr = MakeArray<ArrayT>(state);
  mochi::DynamicArray<uint8_t> buffer(SReflect::ToBytes(arr).size());
  for (auto _ : state) {
    SReflect::SpanStreamWriter writer{SReflect::Span<uint8_t>{buffer}};
    bool const success = SReflect::ToBytes(arr, writer);
    benchmark::DoNotOptimize(success);
    benchmark::ClobberMemory();
  }
}

template <class ArrayT>
static void DeserializeBytes(benchmark::State& state) {
  std::vector<uint8_t> const bytes = SReflect::ToBytes(MakeArray<ArrayT>(state));
  ArrayT arr;
  for (auto _ : state) {
    SReflect::SpanStreamReader reader{SReflect::Span<uint8_t const>{bytes}};
    bool const success = SReflect::FromBytes(reader, arr);
    benchmark::DoNotOptimize(success);
    benchmark::ClobberMemory();
  }
}

static void PayloadBytes(benchmark::internal::Benchmark* b) {
  b->ArgName("bytes")->Arg(1200)->Arg(120000)->Arg(12000000);
}

BENCHMARK_TEMPLATE(SerializeBytes, mochi::DynamicArray<int32_t>)
    ->Name("Reflection/SerializeBytes/DynamicArrayInt32")
    ->Apply(PayloadBytes);
BENCHMARK_TEMPLATE(SerializeBytes, std::vector<int32_t>)
    ->Name("Reflection/SerializeBytes/VectorInt32")
    ->Apply(PayloadBytes);
BENCHMARK_TEMPLATE(SerializeBytes, mochi::DynamicArray<mochi::real>)
    ->Name("Reflection/SerializeBytes/DynamicArrayReal")
    ->Apply(PayloadBytes);
BENCHMARK_TEMPLATE(SerializeBytes, std::vector<mochi::real>)
    ->Name("Reflection/SerializeBytes/VectorReal")
    ->Apply(PayloadBytes);
BENCHMARK_TEMPLATE(SerializeBytes, mochi::DynamicArray<mochi::Real3>)
    ->Name("Reflection/SerializeBytes/DynamicArrayReal3")
    ->Apply(PayloadBytes);
BENCHMARK_TEMPLATE(SerializeBytes, std::vector<mochi::Real3>)
    ->Name("Reflection/SerializeBytes/VectorReal3")
    ->Apply(PayloadBytes);

BENCHMARK_TEMPLATE(DeserializeBytes, mochi::DynamicArray<int32_t>)
    ->Name("Reflection/DeserializeBytes/DynamicArrayInt32")
    ->Apply(PayloadBytes);
BENCHMARK_TEMPLATE(DeserializeBytes, std::vector<int32_t>)
    ->Name("Reflection/DeserializeBytes/VectorInt32")
    ->Apply(PayloadBytes);
BENCHMARK_TEMPLATE(DeserializeBytes, mochi::DynamicArray<mochi::real>)
    ->Name("Reflection/DeserializeBytes/DynamicArrayReal")
    ->Apply(PayloadBytes);
BENCHMARK_TEMPLATE(DeserializeBytes, std::vector<mochi::real>)
    ->Name("Reflection/DeserializeBytes/VectorReal")
    ->Apply(PayloadBytes);
BENCHMARK_TEMPLATE(DeserializeBytes, mochi::DynamicArray<mochi::Real3>)
    ->Name("Reflection/DeserializeBytes/DynamicArrayReal3")
    ->Apply(PayloadBytes);
BENCHMARK_TEMPLATE(DeserializeBytes, std::vector<mochi::Real3>)
    ->Name("Reflection/DeserializeBytes/VectorReal3")
    ->Apply(PayloadBytes);

} // namespace mochi_benchmark

#endif // MOCHI_USE_REFLECTION
