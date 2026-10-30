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

#include <mochi_core/geometry/spatial_hash_table.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>

#include <cstdint>

using namespace mochi;

namespace mochi_benchmark {

// Shape of the populated grid cells within the 3x3x3 query stencil centered on cell (0, 0, 0).
enum class OccupiedStencilCells {
  CenterCell,
  XAxisCells,
  XYPlaneCells,
  Full3x3x3Stencil,
};

static DynamicArray<Int3> MakeOccupiedCells(OccupiedStencilCells occupiedStencilCells) {
  DynamicArray<Int3> cells;
  cells.reserve(3 * 3 * 3);
  switch (occupiedStencilCells) {
    case OccupiedStencilCells::CenterCell:
      cells.push_back(Int3{0, 0, 0});
      break;
    case OccupiedStencilCells::XAxisCells:
      for (int x = -1; x <= 1; x++) {
        cells.push_back(Int3{x, 0, 0});
      }
      break;
    case OccupiedStencilCells::XYPlaneCells:
      for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
          cells.push_back(Int3{x, y, 0});
        }
      }
      break;
    case OccupiedStencilCells::Full3x3x3Stencil:
      for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
          for (int z = -1; z <= 1; z++) {
            cells.push_back(Int3{x, y, z});
          }
        }
      }
      break;
  }
  return cells;
}

static Real3 GetCellPoint(Int3 const& cell) {
  return Real3{
      (static_cast<real>(cell[0]) + 0.25_r),
      (static_cast<real>(cell[1]) + 0.25_r),
      (static_cast<real>(cell[2]) + 0.25_r)};
}

static void SpatialHashTableIteratePointsNearPosition(
    benchmark::State& state,
    OccupiedStencilCells occupiedStencilCells,
    int numBins,
    int pointsPerCell) {
  DynamicArray<Int3> const cells = MakeOccupiedCells(occupiedStencilCells);
  int const numPoints = isize(cells) * pointsPerCell;
  SpatialHashTable spatialHashTable(1.0_r, numPoints, numBins);
  int pointIndex = 0;
  for (Int3 const& cell : cells) {
    Real3 const point = GetCellPoint(cell);
    for (int i = 0; i < pointsPerCell; i++) {
      spatialHashTable.AddPoint(pointIndex, point);
      pointIndex++;
    }
  }

  Real3 const queryPosition{0.25_r, 0.25_r, 0.25_r};
  std::int64_t totalPointsVisited = 0;
  std::int64_t pointIndexChecksum = 0;
  for (auto _ : state) {
    int pointsVisited = 0;
    int pointIndexSum = 0;
    spatialHashTable.IteratePointsNearPosition(queryPosition, [&](int nearbyPointIndex) {
      pointsVisited++;
      pointIndexSum += nearbyPointIndex + 1;
    });
    totalPointsVisited += pointsVisited;
    pointIndexChecksum += pointIndexSum;
    benchmark::DoNotOptimize(pointIndexChecksum);
  }

  state.counters["Queries/second"] =
      benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
  state.counters["Visited/Query"] =
      benchmark::Counter(totalPointsVisited, benchmark::Counter::kAvgIterations);
}

#define MOCHI_SPATIAL_HASH_TABLE_BENCHMARK(OccupiedCells, NumBins, PointsPerCell)                \
  BENCHMARK_CAPTURE(                                                                             \
      SpatialHashTableIteratePointsNearPosition,                                                 \
      OccupiedCells##_##NumBins##Bins_##PointsPerCell##PointsPerCell,                            \
      OccupiedStencilCells::OccupiedCells,                                                       \
      NumBins,                                                                                   \
      PointsPerCell)                                                                             \
      ->Name(                                                                                    \
          "Geometry/SpatialHashTable/IteratePointsNearPosition/" #OccupiedCells "/Bins" #NumBins \
          "/PointsPerCell" #PointsPerCell)

MOCHI_SPATIAL_HASH_TABLE_BENCHMARK(CenterCell, 4096, 1);
MOCHI_SPATIAL_HASH_TABLE_BENCHMARK(XAxisCells, 4096, 1);
MOCHI_SPATIAL_HASH_TABLE_BENCHMARK(XYPlaneCells, 4096, 1);
MOCHI_SPATIAL_HASH_TABLE_BENCHMARK(Full3x3x3Stencil, 4096, 1);
MOCHI_SPATIAL_HASH_TABLE_BENCHMARK(Full3x3x3Stencil, 4096, 8);
MOCHI_SPATIAL_HASH_TABLE_BENCHMARK(Full3x3x3Stencil, 16, 1);

#undef MOCHI_SPATIAL_HASH_TABLE_BENCHMARK

// Queries every point of a sphere shell whose points are about 0.45 cells apart, like cloth
// contact, so which stencil bins are empty changes from query to query. The cases above repeat one
// query, so their branches always predict.
static void SpatialHashTableIteratePointsNearSphereShell(benchmark::State& state) {
  int constexpr kNumPoints = 4096;
  real constexpr kShellRadius = 8_r; // [cells]
  real constexpr kGoldenAngle = 2.4_r; // [rad], about pi * (3 - sqrt(5))
  // 16 bins per point: the default PointCloudColliderParams::spatialHashLoadFactor is 1/16.
  SpatialHashTable spatialHashTable(1.0_r, kNumPoints, 16 * kNumPoints);
  // A golden-angle spiral from pole to pole: evenly spaced z gives each point the same area.
  DynamicArray<Real3> points;
  points.reserve(kNumPoints);
  for (int i = 0; i < kNumPoints; ++i) {
    real const z = 1_r - 2_r * (StaticCast<real>(i) + 0.5_r) / StaticCast<real>(kNumPoints);
    real const ringRadius = Sqrt(1_r - z * z);
    real const angle = kGoldenAngle * StaticCast<real>(i);
    points.push_back(kShellRadius * Real3{ringRadius * Cos(angle), ringRadius * Sin(angle), z});
    spatialHashTable.AddPoint(i, points.back());
  }

  std::int64_t pointIndexChecksum = 0;
  for (auto _ : state) {
    for (Real3 const& point : points) {
      spatialHashTable.IteratePointsNearPosition(
          point, [&](int nearbyPointIndex) { pointIndexChecksum += nearbyPointIndex + 1; });
    }
    benchmark::DoNotOptimize(pointIndexChecksum);
  }

  state.counters["Queries/second"] =
      benchmark::Counter(state.iterations() * kNumPoints, benchmark::Counter::kIsRate);
}

BENCHMARK(SpatialHashTableIteratePointsNearSphereShell)
    ->Name("Geometry/SpatialHashTable/IteratePointsNearPosition/SphereShell");

} // namespace mochi_benchmark
