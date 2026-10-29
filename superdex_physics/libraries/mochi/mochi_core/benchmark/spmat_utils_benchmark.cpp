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

#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/rand_utils.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/spmat_utils.h>

#include <algorithm>
#include <numeric>

using namespace mochi;

namespace mochi_benchmark {
namespace {

// The sparsity of numRows rows and the columns that G groups hit in each: every row stores R of 4 R
// columns, at random, and the groups hit G of them, either at random or at the end of the row. Both
// are sorted within each row.
struct RowsAndGroups {
  DynamicArray<int> cols; // numRows x R
  DynamicArray<int> groupCols; // numRows x G
};

} // namespace

static RowsAndGroups MakeRowsAndGroups(int numRows, int R, int G, bool groupsAtEnd) {
  auto rng = RandomGenerator(/*seed=*/1);
  RowsAndGroups out{
      .cols = DynamicArray<int>(numRows * R), .groupCols = DynamicArray<int>(numRows * G)};
  for (int r = 0; r < numRows; ++r) {
    DynamicArray<int> rowCols(4 * R);
    std::iota(rowCols.begin(), rowCols.end(), 0);
    RandomSubset(rowCols, R, rng);
    std::ranges::sort(rowCols);
    std::ranges::copy(rowCols, &out.cols[r * R]);
    if (groupsAtEnd) {
      rowCols.erase(rowCols.begin(), rowCols.end() - G);
    } else {
      RandomSubset(rowCols, G, rng);
      std::ranges::sort(rowCols);
    }
    std::ranges::copy(rowCols, &out.groupCols[r * G]);
  }
  return out;
}

// A block sparse matrix of zero 3 x 3 blocks with the sparsity of pattern: numBlockRows block rows
// of R blocks each, out of 4 R block columns.
static BlockSparseMatrix<real, 3>
MakeBlockMatrix(RowsAndGroups const& pattern, int numBlockRows, int R) {
  DynamicArray<int> blockPointers(numBlockRows + 1);
  for (int br = 0; br <= numBlockRows; ++br) {
    blockPointers[br] = R * br;
  }
  DynamicArray<real> blockValues(isize(pattern.cols) * 3 * 3); // Zeros.
  return {4 * R, blockPointers, pattern.cols, blockValues};
}

// Adds one 3 x (3 G) dense block into one block row of a block sparse matrix per call, as a contact
// scatter does. Each block row stores R blocks, and the G column groups hit G of them (see
// MakeRowsAndGroups). Each call visits the next block row, so it finds the groups in that row
// (MatAddSubBlocks's UpdateCache) before adding the G blocks.
static void MatAddSubBlocksBlockRow(benchmark::State& state) {
  auto const R = int(state.range(0)), G = int(state.range(1));
  int const numBlockRows = Max(16, 32768 / R);
  RowsAndGroups const pattern = MakeRowsAndGroups(numBlockRows, R, G, state.range(2) != 0);
  BlockSparseMatrix<real, 3> A = MakeBlockMatrix(pattern, numBlockRows, R);
  RowMatrix<real> const values = RowMatrix<real>::Zero(3, 3 * G);

  DynamicArray<IndexGroup> rows(1), cols(G);
  int br = 0;
  for (auto _ : state) {
    rows[0] = {.src = 0, .dst = 3 * br, .count = 3};
    for (int j = 0; j < G; ++j) {
      cols[j] = {.src = 3 * j, .dst = 3 * pattern.groupCols[br * G + j], .count = 3};
    }
    MatAddSubBlocks(AsView(A), MakeSpan(rows), MakeSpan(cols), values);
    br = (br + 1 == numBlockRows) ? 0 : br + 1;
  }
}

BENCHMARK(MatAddSubBlocksBlockRow)
    ->Name("SpmatUtils/MatAddSubBlocks/BlockRow")
    ->ArgNames({"R", "G", "groupsAtEnd"})
    ->Args({24, 4, 0}) // A typical contact row: the forward scan's common case.
    ->Args({1000, 1, 1}) // A long row with few groups at its end: the unguarded scan's worst case,
    ->Args({1000, 4, 1}) // which the R <= 32 G guard sends to the binary search.
    ->Args({128, 4, 1}) // R = 32 G with the groups at its end: the most compares the scan can do.
    ->Args({90, 90, 0}); // A dense row (G = R): the binary searches' worst case, which the guard
                         // sends to the scan.

// The scalar (CSR) counterpart of BlockRow: adds one 3 x (3 G) dense block into three rows of a
// sparse matrix per call. The three rows share their sparsity: R entries, in runs of three
// consecutive columns (one node each), and each group adds to one run. R counts entries and must be
// a multiple of 3. Each call visits the next three rows, so it finds the groups once before adding
// to the rows.
static void MatAddSubBlocksScalarRow(benchmark::State& state) {
  auto const R = int(state.range(0)), G = int(state.range(1));
  int const numNodeRows = Max(16, 32768 / R);
  RowsAndGroups const pattern = MakeRowsAndGroups(numNodeRows, R / 3, G, state.range(2) != 0);
  SparseMatrix<real> A = ToSparseMatrix(MakeBlockMatrix(pattern, numNodeRows, R / 3));
  RowMatrix<real> const values = RowMatrix<real>::Zero(3, 3 * G);

  DynamicArray<IndexGroup> rows(1), cols(G);
  int nr = 0;
  for (auto _ : state) {
    rows[0] = {.src = 0, .dst = 3 * nr, .count = 3};
    for (int j = 0; j < G; ++j) {
      cols[j] = {.src = 3 * j, .dst = 3 * pattern.groupCols[nr * G + j], .count = 3};
    }
    MatAddSubBlocks(AsView(A), MakeSpan(rows), MakeSpan(cols), values);
    nr = (nr + 1 == numNodeRows) ? 0 : nr + 1;
  }
}

BENCHMARK(MatAddSubBlocksScalarRow)
    ->Name("SpmatUtils/MatAddSubBlocks/ScalarRow")
    ->ArgNames({"R", "G", "groupsAtEnd"})
    ->Args({48, 3, 0}) // A typical rod or hand contact row.
    ->Args({3000, 1, 1}) // A long row with one group at its end: the unguarded scan's worst case.
    ->Args({96, 3, 1}); // R = 32 G with the groups at its end: the most compares the scan can do.

} // namespace mochi_benchmark
