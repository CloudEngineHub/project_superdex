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

#pragma once

#include <mochi_core/elements/triangular/basis_functions_evaluated.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/matrix_utils.h>
namespace mochi::triangular {

/**
  The lagrange triangle element.

  In the mindset of keeping memory demand as low as possible
  and high efficiency, the element assumes the same interpolant
  for each of the fields; a reasonable assumption in the case of
  elasticity. This code is not intended to be as general as
  possible rather striking a balance between somewhat
  efficient and versatile.
*/
template <int kPolyOrder_, int kNumQuadPoints_ = 1>
class Pk2DElement final {
  static_assert(kPolyOrder_ == 1, "Higher order polynomials not yet implemented");

 public:
  static constexpr int kPolyOrder = kPolyOrder_;
  static constexpr int kNumQuadPoints = kNumQuadPoints_;
  static constexpr int kNumNodes = 3;
  using Basis = BarycentricBasisTriangular<kPolyOrder>;
  using BasisEvaluated = BasisFunctionsEvaluatedTriangular<kPolyOrder, kNumQuadPoints>;

  /**
   The constructor for a lagrange tet element

   Args:
     elementIndex (int): the index of the element
     coordinates (ndarray): reference to the whole set of
       coordinates of the mesh nodes (passed by reference)
     connectivity (ndarray): reference to the connectivity
       of the whole mesh (passed by reference)
  */
  Pk2DElement(int elementIndex, Span<Real3 const> coordinates, Span<Int3 const> connectivity)
      : elementIndex(elementIndex), coordinates(coordinates), connectivity(connectivity) {
    // Construct the array containing the physical coordinates
    // corresponding to the coordinates of each degree of freedom
    InterpolateInteriorNodes();

    Initialize();
  }

 private:
  /**
    Tabulate the isoparametric map, the normals, the quadrature weights and the basis derivatives
    with respect to physical coordinates at the quad points
  */
  void Initialize() {
    for (int q = 0; q < kNumQuadPoints; ++q) {
      // Here we are using an isoparametric map
      mapEvaluated[q] = {};
      NdArray<real, kSpaceDim, kSpaceDimParam> dMap = {};
      for (int f = 0; f < kNumDofs; ++f) {
        // Get the map of the quadrature point
        mapEvaluated[q] += kBasisEvaluatedParametric.kBasisEvaluated[q][f] * nodesCrdsPhys[f];

        // Get the tangent map at the quadrature point
        // effectively here we have the induced basis from the mapping
        // the matrix is a space_dim x param_dim so in each column
        // we have an induced base
        dMap += Outer(nodesCrdsPhys[f], kBasisEvaluatedParametric.kDBasisEvaluated[q][f]);
      }
      normals[q] = Cross(Transpose(dMap)[0], Transpose(dMap)[1]);
      normals[q] /= Norm(normals[q]);

      // Get the metric change
      // this is the norm of the cross product of the
      // induced basis (each stored in [:,baseIndex])
      NdArray<real, kSpaceDimParam, kSpaceDim> dMapInv;
      real dMapDet = 0_r;
      PseudoInvert(dMap, &dMapInv, &dMapDet);

      // Compute quadrature weight
      quadWeights[q] = kQuadrature.weights[q] * dMapDet;

      // Compute gradient with respect to physical coordinates
      dBasisEvaluated[q] = {};
      for (int f = 0; f < kNumDofs; ++f) {
        dBasisEvaluated[q][f] =
            DotMatVec(Transpose(dMapInv), kBasisEvaluatedParametric.kDBasisEvaluated[q][f]);
      }
    }
  }

  /**
    Computes the nodal position in physical space of interior nodes.

    Only relevant for higher order elements this method interpolates
    linearly the mid and interior nodes on the physical domain.
  */
  void InterpolateInteriorNodes() {
    // Here we are simply interpolating linearly the physical coordinates
    // of the interior nodes this could be more general
    constexpr int kNumVertex = 3;
    for (int i = 0; i < kNumVertex; ++i) {
      nodesCrdsPhys[i] = coordinates[connectivity[elementIndex][i]];
    }

    if constexpr (kPolyOrder > 1) {
      auto nodeCrdsParams = PrecomputeNodesCrdsParam();
      for (int i = kNumVertex; i < kNumDofs; ++i) {
        nodesCrdsPhys[i] = nodeCrdsParams[0] * nodeCrdsParams[0] +
            nodeCrdsParams[1] * nodeCrdsParams[1] +
            (1_r - nodeCrdsParams[0] - nodeCrdsParams[1]) * nodeCrdsParams[2];
      }
    }
  }

 public:
  static constexpr int kSpaceDim = 3;
  static constexpr int kSpaceDimParam = 2;
  static constexpr int kNumDofs = Basis::kNumDofs;
  static constexpr TriangleQuadrature<kNumQuadPoints> kQuadrature = {};
  static constexpr BasisEvaluated kBasisEvaluatedParametric = {};
  static constexpr Basis kBasis = {};
  static constexpr NdArray<real, kNumQuadPoints, kNumDofs> kBasisEvaluated =
      kBasisEvaluatedParametric.kBasisEvaluated;
  // FIXME: Creating an alias without `k` for consistency with tetrahedral case, to use a templated
  // method for mass matrix assembly.
  static constexpr NdArray<real, kNumQuadPoints, kNumDofs> basisEvaluated = kBasisEvaluated;

  // Compile-time algorithm to compute kNodesCrdsParam
  static constexpr auto PrecomputeNodesCrdsParam() {
    // Get the coordinates of the nodes in the parametric domain
    // this are the nodes of the exterior vertices, edge nodes,
    // as well as interior nodes. Here we remove the last column
    // that correspond to the additional barycentric coordinate
    NdArray<real, kNumDofs, kSpaceDimParam> nodesCrdsParam = {};
    for (size_t i = 0; i < kBasis.kDofNodes.size(); ++i) { // for each row
      auto const& row = kBasis.kDofNodes[i];
      for (size_t j = 0; j < row.size() - 1; ++j) { // for each column except the last one
        nodesCrdsParam[i][j] = kBasis.kDofNodes[i][j];
      }
    }
    return nodesCrdsParam;
  }

  static constexpr NdArray<real, kNumDofs, kSpaceDimParam> kNodesCrdsParam =
      PrecomputeNodesCrdsParam();

  // NOTE: These seemingly trivial accessor methods are for compatibility with templated
  // code that also supports trace elements, for which these methods are non-trivial.
  inline auto const& GetBaseElement() const {
    return *this;
  }
  inline Int3 Nodes() const {
    return connectivity[elementIndex];
  }
  inline constexpr Int3 LocalNodes() const {
    return Int3{0, 1, 2};
  }
  inline int GetElementIndex() const {
    return elementIndex;
  }

  // Set by constructor
  int const elementIndex;
  Span<Real3 const> const coordinates;
  Span<Int3 const> const connectivity;

  // The nodal position in physical space of interior nodes.
  NdArray<real, kNumDofs, kSpaceDim> nodesCrdsPhys;

  // The array of mapped quadrature points into physical space
  NdArray<real, kNumQuadPoints, kSpaceDim> mapEvaluated;

  // The unit normal at the quadrature points
  NdArray<real, kNumQuadPoints, kSpaceDim> normals;

  // dbasis evaluated at quad pts
  NdArray<real, kNumQuadPoints, kNumDofs, kSpaceDim> dBasisEvaluated;

  // The quadrature weights for area integration
  NdArray<real, kNumQuadPoints> quadWeights = {};
};

} // namespace mochi::triangular
