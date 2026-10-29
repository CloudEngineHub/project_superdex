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

#include <mochi_core/materials/batched_smith_neo_hookean.h>

#include <mochi_core/materials/material_params_utils.h>
#include <mochi_core/materials/material_utils.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/decomposition_utils.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/simd.h>

#include <array>
#include <limits>

namespace mochi::materials {

// Lanes where shift·I − weight·S has a negative eigenvalue, for a symmetric S, decided to rounding
// accuracy; for S = FᵀF and weight >= 0, i.e. weight·max σᵢ² > shift. A symmetric matrix M is
// positive semidefinite iff its largest diagonal entry d is positive and the Schur complement of d
// is positive semidefinite, or M = 0, which counts as indefinite here. Eliminating the largest
// diagonal entry first, as pivoted Cholesky does, keeps the decision accurate when M has two
// eigenvalues near zero, as for S = FᵀF when the two largest singular values of F are close, where
// det(M) cancels to rounding noise.
template <int kBatchSize>
[[nodiscard]] static MOCHI_FORCE_INLINE BatchReal<kBatchSize> IsShiftedIndefinite(
    BatchReal<kBatchSize> const& shift,
    BatchReal<kBatchSize> const& weight,
    BatchSymMatrix3x3<kBatchSize> const& S) {
  using V = BatchReal<kBatchSize>;
  V const m00 = shift - weight * S[0];
  V const m11 = shift - weight * S[1];
  V const m22 = shift - weight * S[2];
  V const m01 = -weight * S[3];
  V const m02 = -weight * S[4];
  V const m12 = -weight * S[5];

  // Order the indices as (p, i, j) = (0, 1, 2), (1, 0, 2) or (2, 0, 1), with d = m_pp the largest.
  auto const p1 = m11 > m00;
  V const d01 = Select(p1, m11, m00);
  auto const p2 = m22 > d01;
  auto const p12 = p1 | p2;
  V const d = Select(p2, m22, d01);
  V const mpi = Select(p2, m02, m01);
  V const mpj = Select(p12, m12, m02);
  V const mii = Select(p12, m00, m11);
  V const mjj = Select(p2, m11, m22);
  V const mij = Select(p2, m01, Select(p1, m02, m12));

  // The Schur complement of d, divided by d, which bounds its entries by 1 when M is positive
  // semidefinite, whatever M's scale.
  V const zero = V{0_r};
  V const one = V{1_r};
  auto const positiveD = d > zero;
  V const invD = one / Select(positiveD, d, one);
  V const ri = mpi * invD;
  V const rj = mpj * invD;
  V const sii = mii * invD - ri * ri;
  V const sjj = mjj * invD - rj * rj;
  V const sij = mij * invD - ri * rj;
  return ~(positiveD & (sii >= zero) & (sjj >= zero) & (sii * sjj >= sij * sij));
}

// Lanes where the Smith tangent's scaling block A (Asym below) may be indefinite, among those with
// J > alpha and nonnegative twist and flip eigenvalues or with 0 <= J < 2/5 * alpha. With
// s = (sigma_i^2) and 1 = (1, 1, 1),
//
//   A' = diag(sigma) * A * diag(sigma) = diag(a * s_i + c) + kappa * 1 * 1^T + c3 * s * s^T,
//
// with a = muHatK, c = lambdaHat * (alpha - J) * J, kappa = lambdaHat * J * (2 * J - alpha) and
// c3 = 2 * muHat / (Ic + 1)^2, all passed divided by 4/3 * lambdaHat. For J > 0, A' is congruent
// to A, so A is positive definite iff A' is. For 2/5 * alpha <= J <= alpha, A' is positive
// definite: diag(a * s_i) is positive definite, c3 * s * s^T is positive semidefinite, and so is
// c * I + kappa * 1 * 1^T, whose eigenvalues are c and
// c + 3 * kappa = lambdaHat * J * (5 * J - 2 * alpha). In the tested lanes, A' has at most one
// negative eigenvalue: for J > alpha, nonnegative twist and flip keep a * s_i + c >= 0 for the two
// largest sigma_i, and the other terms are positive semidefinite; for J < 2/5 * alpha, only the
// rank-one kappa * 1 * 1^T can have a negative eigenvalue. So A' is positive definite iff
// det(A') > 0, a symmetric polynomial in s_i that follows from the invariants Ic,
// IIc = sum_{i<j} s_i * s_j and J^2 of F^T * F, without an SVD. Within the twist and flip test's
// tolerance, A' can have two slightly negative eigenvalues that det(A') misses. Rounding decides
// near-isotropic expansions whose smallest scaling eigenvalue is within a few sqrt(eps) * muHatK of
// zero. det(A') = 0, as for J = 0, counts as indefinite.
template <int kBatchSize>
[[nodiscard]] static MOCHI_FORCE_INLINE BatchReal<kBatchSize> IsScalingBlockIndefinite(
    BatchReal<kBatchSize> const& a,
    BatchReal<kBatchSize> const& c,
    BatchReal<kBatchSize> const& kappa,
    BatchReal<kBatchSize> const& c3,
    BatchReal<kBatchSize> const& Ic,
    BatchReal<kBatchSize> const& IIc,
    BatchReal<kBatchSize> const& Jsq) {
  using V = BatchReal<kBatchSize>;

  // With d_i = a * s_i + c and {i, j, k} = {0, 1, 2}, the matrix determinant lemma gives
  //   det(A') = e3 + kappa * e2 + c3 * sum_i s_i^2 * d_j * d_k
  //             + kappa * c3 * sum_k (s_i - s_j)^2 * d_k
  // for the elementary symmetric functions e2 and e3 of d_i. Below, p2 = sum_i s_i^2,
  // q = sum_{i != j} s_i^2 * s_j, r = sum_k (s_i - s_j)^2 / 2 and u = sum_k (s_i - s_j)^2 * s_k.
  V const p2 = Ic * Ic - V{2_r} * IIc;
  V const q = Ic * IIc - V{3_r} * Jsq;
  V const r = Ic * Ic - V{3_r} * IIc;
  V const u = Ic * IIc - V{9_r} * Jsq;

  V const e2 = a * a * IIc + V{2_r} * a * c * Ic + V{3_r} * c * c;
  V const e3 = a * a * a * Jsq + a * a * c * IIc + a * c * c * Ic + c * c * c;

  V const det = e3 + kappa * e2 + c3 * (a * a * Jsq * Ic + a * c * q + c * c * p2) +
      kappa * c3 * (a * u + V{2_r} * c * r);
  return ~(det > V{0_r});
}

// Upper triangle of c₃·F⊗F + λ̂·cof(F)⊗cof(F) + c₂·I + Σₙ wₙ·Mₙ⊗Mₙ: the direct tangent plus the
// corrections, without the λ̂(J−α)·d²J/dF² term, which the caller adds. Contraction strategy: hoist
// the temporary products, the fastest for rank-2 contractions on x86-64 (AVX2) and ARM (NEON), and
// write all terms in one pass, since each extra pass over the triangle costs a load and a store per
// entry.
template <int kBatchSize, int kModes>
static MOCHI_FORCE_INLINE void AssembleDirectTangentUpperTriangle(
    BatchReal<kBatchSize> const& c2,
    BatchReal<kBatchSize> const& c3,
    BatchReal<kBatchSize> const& lambdaHat,
    BatchReal3x3<kBatchSize> const& F,
    BatchReal3x3<kBatchSize> const& cofF,
    std::array<BatchReal<kBatchSize>, kModes> const& w,
    std::array<BatchReal3x3<kBatchSize>, kModes> const& M,
    NdArray<BatchReal3x3<kBatchSize>, 3, 3>& C) {
  using V = BatchReal<kBatchSize>;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      int const ij = i * 3 + j;
      V const c3Fij = c3 * F[i][j];
      V const lambdaHatCofFij = lambdaHat * cofF[i][j];
      std::array<V, kModes> wMij MOCHI_NO_INIT;
      for (int n = 0; n < kModes; ++n) {
        wMij[n] = w[n] * M[n][i][j];
      }

      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          if (k * 3 + l >= ij) {
            V entry = c3Fij * F[k][l] + lambdaHatCofFij * cofF[k][l];
            // Without the guard, some compilers (e.g. MSVC) keep the k and l loops rolled for
            // kModes = 0.
            if constexpr (kModes > 0) {
              for (int n = 0; n < kModes; ++n) {
                entry += wMij[n] * M[n][k][l];
              }
            }
            C[i][j][k][l] = entry;
          }
        }
      }
      C[i][j][i][j] += c2;
    }
  }
}

template <int kBatchSize>
void BatchedSmithNeoHookeanConstitutiveResponse(
    BatchLameParams<kBatchSize> const& params,
    BatchReal3x3<kBatchSize> const& F,
    BatchDouble<kBatchSize>* outEnergy,
    BatchReal3x3<kBatchSize>* outPK1,
    NdArray<BatchReal3x3<kBatchSize>, 3, 3>* outTangent,
    bool projectPsd,
    MaterialPsdOracle oracle) {
  using V = BatchReal<kBatchSize>;
  using Vd = BatchDouble<kBatchSize>;
  using V3 = BatchReal3<kBatchSize>;
  using V9 = BatchReal9<kBatchSize>;
  using V3x3 = BatchReal3x3<kBatchSize>;
  MOCHI_ASSERT_VERBOSE(
      IsFinite(params.mu) && AllTrue<kBatchSize>(params.mu > V{0_r}) && IsFinite(params.lambda),
      "Invalid Lame parameters.");
  MOCHI_ASSERT_VERBOSE(
      IsResolvedPsdStrategySupported<SmithNeoHookeanMaterialParams>(params.psdStrategy),
      "Batched material response requires a supported, resolved PSD strategy.");
  MaterialPsdStrategy const psdStrategy = params.psdStrategy;

  // Reparameterized Lamé constants (Sec. 3.4 of Smith et al. 2018).
  V const one = V{1_r};
  V const muHat = params.mu * V{4_r / 3_r};
  V const lambdaHat = params.lambda + params.mu * V{5_r / 6_r};
  V const alphaMinusOne = V{3_r / 4_r} * muHat / lambdaHat;
  V const alpha = one + alphaMinusOne;

  // Invariants.
  V const Ic = NormSqr(F);
  V const IcPlus1 = Ic + one;
  V const IcPlus1Inv = one / IcPlus1;
  V const J = Det(F);
  V const Jma = J - alpha;

  if (outEnergy) {
    *outEnergy = StaticCast<Vd>(
        V{0.5_r} * muHat * (Ic - V{3_r}) + V{0.5_r} * lambdaHat * Jma * Jma -
        V{0.5_r} * muHat * Ln(IcPlus1));
  }

  bool projectingPsd = projectPsd && (psdStrategy != MaterialPsdStrategy::None);
  V isIndefiniteMask = ~SimdZero<V>(); // All lanes indefinite unless oracle proves otherwise.

  if (outTangent && projectingPsd && oracle != MaterialPsdOracle::None) {
    // PSD oracle: evaluate indefiniteness per lane, skipping projection if no lane requires it. The
    // tangent's eigenvalues are those of the 3x3 scaling block and the twist and flip eigenvalues
    //
    //   e_twist_i =  lambdaHat * (J - alpha) * sigma_i + muHatK
    //   e_flip_i  = -lambdaHat * (J - alpha) * sigma_i + muHatK
    //
    // where muHatK = muHat * (1 - 1 / (Ic + 1)), Ic = sum_i sigma_i^2, and
    // J = prod_i sigma_i. Inverted elements (J < 0) are indefinite. For J >= 0, the sign cases for
    // (J - alpha) and sigma_i reduce the twist and flip condition to checking:
    //
    //   lambdaHat * abs(J - alpha) * abs(sigma_i) > muHatK.
    //
    // Squaring and taking the largest singular value gives the exact condition
    //
    //   lambdaHat^2 * (J - alpha)^2 * max_i(sigma_i^2) > muHatK^2,
    //
    // evaluated divided by (4/3 * lambdaHat)^2, which keeps its terms O(1) whatever the stiffness:
    // lhsBase * max_i(sigma_i^2) > rhs, with lhsBase = (3/4 * (J - alpha))^2 and rhs = a^2 for
    // a = (alpha - 1) * (1 - 1 / (Ic + 1)), since alpha - 1 = 3/4 * muHat / lambdaHat.
    //
    // max_i(sigma_i^2) is the largest eigenvalue of F^T * F, so Correct decides this without an
    // SVD: the condition holds iff rhs * I - lhsBase * F^T * F is not positive semidefinite.
    // Conservative uses Ic, an upper bound on max_i(sigma_i^2), so it can project more often than
    // necessary but does not miss required projections.
    //
    // At rest and at rotations, J = 1, Ic = 3 and max_i(sigma_i^2) = 1, so the two sides are equal
    // and rounding decides the comparison: (alpha + 1) / (alpha - 1), the condition number of
    // J - alpha, amplifies relative errors in J, such as those from rounding in F, and in alpha.
    // So that Correct counts such lanes as positive semidefinite, its rhs is scaled by 1 + delta,
    // with delta = min(16 * eps * (alpha + 1) / (alpha - 1), 8192 * eps). A lane this exempts can
    // keep a twist or flip eigenvalue negative by up to about delta / 2 * muHatK, which at rest is
    // at most eps * (16 * lambdaHat + 6 * muHat), under 6 eps times the tangent's Frobenius norm
    // there, and in any case at most 4096 * eps * muHatK. The cap binds for Poisson's ratios
    // above about 0.498; above about 0.499, the rounding error at rest and at rotations can exceed
    // delta, so rounding decides such lanes. Conservative flags rest states anyway, since Ic = 3
    // there, so it keeps the unscaled rhs.
    //
    // The scaling block can be indefinite while twist and flip are not, but only for J > alpha, as
    // in anisotropic expansions, or J < 2/5 * alpha, as in compressions of strongly auxetic
    // materials. Correct tests it there with IsScalingBlockIndefinite. Conservative needs no such
    // test: the lanes it skips have lambdaHat * abs(J - alpha) * sqrt(Ic) <= muHatK, which keeps
    // the scaling block positive definite for J > alpha and otherwise implies
    // J >= (alpha + 2) / 3, above 2/5 * alpha for every Poisson's ratio above -1, unless F = 0,
    // where the tangent is zero.
    V const a = alphaMinusOne * (one - IcPlus1Inv);
    V const lhsBase = Sqr(V{3_r / 4_r} * Jma);

    if (oracle == MaterialPsdOracle::Correct) {
      constexpr real kEps = std::numeric_limits<real>::epsilon();
      V const rhs = alphaMinusOne *
          (alphaMinusOne + Min(V{16_r * kEps} * (alpha + one), V{8192_r * kEps} * alphaMinusOne)) *
          Sqr(one - IcPlus1Inv);
      // max_i(sigma_i^2) lies between the largest diagonal entry of F^T * F and the smaller of its
      // trace and largest absolute row sum. Lanes these bounds leave open, and lanes with J > alpha
      // or J < 2/5 * alpha not proven indefinite, count as indefinite unless the exact tests settle
      // them. The Projection strategies project the whole batch once any lane is indefinite, so
      // they skip the tests once a lane is proven indefinite; Fast drops the term per lane, so it
      // always runs them.
      BatchSymMatrix3x3<kBatchSize> const G = {
          F[0][0] * F[0][0] + F[1][0] * F[1][0] + F[2][0] * F[2][0],
          F[0][1] * F[0][1] + F[1][1] * F[1][1] + F[2][1] * F[2][1],
          F[0][2] * F[0][2] + F[1][2] * F[1][2] + F[2][2] * F[2][2],
          F[0][0] * F[0][1] + F[1][0] * F[1][1] + F[2][0] * F[2][1],
          F[0][0] * F[0][2] + F[1][0] * F[1][2] + F[2][0] * F[2][2],
          F[0][1] * F[0][2] + F[1][1] * F[1][2] + F[2][1] * F[2][2]};
      V const lowerBound = Max(G[0], G[1], G[2]);
      V const upperBound =
          Min(Ic,
              Max(G[0] + Abs(G[3]) + Abs(G[4]),
                  G[1] + Abs(G[3]) + Abs(G[5]),
                  G[2] + Abs(G[4]) + Abs(G[5])));

      V const provenIndefinite = (lhsBase * lowerBound > rhs) | (J < V{0_r});
      V const twistFlipUndecided = ~provenIndefinite & (lhsBase * upperBound > rhs);
      V const scalingUndecided = ~provenIndefinite & ((J > alpha) | (V{2.5_r} * J < alpha));
      isIndefiniteMask = provenIndefinite | twistFlipUndecided | scalingUndecided;

      if (AnyTrue<kBatchSize>(twistFlipUndecided | scalingUndecided) &&
          (psdStrategy == MaterialPsdStrategy::Fast || !AnyTrue<kBatchSize>(provenIndefinite))) {
        isIndefiniteMask = provenIndefinite;
        if (AnyTrue<kBatchSize>(twistFlipUndecided)) {
          isIndefiniteMask |= twistFlipUndecided & IsShiftedIndefinite<kBatchSize>(rhs, lhsBase, G);
        }

        if (AnyTrue<kBatchSize>(scalingUndecided)) {
          // IIc from G cancels only for needle-like F, whose tangent is indefinite iff twist or
          // flip is negative, so the cancellation can only add flags to the twist and flip test's.
          // For such F, the rounding of J can already add flags.
          V const IIc =
              G[0] * G[1] + G[0] * G[2] + G[1] * G[2] - G[3] * G[3] - G[4] * G[4] - G[5] * G[5];
          V const scalingIndefinite = IsScalingBlockIndefinite<kBatchSize>(
              a,
              V{-3_r / 4_r} * Jma * J,
              V{3_r / 4_r} * J * (V{2_r} * J - alpha),
              V{2_r} * alphaMinusOne * IcPlus1Inv * IcPlus1Inv,
              Ic,
              IIc,
              J * J);
          isIndefiniteMask |= scalingUndecided & scalingIndefinite;
        }
      }
    } else {
      MOCHI_ASSERT_VERBOSE(oracle == MaterialPsdOracle::Conservative, "Unexpected PSD oracle.");
      isIndefiniteMask = (lhsBase * Ic > Sqr(a)) | (J < V{0_r});
    }

    projectingPsd = AnyTrue<kBatchSize>(isIndefiniteMask);
  }

  // The cofactor of the deformation gradient.
  V3x3 cofF MOCHI_NO_INIT;
  if (outPK1 || outTangent) {
    cofF = Cofactor(F);
  }

  // The first Piola--Kirchhoff.
  if (outPK1) {
    V const c0 = muHat * (one - IcPlus1Inv);
    V const c1 = lambdaHat * Jma;
    // NOTE: *outPK1 = c0 * F + c1 * cofF is cleaner but may cause register spilling.
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        (*outPK1)[r][c] = c0 * F[r][c] + c1 * cofF[r][c];
      }
    }
  }

  if (outTangent) {
    auto& C = *outTangent;

    // Direct tangent: C[ij][kl] = c₃·F[ij]·F[kl] + λ̂·cofF[ij]·cofF[kl]
    //                            + λ̂(J−α)·d²J/dF[ij]dF[kl] + c₂·δᵢₖδⱼₗ
    //
    // Coefficients:
    //   c₂ = μ̂(1 − 1/(Ic+1))   (identity term)
    //   c₃ = 2μ̂ / (Ic+1)²      (Ic² second derivative)
    V const c2 = muHat * (one - IcPlus1Inv);
    V const c3 = V{2_r} * muHat * IcPlus1Inv * IcPlus1Inv;
    V const d2JCoeff = lambdaHat * Jma;

    bool const projectingAnalytically = projectingPsd &&
        (psdStrategy == MaterialPsdStrategy::Projection ||
         psdStrategy == MaterialPsdStrategy::AbsEigenProjection);
    if (projectingAnalytically) {
      // Projection with the analytic eigensystem (Smith et al. 2018).
      // Note: BatchedRotationVariantSvd3x3 returns U, V^T and sigma such that det(U) >= 0, det(V^T)
      // >= 0, and sigma[2] < 0 iff det(F) < 0. On nearly flat elements sigma[2] is accurate only to
      // about ε·(σ₀/σ₂)², so the check below that skips the scaling block's projection can accept a
      // slightly indefinite block there, leaving a negative eigenvalue of up to about 3e-5·‖H‖ in
      // float and 2e-9·‖H‖ in double. See D124644721 for an accurate sigma[2] and its cost.
      V3x3 U MOCHI_NO_INIT, VT MOCHI_NO_INIT;
      V3 sigma MOCHI_NO_INIT;
      BatchedRotationVariantSvd3x3<kBatchSize>(F, U, sigma, VT);

      // Scaling mode matrix A (3×3 symmetric).
      V const J2alphaLambda = (V{2_r} * J - alpha) * lambdaHat;
      V3 const sigmaSqr = Sqr(sigma);
      BatchSymMatrix3x3<kBatchSize> const Asym = {
          c2 + c3 * sigmaSqr[0] + lambdaHat * sigmaSqr[1] * sigmaSqr[2],
          c2 + c3 * sigmaSqr[1] + lambdaHat * sigmaSqr[0] * sigmaSqr[2],
          c2 + c3 * sigmaSqr[2] + lambdaHat * sigmaSqr[0] * sigmaSqr[1],
          J2alphaLambda * sigma[2] + c3 * sigma[0] * sigma[1],
          J2alphaLambda * sigma[1] + c3 * sigma[0] * sigma[2],
          J2alphaLambda * sigma[0] + c3 * sigma[1] * sigma[2]};

      V const minEig = V{kMinProjectedEigenvalue};
      auto const clamped = [&](V const& e) MOCHI_FORCE_INLINE_LAMBDA {
        return psdStrategy == MaterialPsdStrategy::Projection ? Max(e, minEig)
                                                              : Max(Abs(e), minEig);
      };

      // The corrections below cancel the clamped eigenvalues only as accurately as the SVD's
      // singular vectors, whose error grows like ε·(σ₀/σ₁)²: they fail beyond a ratio of about 200
      // in float and 1000 in double. 16 keeps that error near 256·ε, and the residual eigenvalues
      // within about 50·ε·‖H‖.
      constexpr real kMaxSigmaRatio = 16_r;

      // The whole batch takes the eigensystem path if any lane has
      // - c₂ < minEig, where both eigenvalues of a twist-flip pair can need clamping;
      // - σ₀ > kMaxSigmaRatio·σ₁;
      // - a scaling eigenvalue below minEig, i.e. A − minEig·I not positive semidefinite.
      if (AnyTrue<kBatchSize>(
              (c2 < minEig) | (sigma[0] > V{kMaxSigmaRatio} * sigma[1]) |
              IsShiftedIndefinite<kBatchSize>(-minEig, V{-1_r}, Asym))) {
        // 9 eigenvalues of the Hessian in layout [scaling_0..2, twist_0..2, flip_0..2], with twist
        // eigenvalues c₂ + λ̂(J−α)·σₙ and flip eigenvalues c₂ − λ̂(J−α)·σₙ.
        V9 eigLambda MOCHI_NO_INIT;
        V3 scalingEigvals MOCHI_NO_INIT;
        V3x3 AqT MOCHI_NO_INIT;
        BatchedAnalyticalEigendecompSym3x3<kBatchSize>(Asym, scalingEigvals, &AqT);

        for (int n = 0; n < 3; ++n) {
          eigLambda[n] = clamped(scalingEigvals[n]);
          eigLambda[3 + n] = clamped(c2 + d2JCoeff * sigma[n]);
          eigLambda[6 + n] = clamped(c2 - d2JCoeff * sigma[n]);
        }

        utils::BatchedAssembleTangentFromEigensystem<kBatchSize>(eigLambda, U, VT, AqT, C);
        return;
      }

      // The scaling eigenvalues are at least minEig in every lane, and so is the larger of each
      // twist-flip pair c₂ ± λ̂(J−α)·σₙ, as c₂ ≥ minEig. So projecting adds (clamped(e) − e)·m⊗m
      // to the direct tangent for the smaller one, e = c₂ − |λ̂(J−α)·σₙ|, along its mode
      // m = (u_p·v_qᵀ + sign(λ̂(J−α)·σₙ)·u_q·v_pᵀ)/√2: twist where λ̂(J−α)·σₙ < 0, else flip, as in
      // BatchedAssembleTangentFromEigensystem. The modes omit the 1/√2, so the weights are halved.
      // Unlike that assembly, which is PSD by construction, the result is PSD only as accurately as
      // U and V match F.
      constexpr int kPairs[3][2] = {{1, 2}, {0, 2}, {0, 1}};
      std::array<V, 3> weights MOCHI_NO_INIT;
      std::array<V3x3, 3> modes MOCHI_NO_INIT;
      for (int n = 0; n < 3; ++n) {
        V const dJSigma = d2JCoeff * sigma[n];
        V const e = c2 - Abs(dJSigma);
        weights[n] = V{0.5_r} * (clamped(e) - e);
        int const p = kPairs[n][0];
        int const q = kPairs[n][1];
        V const sign = Sign(dJSigma);
        for (int r = 0; r < 3; ++r) {
          V const signUrq = sign * U[r][q];
          for (int c = 0; c < 3; ++c) {
            modes[n][r][c] = U[r][p] * VT[q][c] + signUrq * VT[p][c];
          }
        }
      }

      AssembleDirectTangentUpperTriangle<kBatchSize, 3>(
          c2, c3, lambdaHat, F, cofF, weights, modes, C);
    } else {
      AssembleDirectTangentUpperTriangle<kBatchSize, 0>(c2, c3, lambdaHat, F, cofF, {}, {}, C);
    }

    // Add the λ̂(J−α)·d²J/dF² term, except under the Fast PSD strategy in the lanes where it makes
    // the tangent indefinite, and altogether when that is every lane.
    bool const projectingFast = projectingPsd && (psdStrategy == MaterialPsdStrategy::Fast);
    if (!projectingFast || !AllTrue<kBatchSize>(isIndefiniteMask)) {
      utils::BatchAddD2JdF2UpperTriangle<kBatchSize>(
          projectingFast ? Select(isIndefiniteMask, V{0_r}, d2JCoeff) : d2JCoeff, F, C);
    }

    utils::BatchMirrorTangentUpperToLower<kBatchSize>(C);
  }
}

MOCHI_INSTANTIATE_BATCHED_MATERIAL(
    BatchedSmithNeoHookeanConstitutiveResponse,
    BatchLameParams,
    MaterialPsdOracle);

} // namespace mochi::materials
