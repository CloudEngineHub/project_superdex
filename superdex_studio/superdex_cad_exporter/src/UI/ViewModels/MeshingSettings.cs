/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using Meshing;

namespace CADRobotExporter.UI.ViewModels
{
    /// <summary>
    /// Which controls in one meshing column are usable, given that column's own state.
    /// </summary>
    public struct MeshingColumnEnableState
    {
        public bool Backend;
        public bool EdgeSampling;
        public bool TargetEdgeLength;
        public bool EdgeLengthFraction;
    }

    /// <summary>
    /// Presentation rules for the meshing settings columns: how combo box indices map to the
    /// meshing enums, which controls are usable, and how persisted values are kept in range.
    /// </summary>
    public static class MeshingSettings
    {
        // Mapped explicitly rather than cast, so the combo item order stays a presentation choice
        // and cannot silently change which backend a saved configuration means.
        public const int BackendIndexIsotropic = 0;
        public const int BackendIndexDelabella = 1;
        public const int EdgeSamplingIndexAdaptive = 0;
        public const int EdgeSamplingIndexUniform = 1;

        public static MeshingBackend BackendFromIndex(int index)
        {
            return index == BackendIndexDelabella ? MeshingBackend.Delabella : MeshingBackend.Isotropic;
        }

        public static int IndexFromBackend(MeshingBackend backend)
        {
            return backend == MeshingBackend.Delabella ? BackendIndexDelabella : BackendIndexIsotropic;
        }

        public static EdgeSampling EdgeSamplingFromIndex(int index)
        {
            return index == EdgeSamplingIndexUniform ? EdgeSampling.Uniform : EdgeSampling.Adaptive;
        }

        public static int IndexFromEdgeSampling(EdgeSampling sampling)
        {
            return sampling == EdgeSampling.Uniform ? EdgeSamplingIndexUniform : EdgeSamplingIndexAdaptive;
        }

        /// <summary>
        /// The isotropic-only settings follow the column's backend, and the edge length fraction
        /// is only meaningful while the absolute target length is zero.
        /// </summary>
        public static MeshingColumnEnableState EnableStateFor(
            bool columnEnabled, MeshingBackend backend, bool targetEdgeLengthIsZero)
        {
            bool isotropic = backend == MeshingBackend.Isotropic;
            return new MeshingColumnEnableState
            {
                Backend = columnEnabled,
                EdgeSampling = columnEnabled && isotropic,
                TargetEdgeLength = columnEnabled && isotropic,
                EdgeLengthFraction = columnEnabled && isotropic && targetEdgeLengthIsZero,
            };
        }

        /// <summary>
        /// Persisted values are never validated against the control's range, and NumericUpDown
        /// throws rather than clamping when assigned out of range -- inside a CAD host that means
        /// taking the session down over a stale configuration.
        /// </summary>
        public static decimal Clamp(decimal value, decimal minimum, decimal maximum)
        {
            if (value < minimum)
            {
                return minimum;
            }
            if (value > maximum)
            {
                return maximum;
            }
            return value;
        }
    }
}
