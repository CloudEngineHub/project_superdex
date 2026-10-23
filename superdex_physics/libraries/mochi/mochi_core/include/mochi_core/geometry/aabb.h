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

// PLEASE DO NOT ADD OTHER INCLUDES HERE. This header is included in the mochi_physics public API.
#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/concepts.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/vmatrix.h>

#include <type_traits>

namespace mochi {

/**************************************************************************************************
  Aabb - Axis-Aligned Bounding Box
*/
class Aabb final {
 public:
  // clang-format off

  MOCHI_FORCE_INLINE Aabb() = default;
  MOCHI_FORCE_INLINE Aabb(Real3 const& min, Real3 const& max) : _min{min[0], min[1], min[2], 1_r}, _max{max[0], max[1], max[2], 1_r} {}
  MOCHI_FORCE_INLINE Aabb(Vec4r min, Vec4r max) : Aabb(ToReal3(min), ToReal3(max)) {}

  // SIMD accessors (note that the 4th component is always 1 by convention)
  [[nodiscard]] MOCHI_FORCE_INLINE Vec4r VGetMin() const { return ToSimd(_min); }
  [[nodiscard]] MOCHI_FORCE_INLINE Vec4r VGetMax() const { return ToSimd(_max); }
  [[nodiscard]] MOCHI_FORCE_INLINE Vec4r VGetHalfExtents() const { return 0.5_r * VGetSize(); }
  [[nodiscard]] MOCHI_FORCE_INLINE Vec4r VGetCenter() const { return 0.5_r * (VGetMax() + VGetMin()); }
  [[nodiscard]] MOCHI_FORCE_INLINE Vec4r VGetSize() const { return VGetMax() - VGetMin(); }

  // Scalar accessors
  [[nodiscard]] MOCHI_FORCE_INLINE Real3 GetMin() const { return Real3{_min[0], _min[1], _min[2]}; }
  MOCHI_FORCE_INLINE void SetMin(Real3 const& min) { _min = Real4{min[0], min[1], min[2], 1_r}; }
  [[nodiscard]] MOCHI_FORCE_INLINE Real3 GetMax() const { return Real3{_max[0], _max[1], _max[2]}; }
  MOCHI_FORCE_INLINE void SetMax(Real3 const& max) { _max = Real4{max[0], max[1], max[2], 1_r}; }
  [[nodiscard]] MOCHI_FORCE_INLINE Real3 GetHalfExtents() const { return ToReal3(VGetHalfExtents()); }
  [[nodiscard]] MOCHI_FORCE_INLINE Real3 GetCenter() const { return ToReal3(VGetCenter()); }
  [[nodiscard]] MOCHI_FORCE_INLINE Real3 GetSize() const { return ToReal3(VGetSize()); }

  // Exact equality
  [[nodiscard]] MOCHI_FORCE_INLINE bool operator==(Aabb const& rhs) const { return AllTrue<3>(VEqual(VGetMin(), rhs.VGetMin()) & VEqual(VGetMax(), rhs.VGetMax())); }
  [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Aabb const& rhs) const { return !(*this == rhs); }

  // clang-format on
 private:
  Real4 _min{0_r, 0_r, 0_r, 1_r};
  Real4 _max{0_r, 0_r, 0_r, 1_r};

#if MOCHI_USE_REFLECTION
  template <typename T>
  friend struct ::SReflectTypeTraits;
#endif // MOCHI_USE_REFLECTION
};

namespace details {
template <>
inline constexpr bool IsPrimitiveShapeDef<Aabb> = true;
} // namespace details

} // namespace mochi

/************************************************************************************
  Reflection support for Aabb.
  Serializes like struct with two fields of type Real3.
*/
#if MOCHI_USE_REFLECTION
template <>
struct SReflectTypeTraits<mochi::Aabb> {
  static constexpr SReflect::CoreType coreType = SReflect::CoreType::CT_struct;
  static SReflect::StructTypeInfo const& GetTypeInfo() {
    static auto const* s_typeInfo = []() {
      auto* ti = SReflect::MakeStructTypeInfo<mochi::Aabb>("mochi::Aabb");
      SReflect::detail::AddField(
          *ti, "min", offsetof(mochi::Aabb, _min), SReflect::GetTypeInfo<mochi::Real3>());
      SReflect::detail::AddField(
          *ti, "max", offsetof(mochi::Aabb, _max), SReflect::GetTypeInfo<mochi::Real3>());
      SReflect::FinalizeStruct(ti);
      return ti;
    }();
    return *s_typeInfo;
  }
};
#endif // MOCHI_USE_REFLECTION
