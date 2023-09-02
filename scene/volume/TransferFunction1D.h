// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

// visionaray
#include "visionaray/texture/texture.h"
// ours
#include "Volume.h"
#include "array/Array1D.h"
#include "spatial_field/SpatialField.h"

namespace visionaray {

struct TransferFunction1D : public Volume
{
  TransferFunction1D(VisionarayGlobalState *d);

  void commit() override;

  bool isValid() const override;

  aabb bounds() const override;

 private:
   void dispatch();

  // Data //

  helium::IntrusivePtr<SpatialField> m_field;

  aabb m_bounds;

  box1 m_valueRange{0.f, 1.f};
  float m_densityScale{1.f};

  helium::IntrusivePtr<Array1D> m_colorData;
  helium::IntrusivePtr<Array1D> m_opacityData;

  texture<float4, 1> transFuncTexture;
};

} // namespace visionaray
