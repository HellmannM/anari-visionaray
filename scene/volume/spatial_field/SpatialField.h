// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"

namespace visionaray {

struct SpatialField : public Object
{
  SpatialField(VisionarayGlobalState *d);
  virtual ~SpatialField();
  static SpatialField *createInstance(
      std::string_view subtype, VisionarayGlobalState *d);

//  virtual float sampleAt(const float3 &coord) const = 0;
//
  virtual aabb bounds() const = 0;
//
//  float stepSize() const;
//
// protected:
//  void setStepSize(float size);
//
// private:
//  float m_stepSize{0.f};
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(
    visionaray::SpatialField *, ANARI_SPATIAL_FIELD);
