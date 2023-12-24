// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "DeviceArray.h"
#include "Geometry.h"

namespace visionaray {

struct Triangle : public Geometry
{
  Triangle(VisionarayGlobalState *s);

  void commit() override;

  //float4 getAttributeValue(
  //    const Attribute &attr, const Ray &ray) const override;

 private:
  void cleanup();

  HostDeviceArray<basic_triangle<3, float>> m_triangles;
  helium::IntrusivePtr<Array1D> m_index;
  helium::IntrusivePtr<Array1D> m_vertexPosition;
  std::array<helium::IntrusivePtr<Array1D>, 5> m_vertexAttributes;

  HostDeviceArray<uint3> vindex;
  HostDeviceArray<uint8_t> vattributes[5];
};

} // namespace visionaray
