// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "array/Array3D.h"

namespace visionaray {

Array3D::Array3D(VisionarayGlobalState *state, const Array3DMemoryDescriptor &d)
    : Array(ANARI_ARRAY3D, state, d)
{
  m_size[0] = d.numItems1;
  m_size[1] = d.numItems2;
  m_size[2] = d.numItems3;

  initManagedMemory();
}

size_t Array3D::totalSize() const
{
  return size(0) * size(1) * size(2);
}

size_t Array3D::size(int dim) const
{
  return m_size[dim];
}

uint3 Array3D::size() const
{
  return uint3(uint32_t(size(0)), uint32_t(size(1)), uint32_t(size(2)));
}

void Array3D::privatize()
{
  makePrivatizedCopy(size(0) * size(1) * size(2));
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Array3D *);
