// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Material.h"

namespace visionaray {

struct Matte : public Material
{
  Matte(VisionarayGlobalState *s);
  void commit() override;
};

} // namespace visionaray
