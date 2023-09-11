// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Material.h"
// subtypes
#include "Matte.h"

namespace visionaray {

Material::Material(VisionarayGlobalState *s) : Object(ANARI_MATERIAL, s)
{
  vmat.matID = s->objectCounts.materials++;
}

Material::~Material()
{
  deviceState()->objectCounts.materials--;
}

Material *Material::createInstance(
    std::string_view subtype, VisionarayGlobalState *s)
{
  if (subtype == "matte")
    return new Matte(s);
  else
    return (Material *)new UnknownObject(ANARI_MATERIAL, s);
}

void Material::commit()
{
  // m_alphaMode = alphaModeFromString(getParamString("alphaMode", "opaque"));
  // m_alphaCutoff = getParam<float>("alphaCutoff", 0.5f);
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Material *);
