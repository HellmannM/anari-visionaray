// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

// visionaray
#ifdef WITH_CUDA
#include <visionaray/texture/cuda_texture.h>
#elif defined(WITH_HIP)
#include <visionaray/texture/hip_texture.h>
#endif
// ours
#include "TransferFunction1D.h"

namespace visionaray {

TransferFunction1D::TransferFunction1D(VisionarayGlobalState *d)
  : Volume(d)
  , m_field(this)
  , m_colorData(this)
  , m_opacityData(this)
{
  vgeom.type = dco::Geometry::Volume;
  vgeom.geomID = deviceState()->dcos.geometries.alloc(vgeom);
}

TransferFunction1D::~TransferFunction1D()
{
  deviceState()->dcos.geometries.free(vgeom.geomID);
}

void TransferFunction1D::commit()
{
  Volume::commit();

  m_field = getParamObject<SpatialField>("value");

  if (!m_field) {
    // Some apps might still use "field" (from the provisional specs)
    m_field = getParamObject<SpatialField>("field");
  }

  if (!m_field) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "no spatial field provided to transferFunction1D volume");
    return;
  }

  if (!m_field->isValid()) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "invalid spatial field provided to transferFunction1D volume");
    return;
  }

  m_bounds = m_field->bounds();

  m_valueRange = getParam<box1>("valueRange", box1(0.f, 1.f));

  m_colorData = getParamObject<Array1D>("color");
  m_opacityData = getParamObject<Array1D>("opacity");
  float densityScale = 1.f; // old, some apps may still use this!
  if (getParam("densityScale", ANARI_FLOAT32, &densityScale))
    m_unitDistance = densityScale;
  else
    m_unitDistance = getParam<float>("unitDistance", 1.f);

  if (!m_colorData) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "no color data provided to transferFunction1D volume");
    return;
  }

  if (!m_opacityData) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "no opacity data provided to transfer function");
    return;
  }

  auto *colorData = m_colorData->beginAs<vec3>();
  auto *opacityData = m_opacityData->beginAs<float>();

  size_t tfSize = max(m_colorData->size(), m_opacityData->size());

  std::vector<float4> tf(tfSize);
  for (size_t i=0; i<tfSize; ++i) {
    float colorPos = tfSize > 1 ? (float(i)/(tfSize-1))*(m_colorData->size()-1) : 0.f;
    float colorFrac = colorPos-floorf(colorPos);

    vec3f color0 = colorData[int(floorf(colorPos))];
    vec3f color1 = colorData[int(ceilf(colorPos))];
    vec3f color = lerp(color0, color1, colorFrac);

    float alphaPos = tfSize > 1 ? (float(i)/(tfSize-1))*(m_opacityData->size()-1) : 0.f;
    float alphaFrac = alphaPos-floorf(alphaPos);

    float alpha0 = opacityData[int(floorf(alphaPos))];
    float alpha1 = opacityData[int(ceilf(alphaPos))];
    float alpha = lerp(alpha0, alpha1, alphaFrac);

    tf[i] = vec4(color, alpha);
  }
#if defined(WITH_CUDA) || defined(WITH_HIP)
  texture<float4, 1> tex(tf.size());
#else
  transFuncTexture = texture<float4, 1>(tf.size());
  auto &tex = transFuncTexture;
#endif
  tex.reset(tf.data());
  tex.set_filter_mode(Linear);
  tex.set_address_mode(Clamp);

  m_volume.resize(1);

  m_volume[0].type = dco::Volume::TransferFunction1D;
  m_volume[0].bounds = m_bounds;
  m_volume[0].volID = m_field->visionaraySpatialField().fieldID;
  m_volume[0].field = m_field->visionaraySpatialField();
  m_volume[0].unitDistance = m_unitDistance;

  m_volume[0].asTransferFunction1D.numValues = tex.size()[0];
  m_volume[0].asTransferFunction1D.valueRange = m_valueRange;
#ifdef WITH_CUDA
  transFuncTexture = cuda_texture<float4, 1>(tex);
  m_volume[0].asTransferFunction1D.sampler
      = cuda_texture_ref<float4, 1>(transFuncTexture);
#elif defined(WITH_HIP)
  transFuncTexture = hip_texture<float4, 1>(tex);
  m_volume[0].asTransferFunction1D.sampler
      = hip_texture_ref<float4, 1>(transFuncTexture);
#else
  m_volume[0].asTransferFunction1D.sampler
      = texture_ref<float4, 1>(transFuncTexture);
#endif

  vgeom.primitives.data = m_volume.devicePtr();
  vgeom.primitives.len = m_volume.size();

  // Trigger a BVH rebuild:
  lastUpdateRequest = helium::newTimeStamp();

  dispatch();

#if !defined(WITH_CUDA) && !defined(WITH_HIP)
  m_field->gridAccel().computeMaxOpacities(m_volume[0].asTransferFunction1D);
#endif
}

void TransferFunction1D::markCommitted()
{
  Object::markCommitted();
  deviceState()->objectUpdates.lastBLSCommitSceneRequest =
      helium::newTimeStamp();
}

bool TransferFunction1D::isValid() const
{
  return m_field && m_field->isValid() && m_colorData && m_opacityData;
}

aabb TransferFunction1D::bounds() const
{
  return m_bounds;
}

void TransferFunction1D::dispatch()
{
  deviceState()->dcos.geometries.update(vgeom.geomID, vgeom);

  // Upload/set accessible pointers
  deviceState()->onDevice.geometries = deviceState()->dcos.geometries.devicePtr();
}

} // namespace visionaray
