// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "StructuredRegularField.h"
// std
#include <limits>

namespace visionaray {

StructuredRegularField::StructuredRegularField(VisionarayGlobalState *d)
    : SpatialField(d)
{
  vfield.type = dco::SpatialField::StructuredRegular;
}

void StructuredRegularField::commit()
{
  m_dataArray = getParamObject<Array3D>("data");

  if (!m_dataArray) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'data' on 'structuredRegular' field");
    return;
  }

//m_data = m_dataArray->data();
  m_type = m_dataArray->elementType();
  m_dims = m_dataArray->size();

  m_origin = getParam<float3>("origin", float3(0.f));
  m_spacing = getParam<float3>("spacing", float3(1.f));

  setStepSize(min_element(m_spacing / 2.f));

  m_dataTexture = texture<float, 3>(m_dims.x, m_dims.y, m_dims.z);
  if (m_type == ANARI_UFIXED8) {
    std::vector<float> data(m_dims.x * size_t(m_dims.y) * m_dims.z);
    auto data8 = (uint8_t *)m_dataArray->data();
    for (size_t i=0; i<data.size(); ++i) {
      data[i] = data8[i] / 255.f;
    }
    m_dataTexture.reset(data.data());
  } else if (m_type == ANARI_UFIXED16) {
    std::vector<float> data(m_dims.x * size_t(m_dims.y) * m_dims.z);
    auto data16 = (uint16_t *)m_dataArray->data();
    for (size_t i=0; i<data.size(); ++i) {
      data[i] = data16[i] / 65535.f;
    }
    m_dataTexture.reset(data.data());
  } else if (m_type == ANARI_FLOAT32)
    m_dataTexture.reset((float *)m_dataArray->data());

  m_dataTexture.set_filter_mode(Linear);
  m_dataTexture.set_address_mode(Clamp);

  vfield.asStructuredRegular.origin = m_origin;
  vfield.asStructuredRegular.spacing = m_spacing;
  vfield.asStructuredRegular.dims = m_dims;
  vfield.asStructuredRegular.sampler = texture_ref<float, 3>(m_dataTexture);

  buildGrid();

  dispatch();
}

bool StructuredRegularField::isValid() const
{
  return m_dataArray;
}

aabb StructuredRegularField::bounds() const
{
  return aabb(m_origin, m_origin + ((float3(m_dims) - 1.f) * m_spacing));
}

void StructuredRegularField::buildGrid()
{
  int3 gridDims{16, 16, 16};
  box3f worldBounds = {bounds().min,bounds().max};
  m_gridAccel.init(gridDims, worldBounds);

  for (unsigned z=0; z<m_dims.z; ++z) {
    for (unsigned y=0; y<m_dims.y; ++y) {
      for (unsigned x=0; x<m_dims.x; ++x) {
        float3 texCoord = (float3{x,y,z}+float3{0.5f})/float3(m_dims);
        float value = tex3D(vfield.asStructuredRegular.sampler, texCoord);
        box3f cellBounds{
          m_origin+float3{x,y,z}*m_spacing,
          m_origin+float3{x+1,y+1,z+1}*m_spacing
        };

        const vec3i loMC = projectOnGrid(cellBounds.min,gridDims,worldBounds);
        const vec3i upMC = projectOnGrid(cellBounds.max,gridDims,worldBounds);

        for (int mcz=loMC.z; mcz<=upMC.z; ++mcz) {
          for (int mcy=loMC.y; mcy<=upMC.y; ++mcy) {
            for (int mcx=loMC.x; mcx<=upMC.x; ++mcx) {
              const vec3i mcID(mcx,mcy,mcz);
              updateMC(mcID,gridDims,value,m_gridAccel.valueRanges());
            }
          }
        }
      }
    }
  }
}

} // namespace visionaray
