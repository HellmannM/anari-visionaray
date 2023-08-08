// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"
#include "array/Array1D.h"
#include "scene/VisionarayScene.h"

namespace visionaray {

struct PRD
{
  int x, y;
};

struct PixelSample
{
  float4 color;
  float depth;
};

struct VisionarayRenderer
{
  VSNRAY_FUNC
  PixelSample renderSample(Ray ray, PRD &prd, unsigned worldID,
        VisionarayGlobalState::DeviceObjectRegistry onDevice) {
    PixelSample result;
    result.color = m_bgColor;
    result.depth = 1.f;

    auto hr = intersect(ray, onDevice.TLSs[worldID]);

    if (hr.hit) {
      vec3f gn(1.f,0.f,0.f);
      auto inst = onDevice.instances[hr.inst_id];
      const auto &geom = onDevice.groups[inst.groupID].geoms[hr.geom_id];
      if (geom.type == dco::Geometry::Triangle) {
      // TODO: doesn't work for instances yet
        auto tri = geom.asTriangle.data[hr.prim_id];
        gn = normalize(cross(tri.e1,tri.e2));
      }

      shade_record<float> sr;
      sr.normal = gn;
      sr.geometric_normal = gn;
      sr.view_dir = -ray.dir;
      sr.tex_color = float3(1.f);
      sr.light_dir = -ray.dir;
      sr.light_intensity = float3(1.f);

      // That doesn't work for instances..
      const auto &mat = onDevice.materials[hr.geom_id];
      float3 shadedColor = to_rgb(mat.asMatte.data.shade(sr));

      result.color = float4(float3(.8f)*dot(-ray.dir,gn),1.f);
      result.color = float4(shadedColor,1.f);
    }

    return result;
  }

  float4 m_bgColor{float3(0.f), 1.f};
  float m_ambientRadiance{1.f};
};

struct Renderer : public Object
{
  Renderer(VisionarayGlobalState *s);
  ~Renderer() override;

  virtual void commit() override;

  static Renderer *createInstance(
      std::string_view subtype, VisionarayGlobalState *d);

  VisionarayRenderer visionarayRenderer() const { return vrend; }

 private:
  VisionarayRenderer vrend;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Renderer *, ANARI_RENDERER);
