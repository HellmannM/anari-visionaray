#pragma once

#include "Object.h"
#include "array/Array1D.h"
#include "scene/volume/spatial_field/SpatialField.h"
#include "scene/volume/Volume.h"
#include "scene/VisionarayScene.h"

namespace visionaray {

struct PRD
{
  int x, y;
  uint2 frameSize;
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

    auto debug = [=]() {
      return prd.x == prd.frameSize.x/2 && prd.y == prd.frameSize.y/2;
    };

    PixelSample result;
    result.color = m_bgColor;
    result.depth = 1.f;

    if (onDevice.TLSs[worldID].num_primitives() == 0)
      return result; // happens eg with TLSs of unsupported objects

    auto hr = intersect(ray, onDevice.TLSs[worldID]);

    if (hr.hit) {
      auto inst = onDevice.instances[hr.inst_id];
      const auto &geom = onDevice.groups[inst.groupID].geoms[hr.geom_id];

      // TODO: currently, this will arbitrarily pick a volume _or_
      // surface BVH if both are present and do overlap
      if (geom.type == dco::Geometry::Volume) {
        const auto &vol = geom.asVolume.data;
        auto boxHit = intersect(ray, vol.bounds);
        float dt = onDevice.spatialFields[vol.fieldID].baseDT;
        float3 color(0.f);
        float alpha = 0.f;
        for (float t=boxHit.tnear;t<boxHit.tfar&&alpha<0.99f;t+=dt) {
          float3 P = ray.ori+ray.dir*t;
          float v = 0.f;
          if (sampleField(onDevice.spatialFields[vol.fieldID],P,v)) {
            float4 sample = postClassify(vol,v,debug());
            color += dt * (1.f-alpha) * sample.w * sample.xyz();
            alpha += dt * (1.f-alpha) * sample.w;
          }
        }
        result.color = float4(color,1.f);
      } else {
        vec3f gn(1.f,0.f,0.f);
        // TODO: doesn't work for instances yet
        if (geom.type == dco::Geometry::Triangle) {
          auto tri = geom.asTriangle.data[hr.prim_id];
          gn = normalize(cross(tri.e1,tri.e2));
        } else if (geom.type == dco::Geometry::Sphere) {
          auto sph = geom.asSphere.data[hr.prim_id];
          vec3f hitPos = ray.ori + hr.t * ray.dir;
          gn = normalize((hitPos-sph.center) / sph.radius);
        } else if (geom.type == dco::Geometry::Cylinder) {
          auto cyl = geom.asCylinder.data[hr.prim_id];
          vec3f hitPos = ray.ori + hr.t * ray.dir;
          vec3f axis = normalize(cyl.v2-cyl.v1);
          if (length(hitPos-cyl.v1) < cyl.radius)
            gn = -axis;
          else if (length(hitPos-cyl.v2) < cyl.radius)
            gn = axis;
          else {
            float t = dot(hitPos-cyl.v1, axis);
            vec3f pt = cyl.v1 + t * axis;
            gn = normalize(hitPos-pt);
          }
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
    }

    if (prd.x == prd.frameSize.x/2 || prd.y == prd.frameSize.y/2) {
      result.color = float4(1.f) - result.color;
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
