#pragma once

#include "renderer/common.h"
#include "renderer/DDA.h"
#include "DeviceCopyableObjects.h"

namespace visionaray {

VSNRAY_FUNC
inline float4 postClassify(ScreenSample &ss, dco::TransferFunction1D tf, float v) {
  box1 valueRange = tf.valueRange;
  v = (v - valueRange.min) / (valueRange.max - valueRange.min);
  float4 clr = tex1D(tf.sampler, v);
  // if (ss.debug()) {
  //   printf("v: %f, clr: (%f,%f,%f)\n",v,clr.x,clr.y,clr.z);
  // }
  return clr;

  return {};
}

VSNRAY_FUNC
inline float rayMarchVolume(ScreenSample &ss,
                            Ray ray,
                            const dco::Volume &vol,
                            float3 &color,
                            float &alpha) {
  float dt = vol.field.baseDT;
  auto boxHit = intersect(ray, vol.bounds);
  // if (ss.debug()) {
  //   printf("boxHit: %f,%f\n",boxHit.tnear,boxHit.tfar);
  //   print(ray);
  //   print(vol.bounds);
  // }
  float t=boxHit.tnear;
  for (;t<boxHit.tfar&&alpha<0.99f;t+=dt) {
    float3 P = ray.ori+ray.dir*t;
    float v = 0.f;
    if (sampleField(vol.field,P,v)) {
      float4 sample
          = postClassify(ss,vol.asTransferFunction1D,v);
      color += dt * (1.f-alpha) * sample.w * sample.xyz();
      alpha += dt * (1.f-alpha) * sample.w;
    }
  }
  return t;
}

struct HitRecordVolume
{
  bool hit{false};
  float t{FLT_MAX};
  float3 albedo{0.f,0.f,0.f};
  float extinction{0.f};
  float Tr{1.f};
  int geom_id{-1};
  int inst_id{-1};
};

VSNRAY_FUNC
inline HitRecordVolume sampleFreeFlightDistance(
    ScreenSample &ss, Ray ray, const dco::Volume &vol) {

  const auto &sf = vol.field;

  HitRecordVolume hr;

  dco::GridAccel grid = sf.gridAccel;

  auto woodcockFunc = [&](const int leafID, float t0, float t1) {

    const float majorant = grid.isValid() ? grid.maxOpacities[leafID] : 1.f;
    float t = t0;

    while (1) {
      if (majorant <= 0.f)
        break;

      t -= logf(1.f - ss.random()) / majorant;

      if (t >= t1)
        break;

      float3 P = ray.ori+ray.dir*t;
      float v = 0.f;
      if (sampleField(sf,P,v)) {
        float4 sample
            = postClassify(ss,vol.asTransferFunction1D,v);
        hr.albedo = sample.xyz();
        hr.extinction = sample.w;
        float u = ss.random();
        if (hr.extinction >= u * majorant) {
          hr.hit = true;
          hr.Tr = 0.f;
          hr.t = t;
          return false; // stop traversal
        }
      }
    }

    return true; // cont. traversal to the next spat. partition

  };

  // clip ray against volume bounds
  auto boxHit = intersect(ray, vol.bounds);
  ray.tmin = max(ray.tmin, boxHit.tnear);
  ray.tmax = min(ray.tmax, boxHit.tfar);

  // transform ray to voxel space
  ray.ori = sf.pointToVoxelSpace(ray.ori);
  ray.dir = sf.vectorToVoxelSpace(ray.dir);

  const float dt_scale = length(ray.dir);
  ray.dir = normalize(ray.dir);

  ray.tmin = ray.tmin * dt_scale;
  ray.tmax = ray.tmax * dt_scale;

  hr.t = ray.tmax;
  if (sf.gridAccel.isValid())
    dda3(ray, grid.dims, grid.worldBounds, woodcockFunc);
  else
    woodcockFunc(-1, ray.tmin, ray.tmax);

  if (hr.hit)
    hr.t /= dt_scale;

  return hr;
}

VSNRAY_FUNC
inline HitRecordVolume sampleFreeFlightDistanceAllVolumes(
    ScreenSample &ss, Ray ray, unsigned worldID,
    VisionarayGlobalState::DeviceObjectRegistry onDevice) {
  // find closest distance across all volumes
  HitRecordVolume result;
  result.t = ray.tmax;

  while (true) {
    auto hrv = intersectVolumes(ray, onDevice.TLSs[worldID]);
    if (!hrv.hit)
      break;
    const auto &inst = onDevice.instances[hrv.inst_id];
    const auto &group = onDevice.groups[inst.groupID];
    const auto &geom = onDevice.geometries[group.geoms[hrv.geom_id]];
    const auto &vol = geom.as<dco::Volume>(0);
    HitRecordVolume hr = sampleFreeFlightDistance(ss, ray, vol);
    if (hr.t < result.t) {
      result = hr;
      result.geom_id = hrv.geom_id;
      result.inst_id = hrv.inst_id;
    }
    auto boxHit = intersect(ray, vol.bounds);
    float eps = epsilonFrom(ray.ori+boxHit.tfar*ray.dir,
                            ray.dir,
                            boxHit.tfar);
    ray.tmin = boxHit.tfar + eps;
  }

  return result;
}

} // namespace visionaray
