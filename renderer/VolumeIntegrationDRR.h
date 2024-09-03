#pragma once

#include "renderer/common.h"
#include "renderer/DDA.h"
#include "DeviceCopyableObjects.h"

namespace visionaray {

VSNRAY_FUNC
inline float rayMarchVolumeDRR(ScreenSample &ss,
                            Ray ray,
                            const dco::Volume &vol,
                            float3 &color,
                            float &alpha) {
  constexpr float photon_energy = 13500.f;
  constexpr float depth_accum_dist_mm = 5.f;
  constexpr float min_contribution = 0.6f;
  constexpr float min_intensity = 0.5f;

  float dt = vol.field.baseDT;
  auto boxHit = intersect(ray, vol.bounds);

  // render
  float v_max = 0.f;
  float t_at_v_max = 0.f;
  float lac_accumulated = 0.f;
  size_t steps = 0;
  float t=boxHit.tnear;
  for (; t<boxHit.tfar; t+=dt) {
    float3 P = ray.ori+ray.dir*t;
    float v = 0.f;
    if (sampleField(vol.field,P,v)) {
      lac_accumulated += v;
      if (v > v_max) {
        v_max = v;
        t_at_v_max = t;
      }
      ++steps;
    }
  }
  auto lac_averaged = lac_accumulated / steps;
  auto dist_cm = (steps * dt) / 10.f; //TODO assuming dt is in [mm]
  auto remaining = pow(photon_energy, -dist_cm * lac_averaged);
  color = float3(1.f - remaining);
  alpha = 1.f;

  // get depth
  if (color.x < min_intensity)
    return 0.f;
  const float start = max(t_at_v_max - depth_accum_dist_mm / 2.f, boxHit.tnear);
  const float end   = min(t_at_v_max + depth_accum_dist_mm / 2.f, boxHit.tfar);
  auto t2 = start;
  float section_lac_accumulated = 0.f;
  size_t section_steps = 0;
  for (; t2<end; t2+=dt) {
    float3 P = ray.ori+ray.dir*t2;
    float v = 0.f;
    if (sampleField(vol.field,P,v)) {
      section_lac_accumulated += v;
      ++section_steps;
    }
  }
  auto section_lac_averaged = section_lac_accumulated / section_steps;
  auto section_dist_cm = (section_steps * dt) / 10.f; //TODO assuming dt is in [mm]
  auto section_remaining = pow(photon_energy, -section_dist_cm * section_lac_averaged);
  auto contribution = (1.f - section_remaining) / (1.f - remaining);
  if (contribution >= min_contribution)
    return t_at_v_max;
  return 0.f;
}

} // namespace visionaray
