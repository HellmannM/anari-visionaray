#pragma once

#include "renderer/common.h"
#include "renderer/DDA.h"
#include "DeviceCopyableObjects.h"

namespace visionaray {

template<typename T, size_t numElements>
struct MovingAccumBuffer
{
  VSNRAY_FUNC
  MovingAccumBuffer()
    : currentIndex{0}, sum{0.f}, buffer{std::move(std::array<T, numElements>())} {}
  
  VSNRAY_FUNC
  T update(T newValue)
  {
    sum = sum - buffer[currentIndex] + newValue;
    buffer[currentIndex] = newValue;
    // slow if numElements is not power of 2
    currentIndex = (currentIndex + 1) % numElements;
    return sum;
  }

  size_t currentIndex;
  T sum;
  std::array<T, numElements> buffer;
};

VSNRAY_FUNC
inline float rayMarchVolumeDRR(ScreenSample &ss,
                            Ray ray,
                            const dco::Volume &vol,
                            float3 &color,
                            float &alpha,
                            float /*photon_energy*/) {
  constexpr size_t accumBufferSize{32};
  constexpr float min_contribution = 0.4f;
  constexpr float min_intensity = 0.4f;
  constexpr float max_intensity = 0.99f;
  const float cutoff = - std::log(1.f - max_intensity);

  auto boxHit = intersect(ray, vol.bounds);

  const auto &sf = vol.field;

  ray.tmin = max(ray.tmin, boxHit.tnear);
  ray.tmax = min(ray.tmax, boxHit.tfar);

  // transform ray to voxel space
  ray.ori = sf.pointToVoxelSpace(ray.ori);
  ray.dir = sf.vectorToVoxelSpace(ray.dir);

  const float dt_scale = length(ray.dir);
  ray.dir = normalize(ray.dir);

  ray.tmin = ray.tmin * dt_scale;
  ray.tmax = ray.tmax * dt_scale;
  const float dt = dt_scale / vol.unitDistance;
  const float dt_cm = dt / dt_scale / 10.f; // dt is in [mm]

  MovingAccumBuffer<float, accumBufferSize> accum;
  float sectionMax{0.f};
  float tAtSectionMax{-FLT_MAX};

  // render
  float lac_accumulated = 0.f;
  size_t steps = 0;
  for (float t=ray.tmin; t<ray.tmax; t+=dt) {
    float3 P = ray.ori + ray.dir * t;
    float v = 0.f;
    if (sampleField(sf, P, v)) {
      lac_accumulated += v;
      const auto section = accum.update(v);
      if (section > sectionMax) {
        sectionMax = section;
        tAtSectionMax = t;
      }
      if (dt_cm * lac_accumulated > cutoff)
        break;
      ++steps;
    }
  }
  auto remaining = exp(- dt_cm * lac_accumulated);
  color = float3(1.f - remaining);
  alpha = 1.f;

  // get depth
  if (color.x < min_intensity)
    return -FLT_MAX;
  // return center of buffer
  if ((sectionMax / lac_accumulated) > min_contribution)
    return (tAtSectionMax - (accumBufferSize / 2.f * dt)) / dt_scale;
  return -FLT_MAX;
}

} // namespace visionaray
