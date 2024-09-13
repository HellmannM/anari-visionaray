#pragma once

#include "renderer/common.h"
#include "renderer/VolumeIntegrationDRR.h"

namespace visionaray {

struct VisionarayRendererDRR
{
  void renderFrame(const dco::Frame &frame,
                   const dco::Camera &cam,
                   uint2 size,
                   VisionarayGlobalState *state,
                   const VisionarayGlobalState::DeviceObjectRegistry &DD,
                   const RendererState &rendererState,
                   unsigned worldID, int frameID);

  constexpr static bool stochasticRendering{false};
  constexpr static bool supportsTaa{false};
};

} // namespace visionaray
