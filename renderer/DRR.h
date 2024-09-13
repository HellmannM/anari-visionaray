#pragma once

#include "Renderer.h"

namespace visionaray {

struct DRR : public Renderer
{
  DRR(VisionarayGlobalState *s);
  ~DRR() override;
};

} // visionaray
