#include "DRR.h"

namespace visionaray {

DRR::DRR(VisionarayGlobalState *s) : Renderer(s)
{
  vrend.type = VisionarayRenderer::DRR;
}

DRR::~DRR()
{}

} // namespace visionaray
