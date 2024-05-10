
#pragma once

#include "Material.h"

namespace visionaray {

struct PBM : public Material
{
  PBM(VisionarayGlobalState *s);
  void commit() override;

 private:
  struct {
    float4 value{1.f, 1.f, 1.f, 1.f};
    helium::IntrusivePtr<Sampler> sampler;
    dco::Attribute attribute;
  } m_baseColor;

  struct {
    float value{1.f};
    helium::IntrusivePtr<Sampler> sampler;
    dco::Attribute attribute;
  } m_opacity, m_metallic, m_roughness, m_clearcoat, m_clearcoatRoughness;

  float m_ior{1.5f};

  struct {
    helium::IntrusivePtr<Sampler> sampler;
    //float scale{1.f};
  } m_normal;

  dco::AlphaMode m_alphaMode{dco::AlphaMode::Opaque};
  float m_alphaCutoff{0.5f};
};

} // namespace visionaray
