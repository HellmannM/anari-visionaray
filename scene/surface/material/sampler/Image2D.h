
#pragma once

#include "Sampler.h"
#include "array/Array2D.h"

namespace visionaray {

struct Image2D : public Sampler
{
  Image2D(VisionarayGlobalState *d);

  bool isValid() const override;
  void commit() override;

  // float4 getSample(const Geometry &g, const Ray &r) const override;

 private:
  void updateImageData();

  helium::IntrusivePtr<Array2D> m_image;
  dco::Attribute m_inAttribute{dco::Attribute::None};
  tex_address_mode m_wrapMode1{Clamp};
  tex_address_mode m_wrapMode2{Clamp};
  bool m_linearFilter{true};
  mat4 m_inTransform{mat4::identity()};
  float4 m_inOffset{0.f, 0.f, 0.f, 0.f};
  mat4 m_outTransform{mat4::identity()};
  float4 m_outOffset{0.f, 0.f, 0.f, 0.f};

  texture<vector<4, unorm<8>>, 2> vimage;
};

} // namespace visionaray
