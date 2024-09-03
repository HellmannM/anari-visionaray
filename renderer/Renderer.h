#pragma once

#include "DeviceArray.h"
#include "Object.h"
#include "array/Array1D.h"
#include "scene/volume/spatial_field/SpatialField.h"
#include "scene/volume/Volume.h"
// impls
#include "Raycast_impl.h"
#include "DirectLight_impl.h"

namespace visionaray {

struct VisionarayRenderer
{
  enum Type { Raycast, DirectLight, };
  Type type;

  void renderFrame(const dco::Frame &frame,
                   const dco::Camera &cam,
                   uint2 size,
                   VisionarayGlobalState *state,
                   const VisionarayGlobalState::DeviceObjectRegistry &DD,
                   unsigned worldID, int frameID)
  {
    auto start = std::chrono::system_clock::now();
    if (type == Raycast) {
      asRaycast.renderer.renderFrame(
          frame, cam, size, state, DD, rendererState, worldID, frameID);
    } else if (type == DirectLight) {
      asDirectLight.renderer.renderFrame(
          frame, cam, size, state, DD, rendererState, worldID, frameID);
    }
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - start).count();
    //std::cout << "elapsed: " << time << "ms" << std::endl;
    std::cout << ", " << time << "ms" << std::flush;
  }

  VSNRAY_FUNC
  constexpr bool stochasticRendering() const {
    if (type == Raycast) {
      return asRaycast.renderer.stochasticRendering;
    } else if (type == DirectLight) {
      return asDirectLight.renderer.stochasticRendering;
    }
    return type != Raycast;
  }

  VSNRAY_FUNC
  bool taa() const {
    return type == DirectLight && rendererState.taaEnabled;
  }

  struct {
    VisionarayRendererRaycast renderer;
  } asRaycast;

  struct {
    VisionarayRendererDirectLight renderer;
  } asDirectLight;

  RendererState rendererState;
};

struct Renderer : public Object
{
  Renderer(VisionarayGlobalState *s);
  ~Renderer() override;

  virtual void commit() override;

  static Renderer *createInstance(
      std::string_view subtype, VisionarayGlobalState *d);

  VisionarayRenderer &visionarayRenderer() { return vrend; }
  const VisionarayRenderer &visionarayRenderer() const { return vrend; }

  bool stochasticRendering() const { return vrend.stochasticRendering(); }

 protected:
  helium::ChangeObserverPtr<Array1D> m_clipPlanes;
  HostDeviceArray<float4> m_clipPlanesOnDevice;
  VisionarayRenderer vrend;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Renderer *, ANARI_RENDERER);
