#pragma once

#include "RenderingSemaphore.h"
// helium
#include "helium/BaseGlobalDeviceState.h"
// visionaray
#include "visionaray/detail/thread_pool.h"
// ours
#include "DeviceCopyableObjects.h"
#include "DeviceObjectArray.h"

namespace visionaray {

struct Frame;

struct VisionarayGlobalState : public helium::BaseGlobalDeviceState
{
  thread_pool threadPool;

  struct ObjectCounts
  {
    size_t frames{0};
    size_t cameras{0};
    size_t renderers{0};
    size_t worlds{0};
    size_t instances{0};
    size_t groups{0};
    size_t surfaces{0};
    size_t geometries{0};
    size_t materials{0};
    size_t samplers{0};
    size_t volumes{0};
    size_t spatialFields{0};
    size_t lights{0};
    size_t arrays{0};
    size_t unknown{0};
  } objectCounts;

  struct ObjectUpdates
  {
    helium::TimeStamp lastBLSReconstructSceneRequest{0};
    helium::TimeStamp lastBLSCommitSceneRequest{0};
    helium::TimeStamp lastTLSReconstructSceneRequest{0};
  } objectUpdates;

  struct DeviceCopyableObjects
  {
    // One TLS per world
    DeviceObjectArray<dco::TLS> TLSs;
    DeviceObjectArray<dco::Group> groups;
    DeviceObjectArray<dco::Surface> surfaces;
    DeviceObjectArray<dco::Instance> instances;
    DeviceObjectArray<dco::Sampler> samplers;
    DeviceObjectArray<dco::SpatialField> spatialFields;
    DeviceObjectArray<dco::GridAccel> gridAccels;
    DeviceObjectArray<dco::TransferFunction> transferFunctions;
    DeviceObjectArray<dco::Light> lights;
    DeviceObjectArray<dco::Frame> frames;
  } dcos;

  struct DeviceObjectRegistry
  {
    dco::TLS *TLSs{nullptr};
    dco::Group *groups{nullptr};
    dco::Surface *surfaces{nullptr};
    dco::Instance *instances{nullptr};
    dco::Sampler *samplers{nullptr};
    dco::SpatialField *spatialFields{nullptr};
    dco::GridAccel *gridAccels{nullptr};
    dco::TransferFunction *transferFunctions{nullptr};
    dco::Light *lights{nullptr};
    dco::Frame *frames{nullptr};
    unsigned numLights{0};
  } onDevice;

  RenderingSemaphore renderingSemaphore;
  Frame *currentFrame{nullptr};

  // Helper methods //

  VisionarayGlobalState(ANARIDevice d);
  void waitOnCurrentFrame() const;
};

// Helper functions/macros ////////////////////////////////////////////////////

inline VisionarayGlobalState *asVisionarayState(helium::BaseGlobalDeviceState *s)
{
  return (VisionarayGlobalState *)s;
}

#define VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(type, anari_type)              \
  namespace anari {                                                            \
  ANARI_TYPEFOR_SPECIALIZATION(type, anari_type);                              \
  }

#define VISIONARAY_ANARI_TYPEFOR_DEFINITION(type)                              \
  namespace anari {                                                            \
  ANARI_TYPEFOR_DEFINITION(type);                                              \
  }

} // visionaray
