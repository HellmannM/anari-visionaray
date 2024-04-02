// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "array/ObjectArray.h"
#include "light/Light.h"
#include "surface/Surface.h"
#include "volume/Volume.h"
#include "VisionarayScene.h"

namespace visionaray {

struct Group : public Object
{
  Group(VisionarayGlobalState *s);
  ~Group() override;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint32_t flags) override;

  void commit() override;

  const std::vector<Surface *> &surfaces() const;
  const std::vector<Volume *> &volumes() const;
  const std::vector<Light *> &lights() const;

  // void intersectVolumes(VolumeRay &ray) const;

  void markCommitted() override;

  VisionarayScene visionarayScene() const;
  void visionaraySceneConstruct();
  void visionaraySceneCommit();

 private:
  void cleanup();

  // Geometry //

  helium::CommitObserverPtr<ObjectArray> m_surfaceData;
  std::vector<Surface *> m_surfaces;

  // Volume //

  helium::CommitObserverPtr<ObjectArray> m_volumeData;
  std::vector<Volume *> m_volumes;

  // Light //

  helium::CommitObserverPtr<ObjectArray> m_lightData;
  std::vector<Light *> m_lights;

  // BVH //

  struct ObjectUpdates
  {
    helium::TimeStamp lastSceneConstruction{0};
    helium::TimeStamp lastSceneCommit{0};
  } m_objectUpdates;

  VisionarayScene vscene{nullptr};
};

// box3 getEmbreeSceneBounds(RTCScene scene);

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Group *, ANARI_GROUP);
