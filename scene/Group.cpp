// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Group.h"
// std
#include <iterator>

namespace visionaray {

Group::Group(VisionarayGlobalState *s) : Object(ANARI_GROUP, s)
{
  s->objectCounts.groups++;
}

Group::~Group()
{
  cleanup();
  deviceState()->objectCounts.groups--;
}

bool Group::getProperty(
    const std::string_view &name, ANARIDataType type, void *ptr, uint32_t flags)
{
  if (name == "bounds" && type == ANARI_FLOAT32_BOX3) {
    if (flags & ANARI_WAIT) {
      deviceState()->waitOnCurrentFrame();
      deviceState()->commitBufferFlush();
      visionaraySceneConstruct();
      visionaraySceneCommit();
    }
    // auto bounds = getEmbreeSceneBounds(m_embreeScene);
    // for (auto *v : volumes()) {
    //   if (v->isValid())
    //     bounds.extend(v->bounds());
    // }
    // std::memcpy(ptr, &bounds, sizeof(bounds));
    return true;
  }

  return Object::getProperty(name, type, ptr, flags);
}

void Group::commit()
{
  cleanup();

  m_surfaceData = getParamObject<ObjectArray>("surface");
  m_volumeData = getParamObject<ObjectArray>("volume");

  if (m_surfaceData)
    m_surfaceData->addCommitObserver(this);
  if (m_volumeData) {
    m_volumeData->addCommitObserver(this);
    std::transform(m_volumeData->handlesBegin(),
        m_volumeData->handlesEnd(),
        std::back_inserter(m_volumes),
        [](Object *o) { return (Volume *)o; });
  }
}

const std::vector<Surface *> &Group::surfaces() const
{
  return m_surfaces;
}

const std::vector<Volume *> &Group::volumes() const
{
  return m_volumes;
}

// void Group::intersectVolumes(VolumeRay &ray) const
// {
//   Volume *originalVolume = ray.volume;
//   box1 t = ray.t;
//
//   for (auto *v : volumes()) {
//     if (!v->isValid())
//       continue;
//     const box3 bounds = v->bounds();
//     const float3 mins = (bounds.lower - ray.org) * (1.f / ray.dir);
//     const float3 maxs = (bounds.upper - ray.org) * (1.f / ray.dir);
//     const float3 nears = linalg::min(mins, maxs);
//     const float3 fars = linalg::max(mins, maxs);
//
//     const box1 lt(linalg::maxelem(nears), linalg::minelem(fars));
//
//     if (lt.lower < lt.upper && (!ray.volume || lt.lower < t.lower)) {
//       t.lower = clamp(lt.lower, t);
//       t.upper = clamp(lt.upper, t);
//       ray.volume = v;
//     }
//   }
//
//   if (ray.volume != originalVolume)
//     ray.t = t;
// }

void Group::markCommitted()
{
  Object::markCommitted();
  deviceState()->objectUpdates.lastBLSReconstructSceneRequest =
      helium::newTimeStamp();
}

VisionarayScene Group::visionarayScene() const
{
  return vscene;
}

void Group::visionaraySceneConstruct()
{
  const auto &state = *deviceState();
  if (m_objectUpdates.lastSceneConstruction
      > state.objectUpdates.lastBLSReconstructSceneRequest)
    return;

  reportMessage(
      ANARI_SEVERITY_DEBUG, "visionaray::Group rebuilding embree scene");

  if (vscene)
    vscene->release();
  vscene = newVisionarayScene(VisionaraySceneImpl::Group, deviceState());

  uint32_t id = 0;
  if (m_surfaceData) {
    std::for_each(m_surfaceData->handlesBegin(),
        m_surfaceData->handlesEnd(),
        [&](Object *o) {
          auto *s = (Surface *)o;
          if (s && s->isValid()) {
            m_surfaces.push_back(s);
            vscene->attachGeometry(s->geometry()->visionarayGeometry(),
                s->material()->visionarayMaterial(),
                id++);
          } else {
            reportMessage(ANARI_SEVERITY_DEBUG,
                "visionaray::Group rejecting invalid surface(%p) in building BLS",
                s);
            auto *g = s->geometry();
            if (!g || !g->isValid()) {
              reportMessage(
                  ANARI_SEVERITY_DEBUG, "    visionaray::Geometry is invalid");
            }
            auto *m = s->material();
            if (!m || !m->isValid()) {
              reportMessage(
                  ANARI_SEVERITY_DEBUG, "    visionaray::Material is invalid");
            }
          }
        });
  }

  if (m_volumeData) {
    std::for_each(m_volumeData->handlesBegin(),
        m_volumeData->handlesEnd(),
        [&](Object *o) {
          auto *v = (Volume *)o;
          if (v && v->isValid()) {
            m_volumes.push_back(v);
            vscene->attachGeometry(v->visionarayGeometry(), id++);
          } else {
            reportMessage(ANARI_SEVERITY_DEBUG,
                "visionaray::Group rejecting invalid volume(%p) in building BLS",
                v);
          }
        });
  }

  m_objectUpdates.lastSceneConstruction = helium::newTimeStamp();
  m_objectUpdates.lastSceneCommit = 0;
  visionaraySceneCommit();
}

void Group::visionaraySceneCommit()
{
  const auto &state = *deviceState();
  if (m_objectUpdates.lastSceneCommit
      > state.objectUpdates.lastBLSCommitSceneRequest)
    return;

  reportMessage(
      ANARI_SEVERITY_DEBUG, "visionaray::Group committing embree scene");

  vscene->commit();
  m_objectUpdates.lastSceneCommit = helium::newTimeStamp();
}

void Group::cleanup()
{
  if (m_surfaceData)
    m_surfaceData->removeCommitObserver(this);
  if (m_volumeData)
    m_volumeData->removeCommitObserver(this);

  m_surfaces.clear();
  m_volumes.clear();

  m_objectUpdates.lastSceneConstruction = 0;
  m_objectUpdates.lastSceneCommit = 0;

  if (vscene)
    vscene->release();
  vscene = nullptr;
}

// box3 getEmbreeSceneBounds(RTCScene scene)
// {
//   RTCBounds eb;
//   rtcGetSceneBounds(scene, &eb);
//   return box3({eb.lower_x, eb.lower_y, eb.lower_z},
//       {eb.upper_x, eb.upper_y, eb.upper_z});
// }

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Group *);
