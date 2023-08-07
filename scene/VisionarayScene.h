
#pragma once

// std
#include <memory>
// visionaray
#include "visionaray/aligned_vector.h"
#include "visionaray/bvh.h"
// ours
#include "surface/geometry/Geometry.h"
#include "surface/material/Material.h"
#include "DeviceCopyableObjects.h"

namespace visionaray {

typedef index_bvh<basic_triangle<3,float>> TriangleBVH;
typedef index_bvh<basic_sphere<float>>     SphereBVH;
typedef index_bvh<basic_cylinder<float>>   CylinderBVH;

typedef index_bvh<dco::BLS> TLS;

struct VisionaraySceneImpl
{
  enum Type { World, Group, };
  Type type;

  struct {
    dco::TLS theTLS;
    dco::Geometry *geoms{nullptr};
  } onDevice;

  // Geometries //
  aligned_vector<dco::Geometry> m_geometries;

  // Accels //
  TLS m_TLS;
  aligned_vector<dco::BLS> m_BLSs;

  // Accel storage //
  struct {
    aligned_vector<TriangleBVH> triangleBLSs;
    aligned_vector<SphereBVH>   sphereBLSs;
    aligned_vector<CylinderBVH> cylinderBLSs;
  } m_accelStorage;

  // Surface properties //
  aligned_vector<VisionarayMaterial> m_materials;

  // Internal state //
  unsigned m_worldID{UINT_MAX};
  unsigned m_groupID{UINT_MAX};
  VisionarayGlobalState *m_state{nullptr};

  // Interface //
  VisionaraySceneImpl(Type type, VisionarayGlobalState *state);
  void commit();
  void release();
  void attachGeometry(dco::Geometry geom, unsigned geomID);

private:
  void dispatch();
};

typedef std::shared_ptr<VisionaraySceneImpl> VisionarayScene;
VisionarayScene newVisionarayScene(
    VisionaraySceneImpl::Type type, VisionarayGlobalState *state);

} // namespace visionaray
