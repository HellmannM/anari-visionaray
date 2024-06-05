
#pragma once

// std
#include <memory>
// visionaray
#include "visionaray/bvh.h"
// ours
#include "surface/geometry/Geometry.h"
#include "surface/material/Material.h"
#include "light/Light.h"
#include "DeviceCopyableObjects.h"
#if defined(WITH_CUDA) || defined(WITH_HIP)
#include "VisionaraySceneGPU.h"
#endif

namespace visionaray {

struct VisionaraySceneImpl
{
#if defined(WITH_CUDA) || defined(WITH_HIP)
  friend struct VisionaraySceneGPU;
#endif

  typedef index_bvh<basic_triangle<3,float>> TriangleBVH;
  typedef index_bvh<basic_triangle<3,float>> QuadBVH;
  typedef index_bvh<basic_sphere<float>>     SphereBVH;
  typedef index_bvh<dco::Cone>               ConeBVH;
  typedef index_bvh<basic_cylinder<float>>   CylinderBVH;
  typedef index_bvh<dco::BezierCurve>        BezierCurveBVH;
  typedef index_bvh<dco::ISOSurface>         ISOSurfaceBVH;
  typedef index_bvh<dco::Volume>             VolumeBVH;

  typedef index_bvh<dco::BLS> TLS;
  typedef index_bvh<dco::WorldBLS> WorldTLS;

  enum Type { World, Group, };
  Type type;

  // Surface data //
  DeviceHandleArray m_geometries;
  DeviceHandleArray m_materials;
  DeviceHandleArray m_lights;
  HostDeviceArray<uint32_t> m_objIds;
  // flat list of lights (only used if type is World!)
  DeviceHandleArray m_allLights;

  // Accels //
  TLS m_TLS;
  WorldTLS m_worldTLS;
  DeviceObjectArray<dco::BLS> m_BLSs;
  DeviceObjectArray<dco::WorldBLS> m_worldBLSs;

  // Accel storage //
  struct {
    aligned_vector<TriangleBVH>    triangleBLSs;
    aligned_vector<QuadBVH>        quadBLSs;
    aligned_vector<SphereBVH>      sphereBLSs;
    aligned_vector<ConeBVH>        coneBLSs;
    aligned_vector<CylinderBVH>    cylinderBLSs;
    aligned_vector<BezierCurveBVH> bezierCurveBLSs;
    aligned_vector<ISOSurfaceBVH>  isoSurfaceBLSs;
    aligned_vector<VolumeBVH>      volumeBLSs;
  } m_accelStorage;

  // Internal state //
  unsigned m_worldID{UINT_MAX};
  unsigned m_groupID{UINT_MAX};
  VisionarayGlobalState *m_state{nullptr};

  // Interface //
  VisionaraySceneImpl(Type type, VisionarayGlobalState *state);
  ~VisionaraySceneImpl();
  void commit();
  void release();
  bool isValid() const;
  void attachGeometry(dco::Geometry geom, unsigned geomID, unsigned userID=~0u);
  void attachGeometry(
      dco::Geometry geom, dco::Material mat, unsigned geomID, unsigned userID=~0u);
  void updateGeometry(dco::Geometry geom);
  void attachLight(dco::Light light, unsigned id);
  aabb getBounds() const;
#ifdef WITH_CUDA
  cuda_index_bvh<dco::BLS>::bvh_inst instBVH(mat4x3 xfm);
#elif defined(WITH_HIP)
  hip_index_bvh<dco::BLS>::bvh_inst instBVH(mat4x3 xfm);
#else
  index_bvh<dco::BLS>::bvh_inst instBVH(mat4x3 xfm);
#endif

 private:
  void dispatch();

  VisionarayGlobalState *deviceState();
#if defined(WITH_CUDA) || defined(WITH_HIP)
  std::unique_ptr<VisionaraySceneGPU> m_gpuScene{nullptr};
#endif
};

typedef std::shared_ptr<VisionaraySceneImpl> VisionarayScene;
VisionarayScene newVisionarayScene(
    VisionaraySceneImpl::Type type, VisionarayGlobalState *state);

} // namespace visionaray
