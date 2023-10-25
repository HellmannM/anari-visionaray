// std
#include <algorithm>
#include <chrono>
#include <random>
#include <thread>
// visionaray
#include "visionaray/detail/parallel_for.h"
// ours
#include "frame/common.h"
#include "scene/light/HDRI.h"
#include "Frame.h"

namespace visionaray {

template <typename R, typename TASK_T>
static std::future<R> async(TASK_T &&fcn)
{
  auto task = std::packaged_task<R()>(std::forward<TASK_T>(fcn));
  auto future = task.get_future();

  std::thread([task = std::move(task)]() mutable { task(); }).detach();

  return future;
}

template <typename R>
static bool is_ready(const std::future<R> &f)
{
  return !f.valid()
      || f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

// Frame definitions //////////////////////////////////////////////////////////

Frame::Frame(VisionarayGlobalState *s) : helium::BaseFrame(s)
{
  vframe.frameID = s->objectCounts.frames++;
}

Frame::~Frame()
{
  wait();
  detach();
  deviceState()->objectCounts.frames--;
}

bool Frame::isValid() const
{
  return m_valid;
}

VisionarayGlobalState *Frame::deviceState() const
{
  return (VisionarayGlobalState *)helium::BaseObject::m_state;
}

void Frame::commit()
{
  m_renderer = getParamObject<Renderer>("renderer");
  if (!m_renderer) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'renderer' on frame");
  }

  m_camera = getParamObject<Camera>("camera");
  if (!m_camera) {
    reportMessage(
        ANARI_SEVERITY_WARNING, "missing required parameter 'camera' on frame");
  }

  m_world = getParamObject<World>("world");
  if (!m_world) {
    reportMessage(
        ANARI_SEVERITY_WARNING, "missing required parameter 'world' on frame");
  }

  m_valid = m_renderer && m_renderer->isValid() && m_camera
      && m_camera->isValid() && m_world && m_world->isValid();

  vframe.colorType = getParam<anari::DataType>("channel.color", ANARI_UNKNOWN);
  vframe.depthType = getParam<anari::DataType>("channel.depth", ANARI_UNKNOWN);
  vframe.normalType = getParam<anari::DataType>("channel.normal", ANARI_UNKNOWN);
  vframe.albedoType = getParam<anari::DataType>("channel.albedo", ANARI_UNKNOWN);
  vframe.primIdType =
      getParam<anari::DataType>("channel.primitiveId", ANARI_UNKNOWN);
  vframe.objIdType = getParam<anari::DataType>("channel.objectId", ANARI_UNKNOWN);
  vframe.instIdType = getParam<anari::DataType>("channel.instanceId", ANARI_UNKNOWN);

  vframe.size = getParam<uint2>("size", uint2(10));
  vframe.invSize = 1.f / float2(vframe.size);

  const auto numPixels = vframe.size.x * vframe.size.y;

  vframe.stochasticRendering = m_renderer->stochasticRendering();

  vframe.perPixelBytes = 4 * (vframe.colorType == ANARI_FLOAT32_VEC4 ? 4 : 1);
  m_pixelBuffer.resize(numPixels * vframe.perPixelBytes);

  m_depthBuffer.resize(vframe.depthType == ANARI_FLOAT32 ? numPixels : 0);
  m_accumBuffer.resize(numPixels, vec4{0.f});
  m_frameChanged = true;

  m_normalBuffer.clear();
  m_albedoBuffer.clear();
  m_primIdBuffer.clear();
  m_objIdBuffer.clear();
  m_instIdBuffer.clear();

  if (vframe.normalType == ANARI_FLOAT32_VEC3)
    m_normalBuffer.resize(numPixels);
  if (vframe.albedoType == ANARI_FLOAT32_VEC3)
    m_albedoBuffer.resize(numPixels);
  if (vframe.primIdType == ANARI_UINT32)
    m_primIdBuffer.resize(numPixels);
  if (vframe.objIdType == ANARI_UINT32)
    m_objIdBuffer.resize(numPixels);
  if (vframe.instIdType == ANARI_UINT32)
    m_instIdBuffer.resize(numPixels);

  vframe.pixelBuffer = m_pixelBuffer.data();
  vframe.depthBuffer = m_depthBuffer.data();
  vframe.normalBuffer = m_normalBuffer.data();
  vframe.albedoBuffer = m_albedoBuffer.data();
  vframe.primIdBuffer = m_primIdBuffer.data();
  vframe.objIdBuffer = m_objIdBuffer.data();
  vframe.instIdBuffer = m_instIdBuffer.data();
  vframe.accumBuffer = m_accumBuffer.data();

  dispatch();
}

bool Frame::getProperty(
    const std::string_view &name, ANARIDataType type, void *ptr, uint32_t flags)
{
  if (type == ANARI_FLOAT32 && name == "duration") {
    if (flags & ANARI_WAIT)
      wait();
    helium::writeToVoidP(ptr, m_duration);
    return true;
  }

  return 0;
}

void Frame::renderFrame()
{
  this->refInc(helium::RefType::INTERNAL);

  auto *state = deviceState();
  state->waitOnCurrentFrame();

  auto start = std::chrono::steady_clock::now();

  state->commitBuffer.flush();

  if (!isValid()) {
    reportMessage(
        ANARI_SEVERITY_ERROR, "skipping render of incomplete frame object");
    std::fill(m_pixelBuffer.begin(), m_pixelBuffer.end(), 0);
    this->refDec(helium::RefType::INTERNAL);
    return;
  }

  if (state->commitBuffer.lastFlush() <= m_frameLastRendered) {
    if (!m_renderer->stochasticRendering()) {
      this->refDec(helium::RefType::INTERNAL);
      return;
    }
  }

  checkAccumulationReset();

  m_frameLastRendered = helium::newTimeStamp();
  state->currentFrame = this;

  m_future = async<void>([&, state, start]() {
    m_world->visionaraySceneUpdate();

    const auto &size = vframe.size;
    dco::Camera cam = m_camera->visionarayCamera();
    VisionarayRenderer &rend = m_renderer->visionarayRenderer();
    VisionarayScene scene = m_world->visionarayScene();

    if (cam.type == dco::Camera::Pinhole)
      cam.asPinholeCam.begin_frame();
    else if (cam.type == dco::Camera::Matrix)
      cam.asMatrixCam.begin_frame();

    if (m_nextFrameReset) {
      std::fill(m_accumBuffer.begin(), m_accumBuffer.end(), vec4{0.f});
      rend.rendererState().accumID = 0;
      m_nextFrameReset = false;
    }

    rend.rendererState().envID = HDRI::backgroundID;

    parallel_for(state->threadPool,
        tiled_range2d<int>(0, size.x, 64, 0, size.y, 64),
        [&](range2d<int> r) {
          for (int y = r.cols().begin(); y != r.cols().end(); ++y) {
            for (int x = r.rows().begin(); x != r.rows().end(); ++x) {

              ScreenSample ss{x, y, size, {/*RNG*/}};
              Ray ray;

              uint64_t clock_begin = clock64();

              if (rend.stochasticRendering()) {
                // Need an RNG
                int pixelID = ss.x + ss.frameSize.x * ss.y;
                ss.random = Random(pixelID, rend.rendererState().accumID);
              }

              float4 accumColor{0.f};
              PixelSample firstSample;
              for (int sampleID=0; sampleID<rend.spp(); ++sampleID) {

                float xf(x), yf(y);
                if (rend.stochasticRendering()) {
                  // jitter pixel sample
                  vec2f jitter(ss.random() - .5f, ss.random() - .5f);
                  xf += jitter.x;
                  yf += jitter.y;
                }

                if (cam.type == dco::Camera::Pinhole)
                  ray = cam.asPinholeCam.primary_ray(
                      Ray{}, ss.random, xf, yf, float(size.x), float(size.y));
                else if (cam.type == dco::Camera::Matrix)
                  ray = cam.asMatrixCam.primary_ray(
                      Ray{}, xf, yf, float(size.x), float(size.y));

                PixelSample ps = rend.renderSample(ss,
                        ray,
                        scene->m_worldID,
                        deviceState()->onDevice,
                        deviceState()->objectCounts);
                accumColor += ps.color;
                if (sampleID == 0) {
                  firstSample = ps;
                }
              }

              uint64_t clock_end = clock64();
              if (rend.rendererState().heatMapEnabled > 0.f) {
                  float t = (clock_end - clock_begin)
                      * (rend.rendererState().heatMapScale / rend.spp());
                  accumColor = over(vec4f(heatMap(t), .5f), accumColor);
              }

              // Color gets accumulated, depth, IDs, etc. are
              // taken from first sample
              PixelSample finalSample = firstSample;
              finalSample.color = accumColor*(1.f/rend.spp());
              vframe.writeSample(x, y, rend.rendererState().accumID, finalSample);
            }
          }
        });

    if (cam.type == dco::Camera::Pinhole)
      cam.asPinholeCam.end_frame();
    else if (cam.type == dco::Camera::Matrix)
      cam.asMatrixCam.end_frame();

    rend.rendererState().accumID++;

    auto end = std::chrono::steady_clock::now();
    m_duration = std::chrono::duration<float>(end - start).count();
  });
}

void *Frame::map(std::string_view channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType)
{
  wait();

  *width = vframe.size.x;
  *height = vframe.size.y;

  if (channel == "color" || channel == "channel.color") {
    *pixelType = vframe.colorType;
    return mapColorBuffer();
  } else if (channel == "depth" || channel == "channel.depth") {
    *pixelType = ANARI_FLOAT32;
    return mapDepthBuffer();
  } else if (channel == "channel.normal" && !m_normalBuffer.empty()) {
    *pixelType = ANARI_FLOAT32_VEC3;
    return m_normalBuffer.data();
  } else if (channel == "channel.albedo" && !m_albedoBuffer.empty()) {
    *pixelType = ANARI_FLOAT32_VEC3;
    return m_albedoBuffer.data();
  } else if (channel == "channel.primitiveId" && !m_primIdBuffer.empty()) {
    *pixelType = ANARI_UINT32;
    return m_primIdBuffer.data();
  } else if (channel == "channel.objectId" && !m_objIdBuffer.empty()) {
    *pixelType = ANARI_UINT32;
    return m_objIdBuffer.data();
  } else if (channel == "channel.instanceId" && !m_instIdBuffer.empty()) {
    *pixelType = ANARI_UINT32;
    return m_instIdBuffer.data();
  }else {
    *width = 0;
    *height = 0;
    *pixelType = ANARI_UNKNOWN;
    return nullptr;
  }
}

void Frame::unmap(std::string_view channel)
{
  // no-op
}

int Frame::frameReady(ANARIWaitMask m)
{
  if (m == ANARI_NO_WAIT)
    return ready();
  else {
    wait();
    return 1;
  }
}

void Frame::discard()
{
  // no-op
}

void *Frame::mapColorBuffer()
{
  return m_pixelBuffer.data();
}

void *Frame::mapDepthBuffer()
{
  return m_depthBuffer.data();
}

bool Frame::ready() const
{
  return is_ready(m_future);
}

void Frame::wait() const
{
  if (m_future.valid()) {
    m_future.get();
    this->refDec(helium::RefType::INTERNAL);
    if (deviceState()->currentFrame == this)
      deviceState()->currentFrame = nullptr;
  }
}

void Frame::checkAccumulationReset()
{
  if (m_nextFrameReset)
    return;

  auto &state = *deviceState();
  if (m_lastCommitOccured < state.commitBuffer.lastFlush()) {
    m_lastCommitOccured = state.commitBuffer.lastFlush();
    m_nextFrameReset = true;
  }
  // if (m_lastUploadOccured < state.uploadBuffer.lastFlush()) {
  //   m_lastUploadOccured = state.uploadBuffer.lastFlush();
  //   m_nextFrameReset = true;
  // }
}

void Frame::dispatch()
{
  if (deviceState()->dcos.frames.size() <= vframe.frameID) {
    deviceState()->dcos.frames.resize(vframe.frameID+1);
  }
  deviceState()->dcos.frames[vframe.frameID] = vframe;

  // Upload/set accessible pointers
  deviceState()->onDevice.frames = deviceState()->dcos.frames.data();
}

void Frame::detach()
{
  if (deviceState()->dcos.frames.size() > vframe.frameID) {
    if (deviceState()->dcos.frames[vframe.frameID].frameID == vframe.frameID) {
      deviceState()->dcos.frames.erase(
          deviceState()->dcos.frames.begin() + vframe.frameID);
    }
  }

  // Upload/set accessible pointers
  deviceState()->onDevice.frames = deviceState()->dcos.frames.data();
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Frame *);
