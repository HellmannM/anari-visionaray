
#include "for_each.h"
#include "DRR_impl.h"

namespace visionaray {

VSNRAY_FUNC
inline PixelSample renderSample(ScreenSample &ss, Ray ray, unsigned worldID,
    const VisionarayGlobalState::DeviceObjectRegistry &onDevice,
    const RendererState &rendererState)
{
  PixelSample result;
  result.color = rendererState.bgColor;
  result.depth = 1e31f;

  auto hrv = intersectVolumeBounds(ray, onDevice.TLSs[worldID]);

  if (hrv.hit) {
    const auto &inst = onDevice.instances[hrv.instID];
    const auto &group = onDevice.groups[inst.groupID];
    const dco::Volume &vol = onDevice.volumes[group.volumes[hrv.volID]];

    float3 color(0.f);
    float alpha = 0.f;

    result.depth = rayMarchVolumeDRR(ss, ray, vol, color, alpha);
    result.color = over(float4(color,alpha), result.color);
    result.Ng = float3{}; // TODO: gradient
    result.Ns = float3{}; // TODO..
    result.albedo = float3{}; // TODO..
    result.objId = group.objIds[hrv.volID];
    result.instId = inst.userID;
    auto point = ray.ori + result.depth * ray.dir; //TODO store this point instead of depth
  }

  return result;
}

void VisionarayRendererDRR::renderFrame(const dco::Frame &frame,
                                            const dco::Camera &cam,
                                            uint2 size,
                                            VisionarayGlobalState *state,
                                            const VisionarayGlobalState::DeviceObjectRegistry &DD,
                                            const RendererState &rendererState,
                                            unsigned worldID, int frameID)
{
#ifdef WITH_CUDA
  VisionarayGlobalState::DeviceObjectRegistry *onDevicePtr;
  CUDA_SAFE_CALL(cudaMalloc(&onDevicePtr, sizeof(DD)));
  CUDA_SAFE_CALL(cudaMemcpy(onDevicePtr, &DD, sizeof(DD), cudaMemcpyHostToDevice));

  RendererState *rendererStatePtr;
  CUDA_SAFE_CALL(cudaMalloc(&rendererStatePtr, sizeof(rendererState)));
  CUDA_SAFE_CALL(cudaMemcpy(rendererStatePtr,
                            &rendererState,
                            sizeof(rendererState),
                            cudaMemcpyHostToDevice));

  dco::Frame *framePtr;
  CUDA_SAFE_CALL(cudaMalloc(&framePtr, sizeof(frame)));
  CUDA_SAFE_CALL(cudaMemcpy(framePtr, &frame, sizeof(frame), cudaMemcpyHostToDevice));

  cuda::for_each(0, size.x, 0, size.y,
#elif defined(WITH_HIP)
  VisionarayGlobalState::DeviceObjectRegistry *onDevicePtr;
  HIP_SAFE_CALL(hipMalloc(&onDevicePtr, sizeof(DD)));
  HIP_SAFE_CALL(hipMemcpy(onDevicePtr, &DD, sizeof(DD), hipMemcpyHostToDevice));

  RendererState *rendererStatePtr;
  HIP_SAFE_CALL(hipMalloc(&rendererStatePtr, sizeof(rendererState)));
  HIP_SAFE_CALL(hipMemcpy(rendererStatePtr,
                          &rendererState,
                          sizeof(rendererState),
                          hipMemcpyHostToDevice));

  dco::Frame *framePtr;
  HIP_SAFE_CALL(hipMalloc(&framePtr, sizeof(frame)));
  HIP_SAFE_CALL(hipMemcpy(framePtr, &frame, sizeof(frame), hipMemcpyHostToDevice));

  hip::for_each(0, size.x, 0, size.y,
#else
  auto *onDevicePtr = &DD;
  auto *rendererStatePtr = &rendererState;
  auto *framePtr = &frame;
  parallel::for_each(state->threadPool, 0, size.x, 0, size.y,
#endif
      [=] VSNRAY_GPU_FUNC (int x, int y) {

        const VisionarayGlobalState::DeviceObjectRegistry &onDevice = *onDevicePtr;
        const auto &rendererState = *rendererStatePtr;
        const auto &frame = *framePtr;

        ScreenSample ss{x, y, frameID, size, {/*no RNG*/}};
        Ray ray;

        uint64_t clock_begin = clock64();

        float4 accumColor{0.f};
        PixelSample firstSample;
        int spp = rendererState.pixelSamples;

        for (int sampleID=0; sampleID<spp; ++sampleID) {

          ray = cam.primary_ray(
              ss.random, float(x), float(y), float(size.x), float(size.y));
#if 1
          ray.dbg = ss.debug();
#endif

          ray = clipRay(ray, rendererState.clipPlanes, rendererState.numClipPlanes);

          PixelSample ps = renderSample(ss,
                  ray,
                  worldID,
                  onDevice,
                  rendererState);
          accumColor += ps.color;
          if (sampleID == 0) {
            firstSample = ps;
          }
        }

        uint64_t clock_end = clock64();
        if (rendererState.heatMapEnabled > 0.f) {
            float t = (clock_end - clock_begin)
                * (rendererState.heatMapScale / spp);
            accumColor = over(vec4f(heatMap(t), .5f), accumColor);
        }

        // Color gets accumulated, depth, IDs, etc. are
        // taken from first sample
        PixelSample finalSample = firstSample;
        finalSample.color = accumColor*(1.f/spp);
        frame.writeSample(x, y, rendererState.accumID, finalSample);
      });
#ifdef WITH_CUDA
  CUDA_SAFE_CALL(cudaFree(onDevicePtr));
  CUDA_SAFE_CALL(cudaFree(rendererStatePtr));
  CUDA_SAFE_CALL(cudaFree(framePtr));
#elif defined(WITH_HIP)
  HIP_SAFE_CALL(hipFree(onDevicePtr));
  HIP_SAFE_CALL(hipFree(rendererStatePtr));
  HIP_SAFE_CALL(hipFree(framePtr));
#endif
}

} // namespace visionaray
