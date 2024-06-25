#pragma once

// visionaray
#include "visionaray/material.h"
// ours
#include "common.h"
#include "DeviceCopyableObjects.h"
#include "VisionarayGlobalState.h"

namespace visionaray {

template<unsigned int N=4>
struct LCG
{
  inline VSNRAY_FUNC LCG()
  { /* intentionally empty so we can use it in device vars that
       don't allow dynamic initialization (ie, PRD) */
  }
  inline VSNRAY_FUNC LCG(unsigned int val0, unsigned int val1)
  { init(val0,val1); }

  inline VSNRAY_FUNC LCG(const vec2i &seed)
  { init((unsigned)seed.x,(unsigned)seed.y); }
  inline VSNRAY_FUNC LCG(const vec2ui &seed)
  { init(seed.x,seed.y); }
  
  inline VSNRAY_FUNC void init(unsigned int val0, unsigned int val1)
  {
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;
  
    for (unsigned int n = 0; n < N; n++) {
      s0 += 0x9e3779b9;
      v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
      v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
    }
    state = v0;
  }

  // Generate random unsigned int in [0, 2^24)
  inline VSNRAY_FUNC float operator() ()
  {
    const uint32_t LCG_A = 1664525u;
    const uint32_t LCG_C = 1013904223u;
    state = (LCG_A * state + LCG_C);
    return (state & 0x00FFFFFF) / (float) 0x01000000;
  }

  // For compat. with visionaray
  inline VSNRAY_FUNC float next()
  {
    return operator()();
  }

  uint32_t state;
};

typedef LCG<4> Random;

VSNRAY_FUNC
inline float epsilonFrom(const vec3 &P, const vec3 &dir, float t)
{
  constexpr float ulpEpsilon = 0x1.fp-18;
  return max_element(vec4(abs(P), max_element(abs(dir)) * t)) * ulpEpsilon;
}

struct ScreenSample
{
  int x, y;
  int frameID;
  uint2 frameSize;
  Random random;

  inline VSNRAY_FUNC bool debug() {
#if 1
    return x == frameSize.x/2 && y == frameSize.y/2;
#else
    return false;
#endif
  }
};

enum class RenderMode
{
  Default,
  Ng,
  Ns,
  Tangent,
  Bitangent,
  Albedo,
  MotionVec,
  GeometryAttribute0,
  GeometryAttribute1,
  GeometryAttribute2,
  GeometryAttribute3,
  GeometryColor,
};

struct RendererState
{
  float4 bgColor{float3(0.f), 1.f};
  RenderMode renderMode{RenderMode::Default};
  float4 *clipPlanes{nullptr};
  unsigned numClipPlanes{0};
  int pixelSamples{1};
  int accumID{0};
  int envID{-1};
  // TAA
  bool taaEnabled{false};
  float taaAlpha{0.3f};
  mat4 prevMV{mat4::identity()};
  mat4 prevPR{mat4::identity()};
  mat4 currMV{mat4::identity()};
  mat4 currPR{mat4::identity()};
  // Volume
  bool gradientShading{false};
  // AO
  float3 ambientColor{1.f, 1.f, 1.f};
  float ambientRadiance{0.2f};
  float occlusionDistance{1e20f};
  int ambientSamples{1};
  // Heat map
  bool heatMapEnabled{false};
  float heatMapScale{.1f};

};

inline VSNRAY_FUNC
vec3 hsv2rgb(vec3 in)
{
    float      hh, p, q, t, ff;
    long        i;
    vec3         out;

    if(in.y <= 0.0) {       // < is bogus, just shuts up warnings
        out.x = in.z;
        out.y = in.z;
        out.z = in.z;
        return out;
    }
    hh = in.x;
    if(hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = in.z * (1.0 - in.y);
    q = in.z * (1.0 - (in.y * ff));
    t = in.z * (1.0 - (in.y * (1.0 - ff)));

    switch(i) {
        case 0:
            out.x = in.z;
            out.y = t;
            out.z = p;
            break;
        case 1:
            out.x = q;
            out.y = in.z;
            out.z = p;
            break;
        case 2:
            out.x = p;
            out.y = in.z;
            out.z = t;
            break;

        case 3:
            out.x = p;
            out.y = q;
            out.z = in.z;
            break;
        case 4:
            out.x = t;
            out.y = p;
            out.z = in.z;
            break;
        case 5:
        default:
            out.x = in.z;
            out.y = p;
            out.z = q;
            break;
    }
    return out;
}

inline VSNRAY_FUNC int uniformSampleOneLight(Random &rnd, int numLights)
{
  int which = int(rnd() * numLights); if (which == numLights) which = 0;
  return which;
}

VSNRAY_FUNC
inline uint32_t getSphereIndex(const dco::Array &indexArray, unsigned primID)
{
  uint32_t index;
  if (indexArray.len > 0) {
    index = ((uint32_t *)indexArray.data)[primID];
  } else {
    index = primID;
  }
  return index;
}

VSNRAY_FUNC
inline uint2 getConeIndex(const dco::Array &indexArray, unsigned primID)
{
  uint2 index;
  if (indexArray.len > 0) {
    index = ((uint2 *)indexArray.data)[primID];
  } else {
    index = uint2(primID * 2, primID * 2 + 1);
  }
  return index;
}

VSNRAY_FUNC
inline uint2 getCylinderIndex(const dco::Array &indexArray, unsigned primID)
{
  uint2 index;
  if (indexArray.len > 0) {
    index = ((uint2 *)indexArray.data)[primID];
  } else {
    index = uint2(primID * 2, primID * 2 + 1);
  }
  return index;
}

VSNRAY_FUNC
inline uint3 getTriangleIndex(const dco::Array &indexArray, unsigned primID)
{
  uint3 index;
  if (indexArray.len > 0) {
    index = ((uint3 *)indexArray.data)[primID];
  } else {
    index = uint3(primID * 3, primID * 3 + 1, primID * 3 + 2);
  }
  return index;
}

VSNRAY_FUNC
inline uint4 getQuadIndex(const dco::Array &indexArray, unsigned primID)
{
  uint4 index;
  if (indexArray.len > 0) {
    index = ((uint4 *)indexArray.data)[primID/2]; // primID refers to triangles!
  } else {
    primID /= 2; // tri to quad
    index = uint4(primID * 4, primID * 4 + 1, primID * 4 + 2, primID * 4 + 3);
  }
  return index;
}

VSNRAY_FUNC
inline void getNormals(const dco::Geometry &geom,
                       unsigned primID,
                       const vec3 hitPos,
                       const vec2 uv,
                       vec3 &Ng,
                       vec3 &Ns)
{
  // TODO: doesn't work for instances yet
  if (geom.type == dco::Geometry::Triangle) {
    auto tri = geom.as<dco::Triangle>(primID);
    Ng = normalize(cross(tri.e1,tri.e2));
    if (geom.normal.len
        && geom.normal.typeInfo.dataType == ANARI_FLOAT32_VEC3) {
      uint3 index = getTriangleIndex(geom.index, primID);
      auto *normals = (const vec3 *)geom.normal.data;
      vec3 n1 = normals[index.x];
      vec3 n2 = normals[index.y];
      vec3 n3 = normals[index.z];
      Ns = lerp(n1, n2, n3, uv.x, uv.y);
      Ns = normalize(Ns);
    } else {
      Ns = Ng;
    }
  } else if (geom.type == dco::Geometry::Quad) {
    auto qtri = geom.as<dco::Triangle>(primID);
    Ng = normalize(cross(qtri.e1,qtri.e2));
    Ns = Ng;
  } else if (geom.type == dco::Geometry::Sphere) {
    auto sph = geom.as<dco::Sphere>(primID);
    Ng = normalize((hitPos-sph.center) / sph.radius);
    Ns = Ng;
  } else if (geom.type == dco::Geometry::Cone) {
    // reconstruct normal (see https://iquilezles.org/articles/intersectors/)
    auto cone = geom.as<dco::Cone>(primID);
    const vec3f ba = cone.v2 - cone.v1;
    const float m0 = dot(ba,ba);
    if (uv.x <= 0.f) {
      Ng = -ba*rsqrt(m0);
    } else if (uv.x >= 1) {
      Ng = ba*rsqrt(m0);
    } else {
      const float ra = cone.r1;
      const float rr = cone.r1 - cone.r2;
      const float hy = m0 + rr*rr;
      const float y = uv.y; // uv.y stores the unnormalized cone parameter t!
      const vec3f localPos = hitPos-cone.v1;
      Ng = normalize(m0*(m0*localPos+rr*ba*ra)-ba*hy*y);
    }
    Ns = Ng;
  } else if (geom.type == dco::Geometry::Cylinder) {
    auto cyl = geom.as<dco::Cylinder>(primID);
    vec3f axis = normalize(cyl.v2-cyl.v1);
    if (length(hitPos-cyl.v1) < cyl.radius)
      Ng = -axis;
    else if (length(hitPos-cyl.v2) < cyl.radius)
      Ng = axis;
    else {
      float t = dot(hitPos-cyl.v1, axis);
      vec3f pt = cyl.v1 + t * axis;
      Ng = normalize(hitPos-pt);
    }
    Ns = Ng;
  } else if (geom.type == dco::Geometry::BezierCurve) {
    float t = uv.x;
    vec3f curvePos = geom.as<dco::BezierCurve>(primID).f(t);
    Ng = normalize(hitPos-curvePos);
    Ns = Ng;
  } else if (geom.type == dco::Geometry::ISOSurface) {
    if (!sampleGradient(geom.as<dco::ISOSurface>(0).field,hitPos,Ng)) {
      Ng = vec3f(0.f);
    } else {
      Ng = normalize(Ng);
    }
    Ns = Ng;
  }
}

VSNRAY_FUNC
inline vec4 getTangent(
    const dco::Geometry &geom, unsigned primID, const vec3 hitPos, const vec2 uv)
{
  vec4f tng(0.f);

  if (geom.type == dco::Geometry::Triangle) {
    if (geom.tangent.len) {
      uint3 index = getTriangleIndex(geom.index, primID);
      if (geom.tangent.typeInfo.dataType == ANARI_FLOAT32_VEC3) {
        auto *tangents = (const vec3 *)geom.tangent.data;
        vec3 tng1 = tangents[index.x];
        vec3 tng2 = tangents[index.y];
        vec3 tng3 = tangents[index.z];
        tng = vec4(lerp(tng1, tng2, tng3, uv.x, uv.y), 1.f);
      } else if (geom.tangent.typeInfo.dataType == ANARI_FLOAT32_VEC4) {
        auto *tangents = (const vec4 *)geom.tangent.data;
        vec4 tng1 = tangents[index.x];
        vec4 tng2 = tangents[index.y];
        vec4 tng3 = tangents[index.z];
        tng = lerp(tng1, tng2, tng3, uv.x, uv.y);
      }
    }
  }

  return tng;
}

VSNRAY_FUNC
inline vec4 getAttribute(
    const dco::Geometry &geom, dco::Attribute attrib, unsigned primID, const vec2 uv)
{
  vec4f color{0.f, 0.f, 0.f, 1.f};

  if (attrib == dco::Attribute::None)
    return color;

  dco::Array vertexColors = geom.vertexAttributes[(int)attrib];
  dco::Array primitiveColors = geom.primitiveAttributes[(int)attrib];

  const TypeInfo &vertexColorInfo = vertexColors.typeInfo;
  const TypeInfo &primitiveColorInfo = primitiveColors.typeInfo;

  // vertex colors take precedence over primitive colors
  if (vertexColors.len > 0) {
    if (geom.type == dco::Geometry::Triangle) {
      uint3 index = getTriangleIndex(geom.index, primID);
      const auto *source1
          = (const uint8_t *)vertexColors.data
              + index.x * vertexColorInfo.sizeInBytes;
      const auto *source2
          = (const uint8_t *)vertexColors.data
              + index.y * vertexColorInfo.sizeInBytes;
      const auto *source3
          = (const uint8_t *)vertexColors.data
              + index.z * vertexColorInfo.sizeInBytes;
      vec4f c1 = toRGBA(source1, vertexColorInfo);
      vec4f c2 = toRGBA(source2, vertexColorInfo);
      vec4f c3 = toRGBA(source3, vertexColorInfo);
      color = lerp(c1, c2, c3, uv.x, uv.y);
    }
    else if (geom.type == dco::Geometry::Quad) {
      uint4 index = getQuadIndex(geom.index, primID);
      const auto *source1
          = (const uint8_t *)vertexColors.data
              + index.x * vertexColorInfo.sizeInBytes;
      const auto *source2
          = (const uint8_t *)vertexColors.data
              + index.y * vertexColorInfo.sizeInBytes;
      const auto *source3
          = (const uint8_t *)vertexColors.data
              + index.z * vertexColorInfo.sizeInBytes;
      const auto *source4
          = (const uint8_t *)vertexColors.data
              + index.w * vertexColorInfo.sizeInBytes;
      vec4f c1 = toRGBA(source1, vertexColorInfo);
      vec4f c2 = toRGBA(source2, vertexColorInfo);
      vec4f c3 = toRGBA(source3, vertexColorInfo);
      vec4f c4 = toRGBA(source4, vertexColorInfo);
      if (primID%2==0)
        color = lerp(c1, c2, c4, uv.x, uv.y);
      else
        color = lerp(c3, c4, c2, 1.f-uv.x, 1.f-uv.y);
    }
    else if (geom.type == dco::Geometry::Sphere) {
      uint32_t index = getSphereIndex(geom.index, primID);
      const auto *source
          = (const uint8_t *)vertexColors.data
              + index * vertexColorInfo.sizeInBytes;
      color = toRGBA(source, vertexColorInfo);
    }
    else if (geom.type == dco::Geometry::Cone) {
      uint2 index = getConeIndex(geom.index, primID);
      const auto *source1
          = (const uint8_t *)vertexColors.data
              + index.x * vertexColorInfo.sizeInBytes;
      const auto *source2
          = (const uint8_t *)vertexColors.data
              + index.y * vertexColorInfo.sizeInBytes;
      vec4f c1 = toRGBA(source1, vertexColorInfo);
      vec4f c2 = toRGBA(source2, vertexColorInfo);
      color = lerp(c1, c2, uv.x);
    }
    else if (geom.type == dco::Geometry::Cylinder) {
      uint2 index = getCylinderIndex(geom.index, primID);
      const auto *source1
          = (const uint8_t *)vertexColors.data
              + index.x * vertexColorInfo.sizeInBytes;
      const auto *source2
          = (const uint8_t *)vertexColors.data
              + index.y * vertexColorInfo.sizeInBytes;
      vec4f c1 = toRGBA(source1, vertexColorInfo);
      vec4f c2 = toRGBA(source2, vertexColorInfo);
      color = lerp(c1, c2, uv.x);
    }
  } else if (primitiveColors.len > 0) {
    const auto *source
        = (const uint8_t *)primitiveColors.data
            + primID * primitiveColorInfo.sizeInBytes;
    color = toRGBA(source, primitiveColorInfo);
  }

  return color;
}

VSNRAY_FUNC
inline vec4 getSample(
    const dco::Sampler &samp, const float4 *attribs, unsigned primID)
{
  if (samp.type == dco::Sampler::Primitive) {
    const TypeInfo &info = samp.asPrimitive.typeInfo;
    const auto *source = samp.asPrimitive.data
        + (primID * info.sizeInBytes) + (samp.asPrimitive.offset * info.sizeInBytes);
    return toRGBA(source, info);
  } else if (samp.type == dco::Sampler::Transform) {
    vec4f inAttr = attribs[(int)samp.inAttribute];
    return samp.outTransform * inAttr + samp.outOffset;
  } else {
    vec4f inAttr = attribs[(int)samp.inAttribute];

    inAttr = samp.inTransform * inAttr + samp.inOffset;

    vec4f s{0.f, 0.f, 0.f, 1.f};

    if (samp.type == dco::Sampler::Image1D)
      s = tex1D(samp.asImage1D, inAttr.x);
    else if (samp.type == dco::Sampler::Image2D)
      s = tex2D(samp.asImage2D, inAttr.xy());
    else if (samp.type == dco::Sampler::Image3D)
      s = tex3D(samp.asImage3D, inAttr.xyz());

    return samp.outTransform * s + samp.outOffset;
  }
}

VSNRAY_FUNC
inline vec4 getRGBA(const dco::MaterialParamRGB &param,
                    const dco::Sampler *samplers,
                    const float4 *attribs,
                    unsigned primID)
{
  if (param.samplerID < UINT_MAX)
    return getSample(samplers[param.samplerID], attribs, primID);
  else if (param.attribute != dco::Attribute::None)
    return attribs[(int)param.attribute];
  else
    return vec4f(param.rgb, 1.f);
}

VSNRAY_FUNC
inline float getF(const dco::MaterialParamF &param,
                  const dco::Sampler *samplers,
                  const float4 *attribs,
                  unsigned primID)
{
  if (param.samplerID < UINT_MAX)
    return getSample(samplers[param.samplerID], attribs, primID).x;
  else if (param.attribute != dco::Attribute::None)
    return attribs[(int)param.attribute].x;
  else
    return param.f;
}

VSNRAY_FUNC
inline vec4 getColorMatte(const dco::Material &mat,
                          const dco::Sampler *samplers,
                          const float4 *attribs,
                          unsigned primID)
{
  return getRGBA(mat.asMatte.color, samplers, attribs, primID);
}

VSNRAY_FUNC
inline vec4 getColorPBM(const dco::Material &mat,
                        const dco::Sampler *samplers,
                        const float4 *attribs,
                        unsigned primID)
{
  const float metallic = getF(
      mat.asPhysicallyBased.metallic, samplers, attribs, primID);
  vec4f color = getRGBA(mat.asPhysicallyBased.baseColor, samplers, attribs, primID);
  return lerp(color, vec4f(0.f, 0.f, 0.f, color.w), metallic);
}

VSNRAY_FUNC
inline vec4 getColor(const dco::Material &mat,
                     const dco::Sampler *samplers,
                     const float4 *attribs,
                     unsigned primID)
{
  vec4f color{0.f, 0.f, 0.f, 1.f};
  if (mat.type == dco::Material::Matte)
    color = getColorMatte(mat, samplers, attribs, primID);
  else if (mat.type == dco::Material::PhysicallyBased) {
    color = getColorPBM(mat, samplers, attribs, primID);
  }
  return color;
}

VSNRAY_FUNC
inline float getOpacity(const dco::Material &mat,
                        const dco::Sampler *samplers,
                        const float4 *attribs,
                        unsigned primID)
{
  float opacity = 1.f;
  dco::AlphaMode mode{dco::AlphaMode::Opaque};
  float cutoff = 0.5f;

  if (mat.type == dco::Material::Matte) {
    vec4f color = getColorMatte(mat, samplers, attribs, primID);
    opacity = color.w * getF(mat.asMatte.opacity, samplers, attribs, primID);
    mode = mat.asMatte.alphaMode;
    cutoff = mat.asMatte.alphaCutoff;
  } else if (mat.type == dco::Material::PhysicallyBased) {
    vec4f color = getColorPBM(mat, samplers, attribs, primID);
    opacity = color.w * getF(mat.asPhysicallyBased.opacity, samplers, attribs, primID);
    mode = mat.asPhysicallyBased.alphaMode;
    cutoff = mat.asPhysicallyBased.alphaCutoff;
  }

  if (mode == dco::AlphaMode::Opaque)
    return 1.f;
  else if (mode == dco::AlphaMode::Blend)
    return opacity;
  else // mode==Mask
    return opacity >= cutoff ? 1.f : 0.f;
}

VSNRAY_FUNC
inline vec3 getPerturbedNormal(const dco::Material &mat,
                               const dco::Sampler *samplers,
                               const float4 *attribs,
                               unsigned primID,
                               const vec3 T, const vec3 B, const vec3 N)
{
  vec3f pn = N;

  mat3 TBN(T,B,N);
  if (mat.type == dco::Material::PhysicallyBased) {
    const auto &samp = samplers[mat.asPhysicallyBased.normal.samplerID];
    vec4 s = getSample(samp, attribs, primID);
    vec3 tbnN = s.xyz();
    if (length(tbnN) > 0.f) {
      vec3f objN = normalize(TBN * tbnN);
      //pn = lerp(N, objN, 0.5f); // encode in outTransform!
      pn = objN;
    }
  }

  return pn;
}

VSNRAY_FUNC
inline mat3 getNormalTransform(const dco::Instance &inst, const Ray &ray)
{
  if (inst.type == dco::Instance::Transform) {
    return inst.asTransform.normalXfm;
  } else if (inst.type == dco::Instance::MotionTransform) {

    float time01 = ray.time - inst.asMotionTransform.time.min
        / (inst.asMotionTransform.time.max - inst.asMotionTransform.time.min);

    unsigned ID1 = unsigned(float(inst.asMotionTransform.len-1) * time01);
    unsigned ID2 = min((unsigned)inst.asMotionTransform.len-1, ID1+1);

    float frac = time01 * (inst.asMotionTransform.len-1) - ID1;

    return lerp(inst.asMotionTransform.normalXfms[ID1],
                inst.asMotionTransform.normalXfms[ID2],
                frac);
  }

  return {};
}

VSNRAY_FUNC
inline float pow2(float f)
{
  return f*f;
}

VSNRAY_FUNC
inline float pow5(float f)
{
  return f*f*f*f*f;
}

// From: https://google.github.io/filament/Filament.html
VSNRAY_FUNC
inline vec3 F_Schlick(float u, vec3 f0)
{
  return f0 + (vec3f(1.f) - f0) * pow5(1.f - u);
}

VSNRAY_FUNC
inline float F_Schlick(float u, float f0)
{
  return f0 + (1.f - f0) * pow5(1.f - u);
}

VSNRAY_FUNC
inline float F_Schlick(float u, float f0, float f90)
{
  return f0 + (f90 - f0) * pow5(1.f - u);
}

VSNRAY_FUNC
inline float Fd_Lambert()
{
  return constants::inv_pi<float>();
}

VSNRAY_FUNC
inline float Fd_Burley(float NdotV, float NdotL, float LdotH, float roughness)
{
  float f90 = 0.5f + 2.f * roughness * LdotH * LdotH;
  float lightScatter = F_Schlick(NdotL, 1.f, f90);
  float viewScatter = F_Schlick(NdotV, 1.f, f90);
  return lightScatter * viewScatter * constants::inv_pi<float>();
}

VSNRAY_FUNC
inline float D_GGX(float NdotH, float roughness, float EPS)
{
  float alpha = roughness;
  return (alpha*alpha*heaviside(NdotH))
    / (constants::pi<float>()*pow2(NdotH*NdotH*(alpha*alpha-1.f)+1.f));
}

VSNRAY_FUNC
inline float V_Kelemen(float LdotH, const float EPS)
{
  return 0.25f / fmaxf(EPS, (LdotH * LdotH));
}

VSNRAY_FUNC
inline vec3 evalPhysicallyBasedMaterial(const dco::Material &mat,
                                        const dco::Sampler *samplers,
                                        const float4 *attribs,
                                        unsigned primID,
                                        const vec3 Ng, const vec3 Ns,
                                        const vec3 viewDir, const vec3 lightDir,
                                        const vec3 lightIntensity)
{
  const float metallic = getF(
      mat.asPhysicallyBased.metallic, samplers, attribs, primID);
  const float roughness = getF(
      mat.asPhysicallyBased.roughness, samplers, attribs, primID);
  const float clearcoat = getF(
      mat.asPhysicallyBased.clearcoat, samplers, attribs, primID);
  const float clearcoatRoughness = getF(
      mat.asPhysicallyBased.clearcoatRoughness, samplers, attribs, primID);
  const float ior = mat.asPhysicallyBased.ior;

  const float alpha = roughness * roughness;
  const float clearcoatAlpha = clearcoatRoughness * clearcoatRoughness;

  constexpr float EPS = 1e-14f;
  const vec3 H = normalize(lightDir+viewDir);
  const float NdotV = fabsf(dot(Ns,viewDir)) + EPS;
  const float NdotH = fmaxf(EPS,dot(Ns,H));
  const float NdotL = fmaxf(EPS,dot(Ns,lightDir));
  const float VdotH = fmaxf(EPS,dot(viewDir,H));
  const float LdotH = fmaxf(EPS,dot(lightDir,H));

  // Diffuse:
  vec3 diffuseColor = getRGBA(
      mat.asPhysicallyBased.baseColor, samplers, attribs, primID).xyz();

  // Fresnel
  vec3 f0 = lerp(vec3(pow2((1.f-ior)/(1.f+ior))), diffuseColor, metallic);
  vec3 F = F_Schlick(VdotH, f0);

  // Metallic materials don't reflect diffusely:
  diffuseColor = lerp(diffuseColor, vec3f(0.f), metallic);

//vec3 diffuseBRDF = diffuseColor * Fd_Lambert();
  vec3 diffuseBRDF = diffuseColor * Fd_Burley(NdotV, NdotL, LdotH, alpha);

  // GGX microfacet distribution
  float D = D_GGX(NdotH, alpha, EPS);

  // Masking-shadowing term
  float G = ((2.f * NdotL * heaviside(LdotH))
        / (NdotL + sqrtf(alpha*alpha + (1.f-alpha*alpha) * NdotL*NdotL)))
    *       ((2.f * NdotV * heaviside(VdotH))
        / (NdotV + sqrtf(alpha*alpha + (1.f-alpha*alpha) * NdotV*NdotV)));

  float denom = 4.f * NdotV * NdotL;
  vec3 specularBRDF = (F * D * G) / max(EPS,denom);

  // Clearcoat
  float Dc = D_GGX(NdotH, clearcoatAlpha, EPS);
  float Vc = V_Kelemen(LdotH, EPS);
  float Fc = F_Schlick(LdotH, 0.04f) * clearcoat;
  float Frc = (Dc * Vc) * Fc;

  return ((diffuseBRDF + specularBRDF) * (1.f - Fc) + Frc) * lightIntensity * NdotL;
}

VSNRAY_FUNC
inline vec3 evalMaterial(const dco::Material &mat,
                         const dco::Sampler *samplers,
                         const float4 *attribs,
                         unsigned primID,
                         const vec3 Ng, const vec3 Ns,
                         const vec3 viewDir, const vec3 lightDir,
                         const vec3 lightIntensity)
{
  vec3 shadedColor{0.f, 0.f, 0.f};
  if (mat.type == dco::Material::Matte) {
    vec4f color = getColor(mat, samplers, attribs, primID);

    shade_record<float> sr;
    sr.normal = Ns;
    sr.geometric_normal = Ng;
    sr.view_dir = viewDir;
    sr.tex_color = float3(1.f);
    sr.light_dir = normalize(lightDir);
    sr.light_intensity = lightIntensity;

    matte<float> vmat;
    vmat.cd() = from_rgb(color.xyz());
    vmat.kd() = 1.f;

    shadedColor = to_rgb(vmat.shade(sr));
  } else if (mat.type == dco::Material::PhysicallyBased) {
    shadedColor = evalPhysicallyBasedMaterial(mat,
                                              samplers,
                                              attribs,
                                              primID,
                                              Ng, Ns,
                                              viewDir, lightDir,
                                              lightIntensity);
  }
  return shadedColor;
}

VSNRAY_FUNC
inline Ray clipRay(Ray ray, const float4 *clipPlanes, unsigned numClipPlanes)
{
  for (unsigned i=0; i<numClipPlanes; ++i) {
    float3 N(clipPlanes[i].xyz());
    float D(clipPlanes[i].w);
    float s = dot(N,ray.dir);
    if (s != 0.f) {
      float t = (D-dot(N,ray.ori))/s;
      if (s < 0.f) ray.tmin = fmaxf(ray.tmin,t);
      else         ray.tmax = fminf(ray.tmax,t);
    }
  }
  return ray;
}

template <bool EvalOpacity>
VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersectSurfaces(
    ScreenSample &ss, Ray ray,
    const VisionarayGlobalState::DeviceObjectRegistry &onDevice,
    unsigned worldID)
{
  auto hr = intersectSurfaces(ray, onDevice.TLSs[worldID]);
  while (EvalOpacity) {
    if (!hr.hit) break;

    float2 uv{hr.u, hr.v};
    const dco::Instance &inst = onDevice.instances[hr.inst_id];
    const dco::Group &group = onDevice.groups[inst.groupID];
    const dco::Geometry &geom = onDevice.geometries[group.geoms[hr.geom_id]];
    const dco::Material &mat = onDevice.materials[group.materials[hr.geom_id]];

    float4 attribs[5];
    for (int i=0; i<5; ++i) {
      attribs[i] = getAttribute(geom, (dco::Attribute)i, hr.prim_id, uv);
    }

    float opacity = getOpacity(mat, onDevice.samplers, attribs, hr.prim_id);

    float r = ss.random();
    if (r > opacity) {
      ray.tmin = hr.t + 1e-4f;
      hr = intersectSurfaces(ray, onDevice.TLSs[worldID]);
    } else {
      break;
    }
  }
  return hr;
}

inline  VSNRAY_FUNC vec4f over(const vec4f &A, const vec4f &B)
{
  return A + (1.f-A.w)*B;
}

inline VSNRAY_FUNC vec3f hue_to_rgb(float hue)
{
  float s = saturate( hue ) * 6.0f;
  float r = saturate( fabsf(s - 3.f) - 1.0f );
  float g = saturate( 2.0f - fabsf(s - 2.0f) );
  float b = saturate( 2.0f - fabsf(s - 4.0f) );
  return vec3f(r, g, b); 
}
  
inline VSNRAY_FUNC vec3f temperature_to_rgb(float t)
{
  float K = 4.0f / 6.0f;
  float h = K - K * t;
  float v = .5f + 0.5f * t;    return v * hue_to_rgb(h);
}
  
                                  
inline VSNRAY_FUNC
vec3f heatMap(float t)
{
#if 1
  return temperature_to_rgb(t);
#else
  if (t < .25f) return lerp(vec3f(0.f,1.f,0.f),vec3f(0.f,1.f,1.f),(t-0.f)/.25f);
  if (t < .5f)  return lerp(vec3f(0.f,1.f,1.f),vec3f(0.f,0.f,1.f),(t-.25f)/.25f);
  if (t < .75f) return lerp(vec3f(0.f,0.f,1.f),vec3f(1.f,1.f,1.f),(t-.5f)/.25f);
  if (t < 1.f)  return lerp(vec3f(1.f,1.f,1.f),vec3f(1.f,0.f,0.f),(t-.75f)/.25f);
  return vec3f(1.f,0.f,0.f);
#endif
}
  
VSNRAY_FUNC
inline void print(const float3 &v)
{
  printf("float3: (%f,%f,%f)\n", v.x, v.y, v.z);
}

VSNRAY_FUNC
inline void print(const aabb &box)
{
  printf("aabb: [min: (%f,%f,%f), max: (%f,%f,%f)]\n",
      box.min.x, box.min.y, box.min.z, box.max.x, box.max.y, box.max.z);
}

VSNRAY_FUNC
inline void print(const Ray &ray)
{
  printf("ray: [ori: (%f,%f,%f), dir: (%f,%f,%f), tmin: %f, %f, mask: %u]\n",
      ray.ori.x, ray.ori.y, ray.ori.z, ray.dir.x, ray.dir.y, ray.dir.z,
      ray.tmin, ray.tmax, ray.intersectionMask);
}

} // visionaray
