#include <iostream>
#include <random>
#include <GL/glew.h>
#include <anari/anari_cpp/ext/linalg.h>
#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/viewer_glut.h>
#include "TiltShiftCamera.h"

using namespace visionaray;

using box3_t = std::array<anari::math::float3, 2>;
namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(box3_t, ANARI_FLOAT32_BOX3);
ANARI_TYPEFOR_DEFINITION(box3_t);
} // namespace anari

void statusFunc(const void *userData,
    ANARIDevice device,
    ANARIObject source,
    ANARIDataType sourceType,
    ANARIStatusSeverity severity,
    ANARIStatusCode code,
    const char *message)
{
  (void)userData;
  if (severity == ANARI_SEVERITY_FATAL_ERROR)
    fprintf(stderr, "[FATAL] %s\n", message);
  else if (severity == ANARI_SEVERITY_ERROR)
    fprintf(stderr, "[ERROR] %s\n", message);
  else if (severity == ANARI_SEVERITY_WARNING)
    fprintf(stderr, "[WARN ] %s\n", message);
  else if (severity == ANARI_SEVERITY_PERFORMANCE_WARNING)
    fprintf(stderr, "[PERF ] %s\n", message);
  else if (severity == ANARI_SEVERITY_INFO)
    fprintf(stderr, "[INFO] %s\n", message);
}

static void anari_free(const void * /*user_data*/, const void *ptr)
{
  std::free(const_cast<void *>(ptr));
}

static anari::Array2D makeTextureData(anari::Device d, int dim)
{
  using texel = std::array<uint8_t, 3>;
  texel *data = (texel *)std::malloc(dim * dim * sizeof(texel));

  auto makeTexel = [](uint8_t v) -> texel { return {v, v, v}; };

  for (int h = 0; h < dim; h++) {
    for (int w = 0; w < dim; w++) {
      bool even = h & 1;
      if (even)
        data[h * dim + w] = w & 1 ? makeTexel(200) : makeTexel(50);
      else
        data[h * dim + w] = w & 1 ? makeTexel(50) : makeTexel(200);
    }
  }

  return anariNewArray2D(
      d, data, &anari_free, nullptr, ANARI_UFIXED8_VEC3, dim, dim);
}

static anari::Surface makeCones(anari::Device d)
{
  struct {
    int numCones{10};
    float positionRange{1.f};
    float arrowRadius{0.125f};
    float opacity{1.f};
    bool useRandomSeed{false};
    bool caps{true};
  } config;

  std::mt19937 rng;
  if (config.useRandomSeed)
    rng.seed(std::random_device()());
  else
    rng.seed(0);
  std::uniform_real_distribution<float> pos_dist(0.f, config.positionRange);

  std::vector<anari::math::float3> positions(2 * config.numCones);
  std::vector<anari::math::uint2> indices(config.numCones);

  for (auto &s : positions) {
    s.x = pos_dist(rng);
    s.y = pos_dist(rng);
    s.z = pos_dist(rng);
  }

  for (int i = 0; i < config.numCones; i++)
    indices[i] = anari::math::uint2(2 * i) + anari::math::uint2(0, 1);

  std::vector<anari::math::float2> radii(config.numCones);
  std::fill(radii.begin(), radii.end(), anari::math::float2(config.arrowRadius, 0.f));

  auto geom = anari::newObject<anari::Geometry>(d, "cone");
  anari::setAndReleaseParameter(d,
      geom,
      "vertex.position",
      anari::newArray1D(d, positions.data(), positions.size()));
  anari::setAndReleaseParameter(d,
      geom,
      "vertex.radius",
      anari::newArray1D(d, (float *)radii.data(), radii.size() * 2));
  anari::setParameter(d, geom, "caps", config.caps ? "caps" : "none");

  std::uniform_real_distribution<float> col_dist(0.f, 1.f);

  std::vector<anari::math::float4> colors(2 * config.numCones);

  for (auto &s : colors) {
    s.x = col_dist(rng);
    s.y = col_dist(rng);
    s.z = col_dist(rng);
    s.w = 1.f;
  }

  anari::setAndReleaseParameter(d,
      geom,
      "vertex.color",
      anari::newArray1D(d, colors.data(), colors.size()));

  anari::commitParameters(d, geom);

  auto surface = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, surface, "geometry", geom);

  auto mat = anari::newObject<anari::Material>(d, "matte");
  anari::setParameter(d, mat, "color", "color");
  anari::setParameter(d, mat, "opacity", config.opacity);
  anari::setParameter(d, mat, "alphaMode", "blend");
  anari::commitParameters(d, mat);
  anari::setAndReleaseParameter(d, surface, "material", mat);

  anari::commitParameters(d, surface);

  return surface;
}

static anari::Surface makeCylinders(anari::Device d)
{
  struct {
    int numCylinders{10};
    float positionRange{1.f};
    float radius{0.025f};
    float opacity{1.f};
    bool useRandomSeed{false};
    bool caps{true};
  } config;

  std::mt19937 rng;
  if (config.useRandomSeed)
    rng.seed(std::random_device()());
  else
    rng.seed(0);
  std::uniform_real_distribution<float> pos_dist(0.f, config.positionRange);

  std::vector<anari::math::float3> positions(2 * config.numCylinders);
  std::vector<anari::math::uint2> indices(config.numCylinders);

  for (auto &s : positions) {
    s.x = pos_dist(rng);
    s.y = pos_dist(rng);
    s.z = pos_dist(rng);
  }

  for (int i = 0; i < config.numCylinders; i++)
    indices[i] = anari::math::uint2(2 * i) + anari::math::uint2(0, 1);

  auto geom = anari::newObject<anari::Geometry>(d, "cylinder");
  anari::setAndReleaseParameter(d,
      geom,
      "vertex.position",
      anari::newArray1D(d, positions.data(), positions.size()));
  anari::setParameter(d, geom, "radius", config.radius);
  anari::setParameter(d, geom, "caps", config.caps ? "both" : "none");

  std::uniform_real_distribution<float> col_dist(0.f, 1.f);

  std::vector<anari::math::float4> colors(2 * config.numCylinders);

  for (auto &s : colors) {
    s.x = col_dist(rng);
    s.y = col_dist(rng);
    s.z = col_dist(rng);
    s.w = 1.f;
  }

  anari::setAndReleaseParameter(d,
      geom,
      "vertex.color",
      anari::newArray1D(d, colors.data(), colors.size()));

  anari::commitParameters(d, geom);

  auto surface = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, surface, "geometry", geom);

  auto mat = anari::newObject<anari::Material>(d, "matte");
  anari::setParameter(d, mat, "color", "color");
  anari::setParameter(d, mat, "opacity", config.opacity);
  anari::setParameter(d, mat, "alphaMode", "blend");
  anari::commitParameters(d, mat);
  anari::setAndReleaseParameter(d, surface, "material", mat);

  anari::commitParameters(d, surface);

  return surface;
}

static anari::Surface makeBezierCurves(anari::Device d)
{
  anari::math::float3 vertices[] = {
    {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.5f}, {2.0f, 0.5f, 0.0f}, {3.0f, 0.0f, 0.0f},
    //{0.0f, 0.0f, 0.0f}, {0.5f, 1.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {1.5f, 0.0f, 0.0f},
    //{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {2.0f, 1.0f, 0.0f}, {3.0f, 0.0f, 0.0f},
    //{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {2.0f, 2.0f, 0.0f}, {3.0f, 3.0f, 0.0f},
    //{0.0f, 0.0f, 0.0f}, {0.0f, 20.0f, 1.0f}, {0.0f, 0.0f, 2.0f}, {0.0f, 0.0f, 3.0f},
    //{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 2.0f}, {0.0f, 0.0f, 3.0f},
    //{-1.0f, 2.0f, 0.0f}, {0.0f, -2.0f, 0.0f}, {1.0f, 4.0f, 0.0f}, {2.0f, -4.0f, 0.0f},
    //{-5.0f, 2.0f, 0.0f}, {0.0f, -0.5f, 1.0f}, {1.0f, 4.0f, 0.0f}, {2.0f, -4.0f, 8.0f},
    //{0.0f, 10.0f, 0.0f}, {0.0f, 11.0f, 11.0f}, {0.0f, -11.0f, 0.0f}, {0.0f, -10.0f, 0.0f},
    //{3.0f, 8.0f, 0.0f}, {0.0f, 15.0f, 11.0f}, {0.0f, -11.0f, 0.0f}, {7.0f, -10.0f, 0.0f},
    //{-10.0f, 20.0f, 0.0f}, {0.0f, -20.0f, 0.0f}, {1.0f, 4.0f, 0.0f}, {2.0f, -4.0f, 0.0f},
    //{0.0f, 20.0f, 0.0f}, {1.0f, 2.0f, 4.5f}, {2.0f, 3.5f, 0.0f}, {3.0f, 4.0f, 0.0f},
  };

  unsigned indices[] = {0};//,4,8,12,16,20,24,28,32,36,40};

  auto geom = anari::newObject<anari::Geometry>(d, "bezierCurve");
  anari::setAndReleaseParameter(d,
      geom,
      "vertex.position",
      anari::newArray1D(d, vertices, sizeof(vertices) / sizeof(vertices[0])));
  anari::setAndReleaseParameter(d,
      geom,
      "primitive.index",
      anari::newArray1D(d, indices, sizeof(indices) / sizeof(indices[0])));
  anari::setParameter(d, geom, "radius", 0.1f);
  anari::commitParameters(d, geom);

  auto surface = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, surface, "geometry", geom);

  auto mat = anari::newObject<anari::Material>(d, "matte");
  anari::setParameter(d, mat, "color", anari::math::float3(0.f, 0.4f, 0.9f));
  anari::setParameter(d, mat, "alphaMode", "blend");
  anari::commitParameters(d, mat);
  anari::setAndReleaseParameter(d, surface, "material", mat);

  anari::commitParameters(d, surface);

  return surface;
}

static anari::Surface makeCurves(anari::Device d)
{
  // This code is adapted from the OSPRay 'streamlines' example:
  //   https://github.com/ospray/ospray/blob/fdda0889f9143a8b20f26389c22d1691f1a6a527/apps/common/ospray_testing/builders/Streamlines.cpp

  std::vector<anari::math::float3> positions;
  std::vector<float> radii;
  std::vector<unsigned int> indices;
  std::vector<anari::math::float4> colors;

  auto addPoint = [&](const anari::math::float4 &p) {
    positions.emplace_back(p[0], p[1], p[2]);
    radii.push_back(p[3]);
  };

  std::mt19937 rng(0);
  std::uniform_real_distribution<float> radDist(0.5f, 1.5f);
  std::uniform_real_distribution<float> stepDist(0.001f, 0.1f);
  std::uniform_real_distribution<float> sDist(0, 360);
  std::uniform_real_distribution<float> dDist(360, 720);
  std::uniform_real_distribution<float> freqDist(0.5f, 1.5f);

  // create multiple lines
  int numLines = 100;
  for (int l = 0; l < numLines; l++) {
    int dStart = sDist(rng);
    int dEnd = dDist(rng);
    float radius = radDist(rng);
    float h = 0;
    float hStep = stepDist(rng);
    float f = freqDist(rng);

    float r = (720 - dEnd) / 360.f;
    anari::math::float4 c(r, 1 - r, 1 - r / 2, 1.f);

    // spiral up with changing radius of curvature
    for (int d = dStart; d < dStart + dEnd; d += 10, h += hStep) {
      anari::math::float3 p, q;
      float startRadius, endRadius;

      p.x = radius * std::sin(d * M_PI / 180.f);
      p.y = h - 2;
      p.z = radius * std::cos(d * M_PI / 180.f);
      startRadius = 0.015f * std::sin(f * d * M_PI / 180) + 0.02f;

      q.x = (radius - 0.05f) * std::sin((d + 10) * M_PI / 180.f);
      q.y = h + hStep - 2;
      q.z = (radius - 0.05f) * std::cos((d + 10) * M_PI / 180.f);
      endRadius = 0.015f * std::sin(f * (d + 10) * M_PI / 180) + 0.02f;
      if (d == dStart) {
        const auto rim = lerp(q, p, 1.f + endRadius / length(q - p));
        const auto cap = lerp(p, rim, 1.f + startRadius / length(rim - p));
        addPoint(anari::math::float4(cap, 0.f));
        addPoint(anari::math::float4(rim, 0.f));
        addPoint(anari::math::float4(p, startRadius));
        addPoint(anari::math::float4(q, endRadius));
        indices.push_back(positions.size() - 4);
        colors.push_back(c);
        colors.push_back(c);
      } else if (d + 10 < dStart + dEnd && d + 20 > dStart + dEnd) {
        const auto rim = lerp(p, q, 1.f + startRadius / length(p - q));
        const auto cap = lerp(q, rim, 1.f + endRadius / length(rim - q));
        addPoint(anari::math::float4(p, startRadius));
        addPoint(anari::math::float4(q, endRadius));
        addPoint(anari::math::float4(rim, 0.f));
        addPoint(anari::math::float4(cap, 0.f));
        indices.push_back(positions.size() - 7);
        indices.push_back(positions.size() - 6);
        indices.push_back(positions.size() - 5);
        indices.push_back(positions.size() - 4);
        colors.push_back(c);
        colors.push_back(c);
      } else if ((d != dStart && d != dStart + 10) && d + 20 < dStart + dEnd) {
        addPoint(anari::math::float4(p, startRadius));
        indices.push_back(positions.size() - 4);
      }
      colors.push_back(c);
      radius -= 0.05f;
    }
  }

  auto geom = anari::newObject<anari::Geometry>(d, "curve");
  anari::setAndReleaseParameter(d,
      geom,
      "vertex.position",
      anari::newArray1D(d, positions.data(), positions.size()));
  anari::setAndReleaseParameter(d,
      geom,
      "vertex.radius",
      anari::newArray1D(d, radii.data(), radii.size()));
  anari::setAndReleaseParameter(d,
      geom,
      "vertex.color",
      anari::newArray1D(d, colors.data(), colors.size()));
  anari::setAndReleaseParameter(d,
      geom,
      "primitive.index",
      anari::newArray1D(d, indices.data(), indices.size()));
  anari::commitParameters(d, geom);

  auto surface = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, surface, "geometry", geom);

  auto mat = anari::newObject<anari::Material>(d, "matte");
  anari::setParameter(d, mat, "color", "color");
  anari::setParameter(d, mat, "alphaMode", "blend");
  anari::commitParameters(d, mat);
  anari::setAndReleaseParameter(d, surface, "material", mat);

  anari::commitParameters(d, surface);

  return surface;
}

static anari::Surface makePlane(anari::Device d, box3_t bounds)
{
  anari::math::float3 vertices[4];
  float f = length(bounds[1]-bounds[0]);
  vertices[0] = { bounds[0][0] - f, bounds[0][1] - (0.1 * f), bounds[1][2] + f };
  vertices[1] = { bounds[1][0] + f, bounds[0][1] - (0.1 * f), bounds[1][2] + f };
  vertices[2] = { bounds[1][0] + f, bounds[0][1] - (0.1 * f), bounds[0][2] - f };
  vertices[3] = { bounds[0][0] - f, bounds[0][1] - (0.1 * f), bounds[0][2] - f };

  anari::math::float2 texcoords[4] = {
      {0.f, 0.f},
      {0.f, 1.f},
      {1.f, 1.f},
      {1.f, 0.f},
  };

  auto geom = anari::newObject<anari::Geometry>(d, "quad");
  anari::setAndReleaseParameter(d,
      geom,
      "vertex.position",
      anari::newArray1D(d, vertices, 4));
  anari::setAndReleaseParameter(d,
      geom,
      "vertex.attribute0",
      anari::newArray1D(d, texcoords, 4));
  anari::commitParameters(d, geom);

  auto surface = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, surface, "geometry", geom);

  auto tex = anari::newObject<anari::Sampler>(d, "image2D");
  anari::setAndReleaseParameter(d, tex, "image", makeTextureData(d, 8));
  anari::setParameter(d, tex, "inAttribute", "attribute0");
  anari::setParameter(d, tex, "wrapMode1", "clampToEdge");
  anari::setParameter(d, tex, "wrapMode2", "clampToEdge");
  anari::setParameter(d, tex, "filter", "nearest");
  anari::commitParameters(d, tex);

  auto mat = anari::newObject<anari::Material>(d, "matte");
  anari::setAndReleaseParameter(d, mat, "color", tex);
  anari::setParameter(d, mat, "alphaMode", "blend");
  anari::commitParameters(d, mat);
  anari::setAndReleaseParameter(d, surface, "material", mat);

  anari::commitParameters(d, surface);

  return surface;
}

static anari::Instance makePlaneInstance(anari::Device d, const box3_t &bounds)
{
  auto surface = makePlane(d, bounds);

  auto group = anari::newObject<anari::Group>(d);
  anari::setAndReleaseParameter(
      d, group, "surface", anari::newArray1D(d, &surface));
  anari::commitParameters(d, group);

  anari::release(d, surface);

  auto inst = anari::newObject<anari::Instance>(d, "transform");
  anari::setAndReleaseParameter(d, inst, "group", group);
  anari::commitParameters(d, inst);

  return inst;
}

struct Renderer : viewer_glut
{
  Renderer();
  ~Renderer();

  void on_display() override;
  void on_mouse_move(const visionaray::mouse_event &event) override;
  void on_key_press(const visionaray::key_event &event) override;
  void on_resize(int w, int h) override;

  TiltShiftCamera cam;

  struct {
    anari::Library library{nullptr};
    anari::Device device{nullptr};
    anari::World world{nullptr};
    anari::Renderer renderer{nullptr};
    anari::Frame frame{nullptr};
    anari::Light light{nullptr};
  } anari;
};

Renderer::Renderer()
{
  anari.library = anari::loadLibrary("visionaray", statusFunc);
  anari.device = anariNewDevice(anari.library, "default");
  anari.world = anari::newObject<anari::World>(anari.device);
  anari.renderer = anari::newObject<anari::Renderer>(anari.device, "default");
  //anari.renderer = anari::newObject<anari::Renderer>(anari.device, "raycast");
  cam = TiltShiftCamera(anari.device);

  float aspect = width() / float(height());
  cam.perspective(60.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
  cam.set_viewport(0, 0, width(), height());

  auto surf = makeCones(anari.device);
  //auto surf = makeCylinders(anari.device);
  //auto surf = makeCurves(anari.device);
  //auto surf = makeBezierCurves(anari.device);
  anari::setAndReleaseParameter(
      anari.device, anari.world, "surface", anari::newArray1D(anari.device, &surf));
  anari::commitParameters(anari.device, anari.world);

  box3_t bounds;
  bounds[0] = {0.f, 0.f, 0.f};
  bounds[1] = {1.f, 1.f, 1.f};
  //anari::getProperty(anari.device, anari.world, "bounds", bounds, ANARI_WAIT);
  cam.viewAll(bounds);
  cam.commit();

  if (1) {
    auto planeInst = makePlaneInstance(anari.device, bounds);
    anari::setAndReleaseParameter(
        anari.device, anari.world, "instance", anari::newArray1D(anari.device, &planeInst));
    anari::release(anari.device, planeInst);
  }

  anari.light = anari::newObject<anari::Light>(anari.device, "directional");
  anari::setParameter(anari.device, anari.light, "direction",
      anari::math::float3(1.f, -1.f, -1.f));
  anari::setParameter(anari.device, anari.light, "irradiance", 1.f);
  anari::setParameter(anari.device, anari.light, "color",
      anari::math::float3(1.f, 1.f, 1.f));
  anari::setAndReleaseParameter(anari.device,
      anari.world,
      "light",
      anari::newArray1D(anari.device, &anari.light, 1));

  anari::setParameter(anari.device, anari.renderer, "background",
      anari::math::float4(0.6f, 0.6f, 0.6f, 1.0f));

  //anari::setParameter(anari.device, anari.renderer, "ambientRadiance", 0.f);
  //anari::setParameter(anari.device, anari.renderer, "mode", "Ng");
  //anari::setParameter(anari.device, anari.renderer, "mode", "Ns");
  //anari::setParameter(anari.device, anari.renderer, "heatMapEnabled", true);
  //anari::setParameter(anari.device, anari.renderer, "taa", true);
  //anari::setParameter(anari.device, anari.renderer, "taaAlpha", 0.1f);
  //anari::math::float4 clipPlane[] = {{ 0.f, 0.f, -1.f, 0.f }};
  anari::math::float4 clipPlane[] = {{ 0.707f, 0.f, -0.707f, 0.4f }};
  anari::setAndReleaseParameter(
      anari.device, anari.renderer, "clipPlane",
      anari::newArray1D(anari.device, clipPlane, 1));

  anari::commitParameters(anari.device, anari.renderer);
}

Renderer::~Renderer()
{
  cam.~TiltShiftCamera();
  anari::release(anari.device, anari.world);
  anari::release(anari.device, anari.renderer);
  anari::release(anari.device, anari.frame);
  anari::release(anari.device, anari.device);
  anari::unloadLibrary(anari.library);
}

void Renderer::on_display()
{
  anari::render(anari.device, anari.frame);
  anari::wait(anari.device, anari.frame);

  auto channelColor = anari::map<uint32_t>(anari.device, anari.frame, "channel.color");

  glDrawPixels(width(), height(), GL_RGBA, GL_UNSIGNED_BYTE, channelColor.data);

  anari::unmap(anari.device, anari.frame, "channel.color");

  viewer_glut::on_display();
}

void Renderer::on_mouse_move(const visionaray::mouse_event &event)
{
  if (event.buttons() == mouse::NoButton)
    return;

  viewer_glut::on_mouse_move(event);

  cam.commit();
}

void Renderer::on_key_press(const visionaray::key_event &event)
{
  viewer_glut::on_key_press(event);

  static float D = 0.4f;

  if (event.key() == 'a')
    D -= .1f;
  else if (event.key() == 'b')
    D += .1f;

  anari::math::float4 clipPlane[] = {{ 0.707f, 0.f, -0.707f, D }};
  anari::setAndReleaseParameter(
      anari.device, anari.renderer, "clipPlane",
      anari::newArray1D(anari.device, clipPlane, 1));

  anari::commitParameters(anari.device, anari.renderer);
}

void Renderer::on_resize(int w, int h)
{
  if (!anari.frame) {
    anari.frame = anari::newObject<anari::Frame>(anari.device);
  }

  cam.set_viewport(0, 0, w, h);
  float aspect = w / float(h);
  cam.perspective(60.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
  cam.commit();

  anari::math::uint2 size(w, h);

  anari::setParameter(anari.device, anari.frame, "world", anari.world);
  anari::setParameter(anari.device, anari.frame, "renderer", anari.renderer);
  anari::setParameter(anari.device, anari.frame, "camera", cam.getAnariHandle());
  anari::setParameter(anari.device, anari.frame, "channel.color", ANARI_UFIXED8_VEC4);
  anari::setParameter(anari.device, anari.frame, "size", size);
  anari::commitParameters(anari.device, anari.frame);

  viewer_glut::on_resize(w, h);
}

int main(int argc, char *argv[])
{
  Renderer rend;
  try {
    rend.init(argc, argv);
  } catch (...) {
    return EXIT_FAILURE;
  }

  rend.add_manipulator(std::make_shared<arcball_manipulator>(rend.cam, mouse::Left));
  rend.add_manipulator(std::make_shared<pan_manipulator>(rend.cam, mouse::Middle) );
  // Additional "Alt + LMB" pan manipulator for setups w/o middle mouse button
  rend.add_manipulator(std::make_shared<pan_manipulator>(rend.cam, mouse::Left, keyboard::Alt));
  rend.add_manipulator(std::make_shared<zoom_manipulator>(rend.cam, mouse::Right));
  rend.event_loop();
}
