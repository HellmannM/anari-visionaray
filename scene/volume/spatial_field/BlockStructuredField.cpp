
#include "BlockStructuredField.h"
#include "for_each.h"
#include "array/Array3D.h"

namespace visionaray {

BlockStructuredField::BlockStructuredField(VisionarayGlobalState *d)
    : SpatialField(d)
{
  vfield.type = dco::SpatialField::BlockStructured;
}

void BlockStructuredField::commit()
{
  m_params.cellWidth = getParamObject<helium::Array1D>("cellWidth");
  m_params.blockBounds = getParamObject<helium::Array1D>("block.bounds");
  m_params.blockLevel = getParamObject<helium::Array1D>("block.level");
  m_params.blockData = getParamObject<helium::ObjectArray>("block.data");
  m_params.gridOrigin = getParam<float3>("gridOrigin", float3(0.f));
  m_params.gridSpacing = getParam<float3>("gridSpacing", float3(1.f));

  if (!m_params.blockBounds) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'block.bounds' on amr spatial field");
    return;
  }

  if (!m_params.blockLevel) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'block.level' on amr spatial field");
    return;
  }

  if (!m_params.blockData) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'block.data' on amr spatial field");
    return;
  }

  size_t numLevels = m_params.cellWidth->totalSize();
  size_t numBlocks = m_params.blockData->totalSize();
  auto *blockBounds = m_params.blockBounds->beginAs<aabbi>();
  auto *blockLevels = m_params.blockLevel->beginAs<int>();
  auto *blockData = (Array3D **)m_params.blockData->handlesBegin();

  m_blocks.resize(numBlocks);

  std::vector<aabb> levelBounds(numLevels);
  for (auto &lb : levelBounds) {
    lb.invalidate();
  }

  for (size_t i=0; i<numBlocks; ++i) {
    m_blocks[i].bounds = blockBounds[i];
    m_blocks[i].level = blockLevels[i];
    m_blocks[i].scalarOffset = m_scalars.size();
    m_blocks[i].valueRange = box1f(FLT_MAX,-FLT_MAX);

    const Array3D *bd = *(blockData+i);

    for (unsigned z=0;z<bd->size().z;++z) {
      for (unsigned y=0;y<bd->size().y;++y) {
        for (unsigned x=0;x<bd->size().x;++x) {
          // TODO: can we actually iterate linearly here?!
          size_t index = z*size_t(bd->size().x)*bd->size().y 
                       + y*bd->size().x
                       + x;
          float f = bd->dataAs<float>()[index];
          m_scalars.push_back(f);
          m_blocks[i].valueRange.extend(f);
        }
      }
    }

    if (levelBounds.size() <= m_blocks[i].level) {
      levelBounds.resize(m_blocks[i].level + 1);
      levelBounds[m_blocks[i].level].invalidate();
    }
    levelBounds[m_blocks[i].level].insert(m_blocks[i].worldBounds());
  }

  aabb voxelBounds;
  voxelBounds.invalidate();
  for (size_t i = 0; i < levelBounds.size(); ++i) {
    voxelBounds.insert(levelBounds[i]);
  }
  m_bounds.min = m_params.gridOrigin;
  m_bounds.max = m_params.gridOrigin + (voxelBounds.max-voxelBounds.min) * m_params.gridSpacing;

  // do this now that m_scalars doesn't change anymore:
  for (size_t i=0; i<numBlocks; ++i) {
    m_blocks[i].scalarsBuffer = m_scalars.devicePtr();
  }

  // sampling BVH

  binned_sah_builder builder;
  builder.enable_spatial_splits(false);

#ifdef WITH_CUDA
  auto hostBVH = builder.build(
    index_bvh<dco::Block>{}, m_blocks.data(), m_blocks.size());

  m_samplingBVH = cuda_index_bvh<dco::Block>(hostBVH);

  vfield.asBlockStructured.samplingBVH = m_samplingBVH.ref();
#else
  m_samplingBVH = builder.build(
    index_bvh<dco::Block>{}, m_blocks.data(), m_blocks.size());

  vfield.asBlockStructured.samplingBVH = m_samplingBVH.ref();
#endif

  mat3 S = mat3::scaling(voxelBounds.size()/m_bounds.size());
  vec3 T = voxelBounds.min-m_bounds.min;
  vfield.voxelSpaceTransform = mat4x3(S,T);

  setStepSize(length(bounds().max-bounds().min)/50.f);

  buildGrid();

  vfield.gridAccel = m_gridAccel.visionarayAccel();

  dispatch();
}

bool BlockStructuredField::isValid() const
{
  return m_samplingBVH.num_nodes();
}

aabb BlockStructuredField::bounds() const
{
  return m_bounds;
}

void BlockStructuredField::buildGrid()
{
#ifdef WITH_CUDA
  return;
#endif
  int3 dims{128, 128, 128};
  box3f worldBounds = {bounds().min,bounds().max};
  m_gridAccel.init(dims, worldBounds);

  dco::GridAccel &vaccel = m_gridAccel.visionarayAccel();

  size_t numBlocks = m_blocks.size();
  parallel::for_each(deviceState()->threadPool, 0, numBlocks,
    [&](size_t blockID) {
      const auto &block = m_blocks[blockID];
      int cellSize = block.cellSize();
      for (int z=0; z<block.numCells().z; ++z) {
        for (int y=0; y<block.numCells().y; ++y) {
          for (int x=0; x<block.numCells().x; ++x) {
            vec3i cellID(x,y,z);
            vec3i cell_lower = (block.bounds.min+cellID)*cellSize;
            vec3i cell_upper = (block.bounds.min+cellID+vec3i(1))*cellSize;
            aabb cellBounds(vec3f(cell_lower)-vec3f(cellSize*0.5f),
                            vec3f(cell_upper)+vec3f(cellSize*0.5f)); // +/- filterDomain
            // transform to world space (..TODO: untested!)
            cellBounds.min *= m_params.gridSpacing;
            cellBounds.max *= m_params.gridSpacing;
            cellBounds.min += m_params.gridOrigin;
            cellBounds.max += m_params.gridOrigin;
            float scalar = block.getScalar(x,y,z);

            const vec3i loMC = projectOnGrid(cellBounds.min,dims,worldBounds);
            const vec3i upMC = projectOnGrid(cellBounds.max,dims,worldBounds);

            for (int mcz=loMC.z; mcz<=upMC.z; ++mcz) {
              for (int mcy=loMC.y; mcy<=upMC.y; ++mcy) {
                for (int mcx=loMC.x; mcx<=upMC.x; ++mcx) {
                  const vec3i mcID(mcx,mcy,mcz);
                  updateMC(mcID,dims,scalar,vaccel.valueRanges);
                }
              }
            }
          }
        }
      }
  });
}

} // namespace visionaray
