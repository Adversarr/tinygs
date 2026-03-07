#include "tinygs/rasterizer/rasterizer.hpp"
#include "tinygs/rasterizer/fastgs.hpp"
#include "tinygs/rasterizer/gsplat.hpp"
#include "tinygs/rasterizer/cpu.hpp"

namespace tinygs {

RasterizerBase::RasterizerBase(BackendRuntime& runtime)
  : m_runtime(&runtime) {
}

RasterizerBase::RasterizerBase() {
}

void RasterizerBase::set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians) {
  m_gaussians = gaussians;
}

////////////////////////////// RasterizerParams Implementation //////////////////////////////

void RasterizerParams::from_json(const json& j) {
  if (j.contains("data_type")) {
    std::string data_type_str = j["data_type"];
    data_type = from_string<DataType>(data_type_str);
  }
}

json RasterizerParams::to_json() const {
  json j;
  j["data_type"] = to_string(data_type);
  return j;
}

////////////////////////////// RasterizerBase Implementation //////////////////////////////

json RasterizerBase::get_params() const {
  RasterizerParams params;
  return params.to_json();
}

void RasterizerBase::set_params(const json& j) {
  RasterizerParams params;
  params.from_json(j);
  m_params = params;
}

std::unique_ptr<RasterizerBase> create_rasterizer(const std::string& rasterizer_type,
                                                   BackendRuntime& runtime) {
  std::string lower_rasterizer_type = to_lower(rasterizer_type);
  if (lower_rasterizer_type == "fastgs") {
    return std::make_unique<FastGSRasterizer>(runtime);
  } else if (lower_rasterizer_type == "cpu") {
    return std::make_unique<CPUReferenceRasterizer>(runtime);
  } else {
    throw std::runtime_error("Unknown rasterizer type: " + rasterizer_type);
  }
}

}
