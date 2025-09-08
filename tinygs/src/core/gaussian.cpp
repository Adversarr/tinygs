#include "tinygs/core/gaussian.hpp"
#include "tinygs/cuda/common_host.hpp"
#include <vector>
#include <glm/gtx/pca.hpp>
#include <glm/gtx/string_cast.hpp>

namespace tinygs {

vec3 compute_median(const std::vector<vec3>& data) {
  if (data.empty()) return vec3(0.0f);
  std::vector<float> xs, ys, zs;
  for (auto xyz : data) {
    xs.push_back(xyz.x);
    ys.push_back(xyz.y);
    zs.push_back(xyz.z);
  }
  std::sort(xs.begin(), xs.end());
  std::sort(ys.begin(), ys.end());
  std::sort(zs.begin(), zs.end());
  auto x = xs[xs.size() / 2];
  auto y = ys[ys.size() / 2];
  auto z = zs[zs.size() / 2];
  return vec3(x, y, z);
}

mat4x4 normalize_scene(
  const Gaussian3d& gs3d,
  const std::vector<std::pair<mat4x4, mat3x3>>& w2c_k_s,
  float ext_scale
) {
  // // 1. Estimate the +z axis of the scene
  // std::vector<vec3> ups;
  // for (auto [w2c, _] : w2c_k_s) {
  //   auto c2w = glm::inverse(w2c);
  //   auto z_axis = c2w * vec4(0, 0, 1, 0);
  //   ups.push_back(vec3(z_axis));
  // }
  // auto up = compute_median(ups);

  auto pc_center = compute_median(gs3d.means);
  std::vector<vec3> translated_means;
  translated_means.reserve(gs3d.means.size());
  for (auto mean : gs3d.means) {
    translated_means.push_back(mean - pc_center);
  }

  mat3x3 covariance = glm::computeCovarianceMatrix(
    translated_means.data(),
    translated_means.size(),
    pc_center
  );
  glm::vec3 evals;
  glm::mat3 evecs;
  int evcnt = glm::findEigenvaluesSymReal(covariance, evals, evecs);

  // rotation matrix
  mat3x3 rotation = evecs;
  // if det < 0, flip the last column
  if (glm::determinant(rotation) < 0) {
    rotation[2] = -rotation[2];
  }

  rotation *= ext_scale;
  mat4x4 transform = mat4x4(
    rotation[0][0], rotation[0][1], rotation[0][2], 0.0f,
    rotation[1][0], rotation[1][1], rotation[1][2], 0.0f,
    rotation[2][0], rotation[2][1], rotation[2][2], 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
  );

  vec3 translation = -rotation * pc_center;
  transform[3][0] = translation.x;
  transform[3][1] = translation.y;
  transform[3][2] = translation.z;
  log_info("transform: {}", glm::to_string(transform));
  return transform;
}
}  // namespace tinygs
