#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "3dgs_accel/auxiliary.h"
#include "3dgs_accel/backward.h"
#include "3dgs_accel/forward.h"
#include "3dgs_accel/rasterizer.h"
#include "tinygs/cuda/common_device.cuh"
#include "tinygs/cuda/vec.hpp"
#include "tinygs/rasterizer/default.hpp"
#include "utils/scope_timer.hpp"
#include <nvtx3/nvtx3.hpp>

namespace tinygs {

__device__ vec4 bwd_normalize(const vec4 x, const vec4 dy) {
  const float norm = glm::dot(x, x);
  const vec4 y = x / (sqrtf(norm + 1e-8f) + 1e-8f);
  return (dy - y * glm::dot(y, dy)) / (norm + 1e-8f);
}

struct DefaultRasterizer::Impl {
  GPUMemory<mat4x4> viewmatrix;   // [16]
  GPUMemory<mat4x4> projmatrix;  // [16]
  GPUMemory<float3> cam_pos;     // [3]
  GPUMemory<float> background;   // [3]
  size_t num_gaussians;

  int num_buckets;
  int num_rendered;

  // Convert data to formats expected by CudaRasterizer
  thrust::device_vector<float> opacities_normalized; // sigmoid(raw_opacities)
  thrust::device_vector<float> grad_opacities_normalized; // grad of sigmoid(raw_opacities)
  thrust::device_vector<vec4> rotations_normalized;  // normalized rotation quaternion
  thrust::device_vector<vec4> grad_rotations_normalized; // grad of normalized rotation quaternion
  thrust::device_vector<vec3> exp_scales; // sigmoid(raw_opacities)
  thrust::device_vector<vec3> grad_exp_scales; // grad of exp(scales)

  thrust::device_vector<float> dL_dinvdepth; // per-pix
  thrust::device_vector<vec3> dL_dmean2D;
  thrust::device_vector<vec2> absgrad_mean2D; // per-gs absolute grad accumulator for mean2D
  thrust::device_vector<vec4> dL_dconic;
  thrust::device_vector<vec3> dL_dcolor;
  thrust::device_vector<float> dL_dinvdepth_gs;
  thrust::device_vector<float> dL_dcov3D; // [P, 6]

  // output
  thrust::device_vector<float> invdepth; // per-pix
  thrust::device_vector<int> radii; // per-gs

  // Temporary buffers
  std::map<std::string, thrust::device_vector<char>> temp_buffers;

  Impl() : num_gaussians(0) {
    viewmatrix = GPUMemory<mat4x4>(1, true);
    projmatrix = GPUMemory<mat4x4>(1, true);
    cam_pos = GPUMemory<float3>(1, true);
    background = GPUMemory<float>(3);
    // Set default background to black
    background.memset(0);
  }

  char* alloc(const std::string& name, size_t size) {
    NVTX3_FUNC_RANGE();
    auto& buffer = temp_buffers[name];
    if (buffer.size() < size) {
      buffer.resize(size);
    }
    return thrust::raw_pointer_cast(buffer.data());
  }

  char* get_allocated(const std::string& name) {
    return alloc(name, 0);
  }
};

DefaultRasterizer::DefaultRasterizer() {
  m_impl = std::make_unique<Impl>();
}

void DefaultRasterizer::forward(const RasterizeContext& ctx) {
  NVTX3_FUNC_RANGE();
  if (!m_gaussians) {
    throw std::runtime_error("Gaussians not set");
  }

  const int num_gaussians = m_gaussians->size();
  const mat4x4& w2c = ctx.fwd_input.w2c;
  const mat4x4 c2w = inverse(w2c);
  const float3 cam_pos = {c2w[3][0], c2w[3][1], c2w[3][2]};
  m_impl->cam_pos.data()[0] = cam_pos;  // managed.

  // Convert w2c to viewmatrix (column-major order)
  m_impl->viewmatrix.data()[0] = w2c;

  // Calculate projection matrix from camera intrinsics
  const float near = 0.00001f, far = 10000.f;
  //! Check the correctness of these.
  const float fx = ctx.fwd_input.K[0][0];
  const float fy = ctx.fwd_input.K[1][1];
  const float tan_fovx = ctx.fwd_input.width / (2.0f * fx);
  const float tan_fovy = ctx.fwd_input.height / (2.0f * fy);
  const float fovy = 2.0f * std::atan(tan_fovy);
  const float fovx = 2.0f * std::atan(tan_fovx);
  // log_info("fovy: {}, fovx: {}", fovy, fovx);
  // log_info("camera: x: {}, y: {}, z: {}", cam_pos.x, cam_pos.y, cam_pos.z);

  mat4x4 proj;
  {
    proj = 0;
    const float top = tan_fovy * near;
    const float bottom = -top;
    const float right = tan_fovx * near;
    const float left = -right;
    proj[0][0] = 2.0f * near / (right - left);
    proj[1][1] = 2.0f * near / (top - bottom);
    proj[2][0] = (right + left) / (right - left);
    proj[2][1] = (top + bottom) / (top - bottom);
    proj[2][3] = 1.0f;
    proj[2][2] = far / (far - near);
    proj[3][2] = -(far * near) / (far - near);
  }
  m_impl->projmatrix.data()[0] = proj * w2c;

  auto geometryBuffer_func = [this](size_t size) -> char* {
    return m_impl->alloc("geometryBuffer", size);
  };

  auto binningBuffer_func = [this](size_t size) -> char* {
    return m_impl->alloc("binningBuffer", size);
  };

  auto imageBuffer_func = [this](size_t size) -> char* {
    return m_impl->alloc("imageBuffer", size);
  };
  auto sampleBuffer_func = [this](size_t size) -> char* {
    return m_impl->alloc("sampleBuffer", size);
  };

  const auto& means = m_gaussians->means();
  const auto& scales = m_gaussians->scales();
  const auto& rotations = m_gaussians->rotations();
  const auto& opacities = m_gaussians->opacities();

  thrust::transform(
    thrust::device, rotations.begin(), rotations.end(),
    m_impl->rotations_normalized.begin(),
    [] __device__(const vec4& rot) { return glm::normalize(rot); });
  thrust::transform(
    thrust::device, opacities.begin(), opacities.end(),
    m_impl->opacities_normalized.begin(),
    [] __device__(const float& opacity) { return activate_opacity(opacity); });
  thrust::transform(
    thrust::device, scales.begin(), scales.end(),
    m_impl->exp_scales.begin(),
    [] __device__(const vec3& scale) { return activate_scale(scale); });

  m_impl->invdepth.resize(ctx.fwd_input.width * ctx.fwd_input.height);
  m_impl->radii.resize(m_gaussians->size());
  cudaMemset(thrust::raw_pointer_cast(m_impl->radii.data()), 0, m_impl->radii.size() * sizeof(int));
  cudaMemset(thrust::raw_pointer_cast(m_impl->invdepth.data()), 0, m_impl->invdepth.size() * sizeof(float));

  // For spherical harmonics: 3 colors (RGB) per Gaussian for DC component
  const int D = 1 + m_gaussians->get_sh_degree();
  constexpr int M = kMaxSphericalHarmonicsCoefficients - 1;
  float* out_color = static_cast<float*>(ctx.fwd_output.image.data);
  m_impl->num_gaussians = num_gaussians;

  // Call CudaRasterizer forward pass
   auto [num_rendered, num_buckets] = CudaRasterizer::Rasterizer::forward(
      geometryBuffer_func, binningBuffer_func, imageBuffer_func, sampleBuffer_func,
      /* P, D, M */ num_gaussians, D, M,
      /* background */ reinterpret_cast<float*>(thrust::raw_pointer_cast(m_impl->background.data())),   // [3]
      /* width, height */ ctx.fwd_input.width, ctx.fwd_input.height,
      /* means3D */ reinterpret_cast<const float*>(thrust::raw_pointer_cast(means.data())),             // [N, 3]
      /* sh0 */ thrust::raw_pointer_cast(m_gaussians->sh0().data()),    // [3*N] SoA
      /* sh1 */ thrust::raw_pointer_cast(m_gaussians->sh1().data()),    // [9*N] SoA
      /* sh2 */ thrust::raw_pointer_cast(m_gaussians->sh2().data()),    // [15*N] SoA
      /* sh3 */ thrust::raw_pointer_cast(m_gaussians->sh3().data()),    // [21*N] SoA
      /* opacities */ thrust::raw_pointer_cast(m_impl->opacities_normalized.data()),                                // [N]
      /* scales */ reinterpret_cast<const float*>(thrust::raw_pointer_cast(m_impl->exp_scales.data())),             // [N, 3]
      /* scale_modifier */ 1.0f,                                               //! TODO: check this.
      /* rotations */ reinterpret_cast<const float*>(thrust::raw_pointer_cast(m_impl->rotations_normalized.data())),  // [N, 4]
      /* viewmatrix */ reinterpret_cast<const float*>(m_impl->viewmatrix.data()),                 // [4, 4]
      /* projmatrix */ reinterpret_cast<const float*>(m_impl->projmatrix.data()),                 // [4, 4]
      /* cam_pos */ reinterpret_cast<const float*>(m_impl->cam_pos.data()),                       // [3]
      /* tan_fovx, tan_fovy */ tan_fovx, tan_fovy,
      /* prefiltered */ false,
      /* out_color */ out_color,
      /* depth */ thrust::raw_pointer_cast(m_impl->invdepth.data()),
      /* antialiasing */ false,
      /* radii */  thrust::raw_pointer_cast(m_impl->radii.data()),
#ifdef NDEBUG
      false,
#else
      /* debug */ true,
#endif
      /* metric_mode */ ctx.metric_mode,
      /* metric_map */ ctx.metric_map ? ctx.metric_map->data() : nullptr,
      /* metric_counts */ ctx.metric_counts ? ctx.metric_counts->data() : nullptr
  );
  m_impl->num_buckets = num_buckets;
  m_impl->num_rendered = num_rendered;
}

DefaultRasterizer::~DefaultRasterizer() {}

void DefaultRasterizer::backward(RasterizeContext& ctx) {
  NVTX3_FUNC_RANGE();

  uint32_t num_gaussians = m_gaussians->size();

  const auto& means = m_gaussians->means();
  const auto& scales = m_impl->exp_scales;
  const auto& rotations = m_impl->rotations_normalized;
  const auto& opacities = m_impl->opacities_normalized;

  auto &grad_means = ctx.gaussians_grad->means();
  auto &grad_scales = ctx.gaussians_grad->scales();
  auto &grad_rotations = ctx.gaussians_grad->rotations();
  auto &grad_opacities = ctx.gaussians_grad->opacities();

  // clear the internal buffers.
  auto& grad_exp_scales = m_impl->grad_exp_scales;
  auto& grad_rotations_normalized = m_impl->grad_rotations_normalized;
  auto& grad_opacities_normalized = m_impl->grad_opacities_normalized;

  const int D = 1 + m_gaussians->get_sh_degree();
  constexpr int M = kMaxSphericalHarmonicsCoefficients - 1;
  const int R = m_impl->num_rendered, B = m_impl->num_buckets;
  const int width = ctx.fwd_input.width, height = ctx.fwd_input.height;
  const float fx = ctx.fwd_input.K[0][0];
  const float fy = ctx.fwd_input.K[1][1];
  const float tan_fovx = width / (2.0f * fx);
  const float tan_fovy = height / (2.0f * fy);
  m_impl->grad_exp_scales.resize(num_gaussians);
  grad_rotations_normalized.resize(num_gaussians);
  grad_opacities_normalized.resize(num_gaussians);
  m_impl->dL_dinvdepth.resize(width * height, 0);
  m_impl->dL_dmean2D.resize(num_gaussians, vec3(0.f, 0.f, 0.f));
  m_impl->dL_dconic.resize(num_gaussians, vec4(0.f, 0.f, 0.f, 0.f));
  m_impl->dL_dcolor.resize(num_gaussians, vec3(0.f, 0.f, 0.f));
  m_impl->dL_dinvdepth_gs.resize(num_gaussians, 0.f);
  m_impl->dL_dcov3D.resize(6 * num_gaussians, 0.f);
  cudaMemset(thrust::raw_pointer_cast(grad_exp_scales.data()), 0, grad_exp_scales.size() * sizeof(vec3));
  cudaMemset(thrust::raw_pointer_cast(grad_rotations_normalized.data()), 0, grad_rotations_normalized.size() * sizeof(vec4));
  cudaMemset(thrust::raw_pointer_cast(grad_opacities_normalized.data()), 0, grad_opacities_normalized.size() * sizeof(float));
  cudaMemset(thrust::raw_pointer_cast(m_impl->dL_dinvdepth.data()), 0, m_impl->dL_dinvdepth.size() * sizeof(float));
  cudaMemset(thrust::raw_pointer_cast(m_impl->dL_dmean2D.data()), 0, m_impl->dL_dmean2D.size() * sizeof(vec3));
  m_impl->absgrad_mean2D.resize(num_gaussians, vec2(0.f, 0.f));
  cudaMemset(thrust::raw_pointer_cast(m_impl->absgrad_mean2D.data()), 0, m_impl->absgrad_mean2D.size() * sizeof(vec2));
  cudaMemset(thrust::raw_pointer_cast(m_impl->dL_dconic.data()), 0, m_impl->dL_dconic.size() * sizeof(vec4));
  cudaMemset(thrust::raw_pointer_cast(m_impl->dL_dcolor.data()), 0, m_impl->dL_dcolor.size() * sizeof(vec3));
  cudaMemset(thrust::raw_pointer_cast(m_impl->dL_dinvdepth_gs.data()), 0, m_impl->dL_dinvdepth_gs.size() * sizeof(float));
  cudaMemset(thrust::raw_pointer_cast(m_impl->dL_dcov3D.data()), 0, m_impl->dL_dcov3D.size() * sizeof(float));
  // do the backward.
  CudaRasterizer::Rasterizer::backward(
    /* P, D, M, R, B */ num_gaussians, D, M, R, B,
    /* background */ m_impl->background.data(),
    /* width, height */ width, height,
    /* means3D */ reinterpret_cast<const float*>(thrust::raw_pointer_cast(means.data())),
    /* sh0 */ thrust::raw_pointer_cast(m_gaussians->sh0().data()),
    /* sh1 */ thrust::raw_pointer_cast(m_gaussians->sh1().data()),
    /* sh2 */ thrust::raw_pointer_cast(m_gaussians->sh2().data()),
    /* sh3 */ thrust::raw_pointer_cast(m_gaussians->sh3().data()),
    /* opacities */ thrust::raw_pointer_cast(opacities.data()),
    /* scales */ reinterpret_cast<const float*>(thrust::raw_pointer_cast(scales.data())),
    /* scale_modifier */ 1.0f,
    /* rotations */ reinterpret_cast<const float*>(thrust::raw_pointer_cast(rotations.data())),
    /* viewmatrix */ reinterpret_cast<const float*>(m_impl->viewmatrix.data()),
    /* projmatrix */ reinterpret_cast<const float*>(m_impl->projmatrix.data()),
    /* cam_pos */ reinterpret_cast<const float*>(m_impl->cam_pos.data()),
    /* tan_fovx, tan_fovy */ tan_fovx, tan_fovy,
    /* radii */  thrust::raw_pointer_cast(m_impl->radii.data()),
    /* geom_buffer */ m_impl->get_allocated("geometryBuffer"),
    /* binning_buffer */ m_impl->get_allocated("binningBuffer"),
    /* image_buffer */ m_impl->get_allocated("imageBuffer"),
    /* sample_buffer */ m_impl->get_allocated("sampleBuffer"),
    /* dL_dpix*/ static_cast<float*>(ctx.grad_output.image.data),
    /* dL_dinvdepth */ thrust::raw_pointer_cast(m_impl->dL_dinvdepth.data()),
    /* dL_dmean2D */ reinterpret_cast<float*>(thrust::raw_pointer_cast(m_impl->dL_dmean2D.data())),
    /* dL_dconic */ reinterpret_cast<float*>(thrust::raw_pointer_cast(m_impl->dL_dconic.data())),
    /* dL_dopacity */ reinterpret_cast<float*>(thrust::raw_pointer_cast(grad_opacities_normalized.data())),
    /* dL_dcolor */ reinterpret_cast<float*>(thrust::raw_pointer_cast(m_impl->dL_dcolor.data())),
    /* dL_dinvdepth_gs */ reinterpret_cast<float*>(thrust::raw_pointer_cast(m_impl->dL_dinvdepth_gs.data())),
    /* dL_dmeans3D */ reinterpret_cast<float*>(thrust::raw_pointer_cast(grad_means.data())),
    /* dL_dcov3D */ reinterpret_cast<float*>(thrust::raw_pointer_cast(m_impl->dL_dcov3D.data())),
    /* dL_dsh0 */ thrust::raw_pointer_cast(ctx.gaussians_grad->sh0().data()),
    /* dL_dsh1 */ thrust::raw_pointer_cast(ctx.gaussians_grad->sh1().data()),
    /* dL_dsh2 */ thrust::raw_pointer_cast(ctx.gaussians_grad->sh2().data()),
    /* dL_dsh3 */ thrust::raw_pointer_cast(ctx.gaussians_grad->sh3().data()),
    /* dL_dscales */ reinterpret_cast<float*>(thrust::raw_pointer_cast(grad_exp_scales.data())),
    /* dL_drotations */ reinterpret_cast<float*>(thrust::raw_pointer_cast(grad_rotations_normalized.data())),
    /* absgrad_mean2D */ reinterpret_cast<float2*>(thrust::raw_pointer_cast(m_impl->absgrad_mean2D.data())),
    /* antialiasing */ false,
#ifdef NDEBUG
      false
#else
      /* debug */ true
#endif
  );


  // transform the gradients of rotations, opacities, and scales to the original space.
  thrust::for_each(
    thrust::make_counting_iterator<uint32_t>(0),
    thrust::make_counting_iterator<uint32_t>(num_gaussians),
    [
      grad_rotations_normalized = thrust::raw_pointer_cast(grad_rotations_normalized.data()),
      rotations = thrust::raw_pointer_cast(m_gaussians->rotations().data()),
      grad_rotations = thrust::raw_pointer_cast(grad_rotations.data()),
      grad_opacities_normalized = thrust::raw_pointer_cast(grad_opacities_normalized.data()),
      opacities = thrust::raw_pointer_cast(m_gaussians->opacities().data()),
      grad_opacities = thrust::raw_pointer_cast(grad_opacities.data()),
      grad_exp_scales = thrust::raw_pointer_cast(grad_exp_scales.data()),
      scales = thrust::raw_pointer_cast(m_gaussians->scales().data()),
      grad_scales = thrust::raw_pointer_cast(grad_scales.data())
    ] __device__(uint32_t i) {
      // transform rotation gradients
      const vec4 grad_rotation = grad_rotations_normalized[i];
      grad_rotations[i] = bwd_normalize(rotations[i], grad_rotation);
      
      // transform opacity gradients
      const float grad_sigmoid_opacity = grad_opacities_normalized[i];;
      grad_opacities[i] = grad_sigmoid_opacity * activate_opacity_deriv(opacities[i]);
      
      // transform scale gradients
      const vec3 grad_scale = grad_exp_scales[i];
      grad_scales[i] = grad_scale * activate_scale_deriv(scales[i]);
    }
  );

  // Update Densification Info for Default Strategy if set.
  auto& dinfo = ctx.densification_info;
  if (!dinfo) {
    return;
  }
  thrust::for_each(
    thrust::device,
    thrust::make_counting_iterator<uint32_t>(0),
    thrust::make_counting_iterator<uint32_t>(num_gaussians),
    [
      dL_dmean2D = thrust::raw_pointer_cast(m_impl->dL_dmean2D.data()),
      absgrad_mean2D = thrust::raw_pointer_cast(m_impl->absgrad_mean2D.data()),
      radii = m_impl->radii.data(), // actual rendered
      data = dinfo->data(), num_gaussians
    ] __device__ (uint32_t i) {
      if (radii[i] > 0) {
        data[i].accum_grad_mean2d += glm::length(vec2(dL_dmean2D[i].x, dL_dmean2D[i].y));
        data[i].accum_absgrad_mean2d += glm::length(absgrad_mean2D[i]);
        data[i].accum_counter += 1.0f;
      }
    });


  // log_info("Max Radii: {}", thrust::reduce(m_impl->radii.begin(), m_impl->radii.end(), 0, thrust::maximum<int>()));
}


void DefaultRasterizer::set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians) {
  m_gaussians = gaussians;
  const auto n_gaussians = m_gaussians->size();
  m_impl->num_gaussians = n_gaussians;
}

void DefaultRasterizer::set_params(const json& /*j*/) {}

json DefaultRasterizer::get_params() const {
  return json::object({{"type", "default"}});
}

}  // namespace tinygs
