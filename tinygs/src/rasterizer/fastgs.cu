#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <memory>

#include "fastgs_ours/backward.h"
#include "fastgs_ours/forward.h"
#include "fastgs_ours/helper_math.h"
#include "tinygs/rasterizer/fastgs.hpp"
#include "utils/scope_timer.hpp"

namespace tinygs {

inline __host__ __device__ float3 make_float3(vec3 v) {
  return {v.x, v.y, v.z};
}

inline __host__ __device__ float4 make_float4(vec4 v) {
  return {v.x, v.y, v.z, v.w};
}

struct FastGSRasterizer::Impl {
  // use SoA to store the 2d gaussians.
  // 1. 3d gaussian data.
  GPUMemory<float3> primitive_mean3d;
  GPUMemory<float3> primitive_scale;
  GPUMemory<float4> primitive_rotation;
  GPUMemory<float> primitive_opacity;
  GPUMemory<float3> primitive_sh_coeffs_0;
  GPUMemory<float3> primitive_sh_coeffs_rest;
  GPUMemory<float4> w2c;           // [4, 4]
  GPUMemory<float3> cam_position;  // [3, ]
  size_t num_gaussians;

  Impl() : num_gaussians(0) {
    w2c = GPUMemory<float4>(4, true);
    cam_position = GPUMemory<float3>(1, true);
  }

  // 3. helper
  std::shared_ptr<GPUMemoryArena> arena;
  std::map<std::string, std::unique_ptr<GPUBuffer<char>>> temp_buffers;

  char* alloc(const std::string &name, size_t size) {
    auto& buffer = temp_buffers[name];
    if (buffer == nullptr || buffer->size() < size) {
      buffer = std::make_unique<GPUBuffer<char>>(arena, size);
    }

    return buffer->data();
  }

  void copy_from_ours(const GPUGaussian3d& ours) {
    const auto& mean_opacity = ours.means_opacities();
    const auto& scale = ours.scales();
    const auto& rotation = ours.rotations();
    const auto& sh_coeffs = ours.sh_coefficients();

    if (num_gaussians != ours.size()) {
      throw std::runtime_error(fmt::format("Number of gaussians not match: impl={} vs input={}", num_gaussians, ours.size()));
    }

    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(ours.size()),
        [
          o_primitive_mean3d = thrust::raw_pointer_cast(primitive_mean3d.data()),
          o_primitive_scale = thrust::raw_pointer_cast(primitive_scale.data()),
          o_primitive_rotation = thrust::raw_pointer_cast(primitive_rotation.data()),
          o_primitive_opacity = thrust::raw_pointer_cast(primitive_opacity.data()),
          o_primitive_sh_coeffs_0 = thrust::raw_pointer_cast(primitive_sh_coeffs_0.data()),
          o_primitive_sh_coeffs_rest = thrust::raw_pointer_cast(primitive_sh_coeffs_rest.data()),
          i_mean_opacity = thrust::raw_pointer_cast(mean_opacity.data()),
          i_scale = thrust::raw_pointer_cast(scale.data()),
          i_rotation = thrust::raw_pointer_cast(rotation.data()),
          i_sh_coeffs = thrust::raw_pointer_cast(sh_coeffs.data())
        ] __device__(int idx) {
      o_primitive_mean3d[idx] = make_float3(i_mean_opacity[idx].xyz());
      o_primitive_opacity[idx] = i_mean_opacity[idx].w;
      o_primitive_scale[idx] = make_float3(i_scale[idx]);
      o_primitive_rotation[idx] = make_float4(i_rotation[idx]);
      o_primitive_sh_coeffs_0[idx] = make_float3(i_sh_coeffs[idx]);
      for (int i = 0; i < kMaxSphericalHarmonicsCoefficients - 1; i++) {
        o_primitive_sh_coeffs_rest
            [idx * (kMaxSphericalHarmonicsCoefficients - 1) + i] = make_float3(
                i_sh_coeffs[idx * kMaxSphericalHarmonicsCoefficients + 1 + i]);
      }
    });
  }
};


FastGSRasterizer::FastGSRasterizer() {
    m_impl = std::make_unique<Impl>();
    m_impl->arena = m_memory_arena;
}

void FastGSRasterizer::forward(const RasterizeParamsRuntime& params) {
  TINYGS_TIMER("FastGSRasterizer::forward");
    if (!m_gaussians) {
        throw std::runtime_error("Gaussians not set");
    }

    m_impl->copy_from_ours(*m_gaussians);
    const mat4x4 &w2c = params.fwd_input.w2c;
    mat4x4 c2w = inverse(w2c);

    m_impl->w2c.at(0) = {w2c[0][0], w2c[1][0], w2c[2][0], w2c[3][0]};
    m_impl->w2c.at(1) = {w2c[0][1], w2c[1][1], w2c[2][1], w2c[3][1]};
    m_impl->w2c.at(2) = {w2c[0][2], w2c[1][2], w2c[2][2], w2c[3][2]};
    m_impl->w2c.at(3) = {w2c[0][3], w2c[1][3], w2c[2][3], w2c[3][3]};
    // m_impl->cam_position.at(0) = {w2c[3][0], w2c[3][1], w2c[3][2]};

    // m_impl->w2c.at(0) = make_float4(w2c[0]);
    // m_impl->w2c.at(1) = make_float4(w2c[1]);
    // m_impl->w2c.at(2) = make_float4(w2c[2]);
    // m_impl->w2c.at(3) = make_float4(w2c[3]);
    m_impl->cam_position.at(0) = {c2w[3][0], c2w[3][1], c2w[3][2]};

    auto per_primitive_buffers_func = [this](size_t size) -> char * {
      return m_impl->alloc("per_primitive_buffers", size);
    };

    auto per_tile_buffers_func = [this](size_t size) -> char * {
      return m_impl->alloc("per_tile_buffers", size);
    };

    auto per_instance_buffers_func = [this](size_t size) -> char * {
      return m_impl->alloc("per_instance_buffers", size);
    };

    auto per_bucket_buffers_func = [this](size_t size) -> char * {
      return m_impl->alloc("per_bucket_buffers", size);
    };

    const float fx = params.fwd_input.K[0][0];
    const float fy = params.fwd_input.K[1][1];
    const float cx = params.fwd_input.K[2][0];
    const float cy = params.fwd_input.K[2][1];
    log_debug("Render with fx={} fy={} cx={} cy={}", fx, fy, cx, cy);

    fast_gs::rasterization::forward(
        per_primitive_buffers_func,
        per_tile_buffers_func,
        per_instance_buffers_func,
        per_bucket_buffers_func,
        /* means */ m_impl->primitive_mean3d.data(),
        /* scales */ m_impl->primitive_scale.data(),
        /* rotations */ m_impl->primitive_rotation.data(),
        /* opacities */ m_impl->primitive_opacity.data(),
        /* sh_coeffs */ m_impl->primitive_sh_coeffs_0.data(),
        /* sh_coeffs_rest */ m_impl->primitive_sh_coeffs_rest.data(),
        /* w2c */ m_impl->w2c.data(),
        /* cam_position */ m_impl->cam_position.data(),
        /* image */ params.fwd_output.image.data,
        /* alpha */ params.fwd_output.alpha.data,
        /* n_primitives */ m_gaussians->size(),
        /* active_sh_bases */ kMaxSphericalHarmonicsCoefficients, // TODO: fix
        /* total_bases_sh_rest */ kMaxSphericalHarmonicsCoefficients - 1,
        /* width */ params.fwd_input.width,
        /* height */ params.fwd_input.height,
        /* fx */ fx,
        /* fy */ fy,
        /* cx */ cx,
        /* cy */ cy,
        /* near */ params.fwd_input.near,
        /* far */ params.fwd_input.far
    );
}

void FastGSRasterizer::backward(const RasterizeParamsRuntime& params) {
    if (!m_gaussians || !params.gaussians_grad) {
        throw std::runtime_error("Gaussians or gradient gaussians not set");
    }
}

FastGSRasterizer::~FastGSRasterizer() {}

void FastGSRasterizer::set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians) {
  m_gaussians = gaussians;
  const auto n_gaussians = m_gaussians->size();

  // prepare all the buffers in impl.
  m_impl->num_gaussians = n_gaussians;
  m_impl->primitive_mean3d.resize(n_gaussians);
  m_impl->primitive_opacity.resize(n_gaussians);
  m_impl->primitive_scale.resize(n_gaussians);
  m_impl->primitive_rotation.resize(n_gaussians);
  m_impl->primitive_sh_coeffs_0.resize(n_gaussians);
  m_impl->primitive_sh_coeffs_rest.resize(n_gaussians * 15);
}


}