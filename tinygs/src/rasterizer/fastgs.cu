// https://github.com/MrNeRF/gaussian-splatting-cuda

#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <memory>

#include "fastgs_ours/backward.h"
#include "fastgs_ours/forward.h"
#include "fastgs_ours/helper_math.h"
#include "tinygs/rasterizer/fastgs.hpp"
#include "tinygs/cuda/vec.hpp"
#include "utils/scope_timer.hpp"

namespace tinygs {

struct FastGSRasterizer::Impl {
  GPUMemory<float4> w2c;           // [4, 4]
  GPUMemory<float4> w2c_grad;      // [4, 4]
  GPUMemory<float3> cam_position;  // [3, ]
  size_t num_gaussians;

  // 3. helper
  int n_visible_primitives, n_instances, n_buckets;
  int primitive_primitive_indices_selector, instance_primitive_indices_selector;
  // std::shared_ptr<GPUMemoryArena> arena;
  // std::map<std::string, std::unique_ptr<GPUBuffer<char>>> temp_buffers;
  std::map<std::string, thrust::device_vector<char>> temp_buffers;

  Impl() : num_gaussians(0) {
    w2c = GPUMemory<float4>(4, true);
    w2c_grad = GPUMemory<float4>(4, true);
    cam_position = GPUMemory<float3>(1, true);
  }

  char* alloc(const std::string &name, size_t size) { 
    TINYGS_TIMER("FastGSRasterizer::Impl::alloc");
    auto& buffer = temp_buffers[name];
    if (size > buffer.size()) {
      buffer.resize(size);
    }
    return thrust::raw_pointer_cast(buffer.data());
    // if (buffer == nullptr || buffer->size() < size) {
    //   buffer = std::make_unique<GPUBuffer<char>>(arena, size);
    // }
    // return buffer->data();
  }
};

FastGSRasterizer::FastGSRasterizer() {
    m_impl = std::make_unique<Impl>();
    // m_impl->arena = m_memory_arena;
}

void FastGSRasterizer::forward(const RasterizeContext& ctx) {
  TINYGS_TIMER("FastGSRasterizer::forward");
    if (!m_gaussians) {
        throw std::runtime_error("Gaussians not set");
    }

    const mat4x4 &w2c = ctx.fwd_input.w2c;
    mat4x4 c2w = inverse(w2c);

    m_impl->w2c.at(0) = {w2c[0][0], w2c[1][0], w2c[2][0], w2c[3][0]};
    m_impl->w2c.at(1) = {w2c[0][1], w2c[1][1], w2c[2][1], w2c[3][1]};
    m_impl->w2c.at(2) = {w2c[0][2], w2c[1][2], w2c[2][2], w2c[3][2]};
    m_impl->w2c.at(3) = {w2c[0][3], w2c[1][3], w2c[2][3], w2c[3][3]};
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

    const float fx = ctx.fwd_input.K[0][0];
    const float fy = ctx.fwd_input.K[1][1];
    const float cx = ctx.fwd_input.K[2][0];
    const float cy = ctx.fwd_input.K[2][1];

    const auto& means = m_gaussians->means();
    const auto& scales = m_gaussians->scales();
    const auto& rotations = m_gaussians->rotations();
    const auto& opacities = m_gaussians->opacities();
    const auto& sh_coeffs_0 = m_gaussians->sh_coefficient_0();
    const auto& sh_coeffs_rest = m_gaussians->sh_coefficients_rest();

    auto [n_visible_primitives, n_instances, n_buckets,
          primitive_primitive_indices_selector,
          instance_primitive_indices_selector] =
        fast_gs::rasterization::forward( //
            per_primitive_buffers_func,  //
            per_tile_buffers_func,       //
            per_instance_buffers_func,   //
            per_bucket_buffers_func,     //
            /* means */ reinterpret_cast<const float3*>(thrust::raw_pointer_cast(means.data())),
            /* scales */ reinterpret_cast<const float3*>(thrust::raw_pointer_cast(scales.data())),
            /* rotations */ reinterpret_cast<const float4*>(thrust::raw_pointer_cast(rotations.data())),
            /* opacities */ thrust::raw_pointer_cast(opacities.data()),
            /* sh_coeffs */ reinterpret_cast<const float3*>(thrust::raw_pointer_cast(sh_coeffs_0.data())),
            /* sh_coeffs_rest */ reinterpret_cast<const float3*>(thrust::raw_pointer_cast(sh_coeffs_rest.data())),
            /* w2c */ m_impl->w2c.data(),
            /* cam_position */ m_impl->cam_position.data(),
            /* image */ ctx.fwd_output.image.data,
            /* alpha */ ctx.fwd_output.alpha.data,
            /* n_primitives */ m_gaussians->size(),
            /* active_sh_bases */ m_gaussians->get_sh_degree(),
            /* total_bases_sh_rest */ kMaxSphericalHarmonicsCoefficients - 1,
            /* width */ ctx.fwd_input.width,
            /* height */ ctx.fwd_input.height,
            /* fx */ fx,
            /* fy */ fy,
            /* cx */ cx,
            /* cy */ cy,
            /* near */ ctx.fwd_input.near,
            /* far */ ctx.fwd_input.far);
    m_impl->n_visible_primitives = n_visible_primitives;
    m_impl->n_instances = n_instances;
    m_impl->n_buckets = n_buckets;
    m_impl->primitive_primitive_indices_selector = primitive_primitive_indices_selector;
    m_impl->instance_primitive_indices_selector = instance_primitive_indices_selector;
}

void FastGSRasterizer::backward(const RasterizeContext &params) {
  TINYGS_TIMER("FastGSRasterizer::backward");
  if (!m_gaussians || !params.gaussians_grad) {
    throw std::runtime_error("Gaussians or gradient gaussians not set");
  }
  const auto n_gaussians = m_gaussians->size();
  char* grad_mean2d_helper = m_impl->alloc("grad_mean2d_helper", sizeof(float2) * n_gaussians);
  char* grad_conic_helper = m_impl->alloc("grad_conic_helper", sizeof(float3) * n_gaussians);
  CUDA_CHECK_THROW(cudaMemsetAsync(grad_mean2d_helper, 0, sizeof(float2) * n_gaussians, cudaStreamDefault));
  CUDA_CHECK_THROW(cudaMemsetAsync(grad_conic_helper, 0, sizeof(float3) * n_gaussians, cudaStreamDefault));
  CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamDefault));

  float fx = params.fwd_input.K[0][0];
  float fy = params.fwd_input.K[1][1];
  float cx = params.fwd_input.K[2][0];
  float cy = params.fwd_input.K[2][1];

  // zero grad buffer.
  auto& means_grad = params.gaussians_grad->means();
  auto& scales_grad = params.gaussians_grad->scales();
  auto& rotations_grad = params.gaussians_grad->rotations();
  auto& opacities_grad = params.gaussians_grad->opacities();
  auto& sh_coeffs_0_grad = params.gaussians_grad->sh_coefficient_0();
  auto& sh_coeffs_rest_grad = params.gaussians_grad->sh_coefficients_rest();
  m_impl->w2c_grad.memset(0);

  float* densification_info = nullptr;
  if (params.densification_info) {
    densification_info = params.densification_info->data();
  }

  fast_gs::rasterization::backward(
    /* grad_image */ params.grad_output.image.data,
    /* grad_alpha */ params.grad_output.alpha.data,
    /* image */ params.fwd_output.image.data,
    /* alpha */ params.fwd_output.alpha.data,
    /* means */ reinterpret_cast<const float3*>(thrust::raw_pointer_cast(m_gaussians->means().data())),
    /* scales */ reinterpret_cast<const float3*>(thrust::raw_pointer_cast(m_gaussians->scales().data())),
    /* rotations */ reinterpret_cast<const float4*>(thrust::raw_pointer_cast(m_gaussians->rotations().data())),
    /* sh_coeffs_rest */ reinterpret_cast<const float3*>(thrust::raw_pointer_cast(m_gaussians->sh_coefficients_rest().data())),
    /* w2c */ m_impl->w2c.data(),
    /* cam_position */ m_impl->cam_position.data(),
    /* per_primitive_buffers_blob */ thrust::raw_pointer_cast(m_impl->temp_buffers["per_primitive_buffers"].data()),
    /* per_tile_buffers_blob */ thrust::raw_pointer_cast(m_impl->temp_buffers["per_tile_buffers"].data()),
    /* per_instance_buffers_blob */ thrust::raw_pointer_cast(m_impl->temp_buffers["per_instance_buffers"].data()),
    /* per_bucket_buffers_blob */ thrust::raw_pointer_cast(m_impl->temp_buffers["per_bucket_buffers"].data()),
    /* grad_means */ reinterpret_cast<float3*>(thrust::raw_pointer_cast(means_grad.data())),
    /* grad_scales */ reinterpret_cast<float3*>(thrust::raw_pointer_cast(scales_grad.data())),
    /* grad_rotations */ reinterpret_cast<float4*>(thrust::raw_pointer_cast(rotations_grad.data())),
    /* grad_opacities */ thrust::raw_pointer_cast(opacities_grad.data()),
    /* grad_sh_coeffs */ reinterpret_cast<float3*>(thrust::raw_pointer_cast(sh_coeffs_0_grad.data())),
    /* grad_sh_coeffs_rest */ reinterpret_cast<float3*>(thrust::raw_pointer_cast(sh_coeffs_rest_grad.data())),
    /* grad_mean2d_helper */  reinterpret_cast<float2*>(grad_mean2d_helper),
    /* grad_conic_helper */ reinterpret_cast<float*>(grad_conic_helper),
    /* grad_w2c */ m_impl->w2c_grad.data(),
    /* densification_info */ densification_info,
    /* n_primitives */ n_gaussians,
    /* n_visible_primitives */ m_impl->n_visible_primitives,
    /* n_instances */ m_impl->n_instances,
    /* n_buckets */ m_impl->n_buckets,
    /* primitive_primitive_indices_selector */ m_impl->primitive_primitive_indices_selector,
    /* instance_primitive_indices_selector */ m_impl->instance_primitive_indices_selector,
    /* active_sh_bases */ m_gaussians->get_sh_degree(),
    /* total_bases_sh_rest */ kMaxSphericalHarmonicsCoefficients - 1,
    /* width */ params.fwd_input.width,
    /* height */ params.fwd_input.height,
    /* fx */ fx,
    /* fy */ fy,
    /* cx */ cx,
    /* cy */ cy
  );
}

FastGSRasterizer::~FastGSRasterizer() {}

void FastGSRasterizer::set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians) {
  m_gaussians = gaussians;
  const auto n_gaussians = m_gaussians->size();
  m_impl->num_gaussians = n_gaussians;
}

} // namespace tinygs