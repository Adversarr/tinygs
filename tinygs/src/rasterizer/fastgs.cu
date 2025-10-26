// https://github.com/MrNeRF/gaussian-splatting-cuda

#include <cuda_runtime.h>
#include <memory>
#include <nvtx3/nvtx3.hpp>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "tinygs/common.hpp"
#include "tinygs/cuda/vec.hpp"
#include "tinygs/rasterizer/fastgs.hpp"

#include "fastgs_ours/backward.h"
#include "fastgs_ours/forward.h"
#include "fastgs_ours_fp16/backward.h"
#include "fastgs_ours_fp16/forward.h"
namespace tinygs {

struct PoseBlock {
  alignas(64) mat4x4 w2c;
  alignas(64) float3 cam_position;
  alignas(64) mat4x4 w2c_grad;
};

struct FastGSRasterizer::Impl {
  PoseBlock* host_block;
  GPUMemory<PoseBlock> device_block;  // Contains w2c, w2c_grad, and cam_position
  size_t num_gaussians;
  cudaStream_t helper_stream;

  GPUBuffer<float2> grad_mean2d_helper;
  GPUBuffer<float3> grad_conic_helper;
  GPUBuffer<float4> grad_w2c_per_gs;
  GPUBuffer<float3> grad_color;
  GPUBuffer<float2> absgrad_mean2d_helper;

  char* zero_copy = nullptr;

  thrust::device_vector<float> m_alpha_buffer;
  thrust::device_vector<float16_t> m_alpha_buffer_fp16;

  // CUDA events for synchronization
  cudaEvent_t memset_per_tile_done;
  cudaEvent_t copy_n_instances_done;
  cudaEvent_t preprocess_done;

  // 3. helper
  int n_visible_primitives, n_instances, n_buckets;
  int primitive_primitive_indices_selector, instance_primitive_indices_selector;
  // std::shared_ptr<GPUMemoryArena> arena;
  // std::map<std::string, std::unique_ptr<GPUBuffer<char>>> temp_buffers;
  std::map<std::string, thrust::device_vector<char>> temp_buffers;

  Impl() : num_gaussians(0) {
    device_block = GPUMemory<PoseBlock>(1, /*managed=*/ false);
    cudaHostAlloc(&host_block, sizeof(PoseBlock), cudaHostAllocMapped);
    CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&helper_stream, cudaStreamNonBlocking));
    CUDA_CHECK_THROW(cudaHostAlloc(&zero_copy, 1024, cudaHostAllocMapped)); // more than sufficient.
    CUDA_CHECK_THROW(cudaEventCreateWithFlags(&memset_per_tile_done, cudaEventDisableTiming));
    CUDA_CHECK_THROW(cudaEventCreateWithFlags(&copy_n_instances_done, cudaEventDisableTiming));
    CUDA_CHECK_THROW(cudaEventCreateWithFlags(&preprocess_done, cudaEventDisableTiming));
  }


  ~Impl() {
    cudaFreeHost(host_block);
    CUDA_CHECK_PRINT(cudaEventDestroy(memset_per_tile_done));
    CUDA_CHECK_PRINT(cudaEventDestroy(copy_n_instances_done));
    CUDA_CHECK_PRINT(cudaEventDestroy(preprocess_done));
    CUDA_CHECK_PRINT(cudaFreeHost(zero_copy));
    CUDA_CHECK_PRINT(cudaStreamDestroy(helper_stream));
  }

  char* alloc(const std::string &name, size_t size) { 
    NVTX3_FUNC_RANGE();
    auto& buffer = temp_buffers[name];
    if (size > buffer.size()) {
      buffer.resize(size);
    }
    return thrust::raw_pointer_cast(buffer.data());
  }
};

FastGSRasterizer::FastGSRasterizer() {
    m_impl = std::make_unique<Impl>();
}

void FastGSRasterizer::forward(const RasterizeContext& ctx) {
    NVTX3_FUNC_RANGE();
    if (!m_gaussians) {
        throw std::runtime_error("Gaussians not set");
    }

    const mat4x4 &w2c = ctx.fwd_input.w2c;
    mat4x4 c2w = inverse(w2c);

    // Copy w2c and cam_position to host memory
    m_impl->host_block->w2c = glm::transpose(w2c);
    m_impl->host_block->cam_position = {c2w[3][0], c2w[3][1], c2w[3][2]};

    // Copy host memory to device memory
    cudaMemcpyAsync(m_impl->device_block.data(), m_impl->host_block,
                    sizeof(PoseBlock), cudaMemcpyHostToDevice,
                    ctx.stream);

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

    const auto width = ctx.fwd_input.width;
    const auto height = ctx.fwd_input.height;
    const auto pw = ((width + kImageTileMask) >> kImageTileLog2) * kImageTile;
    const auto ph = ((height + kImageTileMask) >> kImageTileLog2) * kImageTile;

    const auto& means = m_gaussians->means();
    const auto& scales = m_gaussians->scales();
    const auto& rotations = m_gaussians->rotations();
    const auto& opacities = m_gaussians->opacities();
    const auto& sh_coeffs_0 = m_gaussians->sh_coefficient_0();
    const auto& sh_coeffs_rest = m_gaussians->sh_coefficients_rest();

    int activated_bases = (m_gaussians->get_sh_degree() + 1) * (m_gaussians->get_sh_degree() + 1);

    // Branch by output image precision
    if (ctx.fwd_output.image.data_type == DataType::Float16) {
      if (m_impl->m_alpha_buffer_fp16.size() < pw * ph) {
        m_impl->m_alpha_buffer_fp16.resize(pw * ph);
      }

      auto [n_visible_primitives, n_instances, n_buckets,
            primitive_primitive_indices_selector,
            instance_primitive_indices_selector] =
          tinygs::fast_gs_fp16::forward( //
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
              /* w2c */ reinterpret_cast<const float4*>(&m_impl->device_block.data()->w2c),
              /* cam_position */ &m_impl->device_block.data()->cam_position,
              /* densification_info */ ctx.densification_info ? ctx.densification_info->data() : nullptr,
              /* image */ static_cast<float16_t*>(ctx.fwd_output.image.data),
              /* alpha */ thrust::raw_pointer_cast(m_impl->m_alpha_buffer_fp16.data()),
              /* n_primitives */ m_gaussians->size(),
              /* active_sh_bases */ activated_bases,
              /* total_bases_sh_rest */ kMaxSphericalHarmonicsCoefficients - 1,
              /* width */ ctx.fwd_input.width,
              /* height */ ctx.fwd_input.height,
              /* fx */ fx,
              /* fy */ fy,
              /* cx */ cx,
              /* cy */ cy,
              /* near */ ctx.fwd_input.near,
              /* far */ ctx.fwd_input.far,
              /* major_stream */ ctx.stream,
              /* helper_stream */ m_impl->helper_stream,
              /* zero_copy */ m_impl->zero_copy,
              /* memset_per_tile_done */ m_impl->memset_per_tile_done,
              /* copy_n_instances_done */ m_impl->copy_n_instances_done,
              /* preprocess_done */ m_impl->preprocess_done);
      m_impl->n_visible_primitives = n_visible_primitives;
      m_impl->n_instances = n_instances;
      m_impl->n_buckets = n_buckets;
      m_impl->primitive_primitive_indices_selector = primitive_primitive_indices_selector;
      m_impl->instance_primitive_indices_selector = instance_primitive_indices_selector;
    } else {
      if (m_impl->m_alpha_buffer.size() < pw * ph) {
        m_impl->m_alpha_buffer.resize(pw * ph);
      }

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
              /* w2c */ reinterpret_cast<const float4*>(&m_impl->device_block.data()->w2c),
              /* cam_position */ &m_impl->device_block.data()->cam_position,
              /* densification_info */ ctx.densification_info ? ctx.densification_info->data() : nullptr,
              /* image */ static_cast<float*>(ctx.fwd_output.image.data),
              /* alpha */ thrust::raw_pointer_cast(m_impl->m_alpha_buffer.data()),
              /* n_primitives */ m_gaussians->size(),
              /* active_sh_bases */ activated_bases,
              /* total_bases_sh_rest */ kMaxSphericalHarmonicsCoefficients - 1,
              /* width */ ctx.fwd_input.width,
              /* height */ ctx.fwd_input.height,
              /* fx */ fx,
              /* fy */ fy,
              /* cx */ cx,
              /* cy */ cy,
              /* near */ ctx.fwd_input.near,
              /* far */ ctx.fwd_input.far,
              /* major_stream */ ctx.stream,
              /* helper_stream */ m_impl->helper_stream,
              /* zero_copy */ m_impl->zero_copy,
              /* memset_per_tile_done */ m_impl->memset_per_tile_done,
              /* copy_n_instances_done */ m_impl->copy_n_instances_done,
              /* preprocess_done */ m_impl->preprocess_done);
      m_impl->n_visible_primitives = n_visible_primitives;
      m_impl->n_instances = n_instances;
      m_impl->n_buckets = n_buckets;
      m_impl->primitive_primitive_indices_selector = primitive_primitive_indices_selector;
      m_impl->instance_primitive_indices_selector = instance_primitive_indices_selector;
    }
}

void FastGSRasterizer::backward(RasterizeContext &ctx) {
  NVTX3_FUNC_RANGE();
  if (!m_gaussians || !ctx.gaussians_grad) {
    throw std::runtime_error("Gaussians or gradient gaussians not set");
  }
  const auto n_gaussians = m_gaussians->size();

  float fx = ctx.fwd_input.K[0][0];
  float fy = ctx.fwd_input.K[1][1];
  float cx = ctx.fwd_input.K[2][0];
  float cy = ctx.fwd_input.K[2][1];

  // zero grad buffer.
  auto& means_grad = ctx.gaussians_grad->means();
  auto& scales_grad = ctx.gaussians_grad->scales();
  auto& rotations_grad = ctx.gaussians_grad->rotations();
  auto& opacities_grad = ctx.gaussians_grad->opacities();
  auto& sh_coeffs_0_grad = ctx.gaussians_grad->sh_coefficient_0();
  auto& sh_coeffs_rest_grad = ctx.gaussians_grad->sh_coefficients_rest();

  DensificationInfo* densification_info = nullptr;
  if (ctx.densification_info) {
    densification_info = ctx.densification_info->data();
  }
  int activated_bases = (m_gaussians->get_sh_degree() + 1) * (m_gaussians->get_sh_degree() + 1);

  float4* grad_w2c_per_gs = nullptr;
  if (m_params.enable_pose_opt) {
    if (m_impl->grad_w2c_per_gs.size() < 4 * n_gaussians) {
      m_impl->grad_w2c_per_gs = GPUBuffer<float4>(ctx.stream, 4 * n_gaussians);
    }
    m_impl->grad_w2c_per_gs.memset_async(ctx.stream, 0);
    grad_w2c_per_gs = m_impl->grad_w2c_per_gs.data();
  }

  if (ctx.fwd_output.image.data_type == DataType::Float16) {
    tinygs::fast_gs_fp16::backward(
      /* grad_image */ static_cast<const float16_t*>(ctx.grad_output.image.data),
      /* image */ static_cast<const float16_t*>(ctx.fwd_output.image.data),
      /* means */ reinterpret_cast<const float3*>(thrust::raw_pointer_cast(m_gaussians->means().data())),
      /* scales */ reinterpret_cast<const float3*>(thrust::raw_pointer_cast(m_gaussians->scales().data())),
      /* rotations */ reinterpret_cast<const float4*>(thrust::raw_pointer_cast(m_gaussians->rotations().data())),
      /* sh_coeffs_rest */ reinterpret_cast<const float3*>(thrust::raw_pointer_cast(m_gaussians->sh_coefficients_rest().data())),
      /* w2c */ reinterpret_cast<const float4*>(&m_impl->device_block.data()->w2c),
      /* cam_position */ &m_impl->device_block.data()->cam_position,
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
      /* grad_w2c */ reinterpret_cast<float4*>(&m_impl->device_block.data()->w2c_grad),
      /* grad_w2c_per_gs */ grad_w2c_per_gs,
      /* densification_info */ densification_info,
      /* n_primitives */ n_gaussians,
      /* n_visible_primitives */ m_impl->n_visible_primitives,
      /* n_instances */ m_impl->n_instances,
      /* n_buckets */ m_impl->n_buckets,
      /* primitive_primitive_indices_selector */ m_impl->primitive_primitive_indices_selector,
      /* instance_primitive_indices_selector */ m_impl->instance_primitive_indices_selector,
      /* active_sh_bases */ activated_bases,
      /* total_bases_sh_rest */ kMaxSphericalHarmonicsCoefficients - 1,
      /* width */ ctx.fwd_input.width,
      /* height */ ctx.fwd_input.height,
      /* fx */ fx,
      /* fy */ fy,
      /* cx */ cx,
      /* cy */ cy,
      /* stream */ ctx.stream
    );
  } else {
    if (m_impl->grad_mean2d_helper.size() < n_gaussians) {
      m_impl->grad_mean2d_helper = GPUBuffer<float2>(ctx.stream, n_gaussians);
    }
    if (m_impl->grad_conic_helper.size() < n_gaussians) {
      m_impl->grad_conic_helper = GPUBuffer<float3>(ctx.stream, n_gaussians);
    }
    if (m_impl->grad_color.size() < n_gaussians) {
      m_impl->grad_color = GPUBuffer<float3>(ctx.stream, n_gaussians);
    }
    if (m_impl->absgrad_mean2d_helper.size() < n_gaussians) {
      m_impl->absgrad_mean2d_helper = GPUBuffer<float2>(ctx.stream, n_gaussians);
    }
    m_impl->grad_mean2d_helper.memset_async(ctx.stream, 0);
    m_impl->grad_conic_helper.memset_async(ctx.stream, 0);
    m_impl->grad_color.memset_async(ctx.stream, 0);
    m_impl->absgrad_mean2d_helper.memset_async(ctx.stream, 0);
    fast_gs::rasterization::backward(
      /* grad_image */ static_cast<float*>(ctx.grad_output.image.data),
      /* image */ static_cast<float*>(ctx.fwd_output.image.data),
      /* means */ reinterpret_cast<const float3*>(thrust::raw_pointer_cast(m_gaussians->means().data())),
      /* scales */ reinterpret_cast<const float3*>(thrust::raw_pointer_cast(m_gaussians->scales().data())),
      /* rotations */ reinterpret_cast<const float4*>(thrust::raw_pointer_cast(m_gaussians->rotations().data())),
      /* sh_coeffs_rest */ reinterpret_cast<const float3*>(thrust::raw_pointer_cast(m_gaussians->sh_coefficients_rest().data())),
      /* w2c */ reinterpret_cast<const float4*>(&m_impl->device_block.data()->w2c),
      /* cam_position */ &m_impl->device_block.data()->cam_position,
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
      /* grad_mean2d_helper */  m_impl->grad_mean2d_helper.data(),
      /* grad_conic_helper */ reinterpret_cast<float*>(m_impl->grad_conic_helper.data()),
      /* grad_color_helper */ m_impl->grad_color.data(),
      /* grad_w2c */ reinterpret_cast<float4*>(&m_impl->device_block.data()->w2c_grad),
      /* grad_w2c_per_gs */ grad_w2c_per_gs,
      /* densification_info */ densification_info,
      /* absgrad_mean2d_helper */ m_impl->absgrad_mean2d_helper.data(),
      /* n_primitives */ n_gaussians,
      /* n_visible_primitives */ m_impl->n_visible_primitives,
      /* n_instances */ m_impl->n_instances,
      /* n_buckets */ m_impl->n_buckets,
      /* primitive_primitive_indices_selector */ m_impl->primitive_primitive_indices_selector,
      /* instance_primitive_indices_selector */ m_impl->instance_primitive_indices_selector,
      /* active_sh_bases */ activated_bases,
      /* total_bases_sh_rest */ kMaxSphericalHarmonicsCoefficients - 1,
      /* width */ ctx.fwd_input.width,
      /* height */ ctx.fwd_input.height,
      /* fx */ fx,
      /* fy */ fy,
      /* cx */ cx,
      /* cy */ cy,
      /* stream */ ctx.stream
    );
  }

  // set grad w2c
  CUDA_CHECK_THROW(cudaMemcpyAsync(
    m_impl->host_block,
    m_impl->device_block.data(),
    m_impl->device_block.get_bytes(),
    cudaMemcpyDeviceToHost,
    ctx.stream
  ));
  CUDA_CHECK_THROW(cudaStreamSynchronize(ctx.stream));

  ctx.grad_input.w2c = glm::transpose(m_impl->host_block->w2c_grad);
}

FastGSRasterizer::~FastGSRasterizer() {}

void FastGSRasterizer::set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians) {
  m_gaussians = gaussians;
  const auto n_gaussians = m_gaussians->size();
  m_impl->num_gaussians = n_gaussians;
}

json FastGSRasterizer::get_params() const {
  return json::object({{"type", "fastgs"}});
}

void FastGSRasterizer::set_params(const json& /*j*/) {}

} // namespace tinygs