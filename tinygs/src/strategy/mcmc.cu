#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform_reduce.h>
#include <thrust/uninitialized_copy.h>

#include "rasterizer/3dgs_accel/auxiliary.h"
#include "tinygs/cuda/common_device.cuh"
#include "tinygs/random/multinomial.hpp"
#include "tinygs/strategy/mcmc.hpp"
#include "utils/scope_timer.hpp"

namespace tinygs {
  
constexpr int max_binom_size = 51;
__constant__ float binom[max_binom_size * max_binom_size];


void init_binom() {
  static bool initialized = false;
  if (initialized) return;
  initialized = true;

  float h_binom[max_binom_size * max_binom_size];
  for (int n = 0; n < max_binom_size; ++n) {
    for (int k = 0; k <= n; ++k) {
      // Compute binomial coefficient C(n,k)
      float binom = 1.0f;
      for (int i = 0; i < k; ++i) {
        binom *= static_cast<float>(n - i) / static_cast<float>(i + 1);
      }
      h_binom[n * max_binom_size + k] = binom;
      h_binom[k * max_binom_size + n] = binom;
    }
  }
  CUDA_CHECK_THROW(cudaMemcpyToSymbol(binom, h_binom, sizeof(h_binom)));
}

// Custom CUDA kernel for `index_add_` (scatter-add) operation
// This function needs to be defined globally or in a namespace, not inside a class method.
__global__ static void index_add_kernel(int* data, const int* indices, int value_to_add, int num_indices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_indices) {
        atomicAdd(&data[indices[i]], value_to_add);
    }
}

// Refer: https://github.com/MrNeRF/LichtFeld-Studio/blob/60b4f2abf080e35afb858373a9fb152a3fdf1d3b/gsplat/RelocationCUDA.cu#L11
// Equation (9) in "3D Gaussian Splatting as Markov Chain Monte Carlo"
__global__ static void relocation_kernel(
    int N,
    float* __restrict__ opacities,
    vec3* __restrict__ scales,
    const int* __restrict__ ratios,
    float* __restrict__ new_opacities,
    vec3* __restrict__ new_scales,
    float opacity_threshold) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N)
        return;

    int n_idx = ratios[idx];
    float denom_sum = 0.0f;

    // compute new opacity
    float nop = 1.0f - ::powf(1.0f - opacities[idx], 1.0f / n_idx);
    nop = clamp(nop, opacity_threshold, 1.0f - 1e-8f);
    new_opacities[idx] = nop;

    // compute new scale
    for (int i = 1; i <= n_idx; ++i) {
        for (int k = 0; k <= (i - 1); ++k) {
            float bin_coeff = binom[(i - 1) * max_binom_size + k];
            float term = (::pow(-1.0f, k) / sqrt(static_cast<float>(k + 1))) *
                          ::pow(nop, k + 1);
            denom_sum += (bin_coeff * term);
        }
    }
    float coeff = (opacities[idx] / denom_sum);
    new_scales[idx] = coeff * scales[idx]; // TODO: This could be much larger than original. we should clamp
    // printf("Coeff: %f, Scale: %f %f %f, Original: %f %f %f", coeff,
    //     new_scales[idx].x, new_scales[idx].y, new_scales[idx].z,
    //     scales[idx].x, scales[idx].y, scales[idx].z);
}

__global__ void add_noise_kernel(
    int N,
    const float* __restrict__ raw_opacities,
    const float* __restrict__ raw_scales,
    const float* __restrict__ raw_quats,
    const float* __restrict__ noise,
    float* __restrict__ means,
    float current_lr) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N)
        return;

    int idx_3d = 3 * idx;
    
    const vec3 raw_scale = vec3(
      raw_scales[idx_3d + 0],
      raw_scales[idx_3d + 1],
      raw_scales[idx_3d + 2]
    );
    mat3x3 S2 = mat3x3(
      __expf(2.f * raw_scale[0]), 0.f, 0.f,
      0.f, __expf(2.f * raw_scale[1]), 0.f,
      0.f, 0.f, __expf(2.f * raw_scale[2])
    );

    quat raw_quat = normalize(quat( //
        raw_quats[4 * idx + 0],     //
        raw_quats[4 * idx + 1],     //
        raw_quats[4 * idx + 2],     //
        raw_quats[4 * idx + 3]      //
        ));
    mat3x3 R = quat_to_mat3(raw_quat);

    mat3x3 covariance = R * S2 * transpose(R);

    vec3 transformed_noise = covariance * vec3(
      noise[idx_3d + 0],
      noise[idx_3d + 1],
      noise[idx_3d + 2]
    );

    float opacity = logistic(-raw_opacities[idx]); // convert to [0, 1]
    float op_sigmoid = __frcp_rn(1.f + __expf(100.f * opacity - 0.5f));
    // float op_sigmoid = 1.0f / (1 + expf(-100.0f * ((1 - opacity) - 0.995f)));
    float noise_factor = current_lr * op_sigmoid;

    means[idx_3d] += noise_factor * transformed_noise.x;
    means[idx_3d + 1] += noise_factor * transformed_noise.y;
    means[idx_3d + 2] += noise_factor * transformed_noise.z;
}

MCMCStrategy::MCMCStrategy(
    std::shared_ptr<GPUGaussian3d> gaussians,
    std::shared_ptr<GPUGaussian3d> gaussians_grad,
    std::shared_ptr<OptimizerBase> optimizer
) : StrategyBase(gaussians, gaussians_grad, optimizer) {
  init_binom();
}


void MCMCStrategy::step_impl(const RasterizeContext& ctx) {
  ctx.densification_info.reset();
  const size_t step = this_step();
  add_noise(ctx);
  if (step % m_params.refine_every == 0 && step >= m_params.start_refine && step <= m_params.end_refine) {
    relocate(ctx);
    add_new_gs(ctx);
  }
}

void MCMCStrategy::reset() {
  // TODO: reset internal states
}

static thrust::default_random_engine rng;
void MCMCStrategy::set_noise_lr(float noise_lr) { m_noise_lr = noise_lr; }

void MCMCStrategy::add_noise(const RasterizeContext& ctx) {
  // TODO: this is simpler than expected.
  TINYGS_TIMER("MCMCStrategy::add_noise");
  m_noise_lr *= 1 - 1e-5;
  size_t num_gaussians = m_gaussians->size();
  if (num_gaussians == 0) return;
  thrust::host_vector<float> h_noise(3 * num_gaussians);
  thrust::uniform_real_distribution<float> dist(0, 1);
  thrust::generate(h_noise.begin(), h_noise.end(), [&]() { return m_noise_lr * dist(rng); });
  thrust::device_vector<float> noise = h_noise;
  add_noise_kernel<<<(num_gaussians + 255) / 256, 256>>>(
    num_gaussians,
    thrust::raw_pointer_cast(m_gaussians->opacities().data()),
    reinterpret_cast<const float*>(thrust::raw_pointer_cast(m_gaussians->scales().data())),
    reinterpret_cast<const float*>(thrust::raw_pointer_cast(m_gaussians->rotations().data())),
    thrust::raw_pointer_cast(noise.data()),
    reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians->means().data())),
    m_noise_lr
  );
}

void MCMCStrategy::add_new_gs(const RasterizeContext& ctx) {
  TINYGS_TIMER("MCMCStrategy::add_new_gs");
  // Expand exponentially.
  int num_gaussians = m_gaussians->size();
  int target_size = std::min(static_cast<int>(round(1.05 * num_gaussians)), m_params.max_num_gaussians);
  int num_to_add = target_size - num_gaussians;
  if (num_to_add <= 0) {
    assert(num_to_add == 0);
    return;
  }
  thrust::device_vector<float> opacities(num_gaussians);
  thrust::transform(
    thrust::device,
    m_gaussians->opacities().begin(),
    m_gaussians->opacities().end(),
    opacities.begin(),
    [] __device__ (float opacity) { return logistic(opacity); }
  ); // actual opacity = sigmoid(opacity)


  // Sample from alive Gaussians based on opacity
  const auto& probs = opacities;
  auto sampled_idxs_local = multinomial_cuda_cpu(
    thrust::raw_pointer_cast(probs.data()),
    num_gaussians,
    num_to_add,
    time(nullptr) // TODO: replace with real seed.
  );
  thrust::device_vector<int> sampled_idxs(num_to_add);
  thrust::copy(
    thrust::device,
    sampled_idxs_local.data(),
    sampled_idxs_local.data() + num_to_add,
    sampled_idxs.begin()
  );

  // Get parameters for sampled Gaussians
  thrust::device_vector<float> sampled_opacities(num_to_add);
  thrust::device_vector<vec3> sampled_scales(num_to_add);
  thrust::transform( // sampled_opacities = opacities.index_select(0, sampled_idxs);
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_to_add),
    sampled_opacities.begin(),
    [
      opacities = thrust::raw_pointer_cast(opacities.data()),
      sampled_idxs = thrust::raw_pointer_cast(sampled_idxs.data())
    ] __device__ (int idx) { return opacities[sampled_idxs[idx]]; }
  );
  thrust::transform( // sampled_scales = get_scale().index_select(0, sampled_idxs);
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_to_add),
    sampled_scales.begin(),
    [
      scales = thrust::raw_pointer_cast(m_gaussians->scales().data()),
      sampled_idxs = thrust::raw_pointer_cast(sampled_idxs.data())
    ] __device__ (int idx) { return exp(scales[sampled_idxs[idx]]); }
  );

  thrust::device_vector<int> new_indices(num_to_add);
  thrust::copy(
    thrust::make_counting_iterator<int>(num_gaussians),
    thrust::make_counting_iterator<int>(num_gaussians + num_to_add),
    new_indices.begin());

  on_duplicate(
        /* src_indices */ thrust::raw_pointer_cast(sampled_idxs.data()),
        /* dst_indices */ thrust::raw_pointer_cast(new_indices.data()),
        /* num_indices */ num_to_add
    );

  // now gaussians should have more space for us to store the duplications
  if (m_gaussians->size() != target_size) {
    throw std::runtime_error("MCMCStrategy::add_new_gs failed to expand the number of gaussians");
  }

  // copy the parameters
  thrust::for_each(
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_to_add),
    [
      sampled_opacities = sampled_opacities.data(),
      sampled_scales = thrust::raw_pointer_cast(sampled_scales.data()), // exp space.
      sampled_idxs = sampled_idxs.data(), // copy from
      new_indices = new_indices.data(),    // copy to
      means = m_gaussians->means().data(),
      opacities = m_gaussians->opacities().data(),
      scales = thrust::raw_pointer_cast(m_gaussians->scales().data()), // raw.
      rotations = m_gaussians->rotations().data(),
      sh_coefficient_0 = m_gaussians->sh_coefficient_0().data(),
      sh_coefficients_rest = m_gaussians->sh_coefficients_rest().data(),
      num_gaussians
    ] __device__ (int idx) {
      int src = sampled_idxs[idx]; // Get the source index.
      int dst = new_indices[idx];  // Get the destination index.
      assert(dst >= num_gaussians);
      means[dst] = means[src];
      opacities[src] = opacities[dst] = logit(sampled_opacities[idx]);
      scales[src] = scales[dst] = log(sampled_scales[idx]);
      rotations[dst] = rotations[src];
      sh_coefficient_0[dst] = sh_coefficient_0[src];
      auto sh_coef_src = sh_coefficients_rest + src * (kMaxSphericalHarmonicsCoefficients - 1);
      auto sh_coef_dst = sh_coefficients_rest + dst * (kMaxSphericalHarmonicsCoefficients - 1);
      for (int i = 0; i < kMaxSphericalHarmonicsCoefficients- 1; i++) {
        sh_coef_dst[i] = sh_coef_src[i];
      }
    }
  );

  log_info("Added new {} gaussians ({} total)", num_to_add, target_size);
}

MCMCStrategy::~MCMCStrategy() = default;

void MCMCStrategy::relocate(const RasterizeContext& ctx) {
  TINYGS_TIMER("MCMCStrategy::relocate");
  size_t num_gaussians = m_gaussians->size();
  thrust::device_vector<float> opacities(num_gaussians);
  thrust::transform(
    thrust::device,
    m_gaussians->opacities().begin(),
    m_gaussians->opacities().end(),
    opacities.begin(),
    [] __device__ (float opacity) { return logistic(opacity); }
  ); // actual opacity = sigmoid(opacity)

  auto rotations = m_gaussians->rotations();
  thrust::device_vector<int> is_alive(num_gaussians);
  thrust::transform(
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_gaussians),
    is_alive.begin(),
    [
      opacities = thrust::raw_pointer_cast(opacities.data()),
      scale = thrust::raw_pointer_cast(m_gaussians->scales().data()),
      scene_scale = m_gaussians->scene_scale(),
      pruning_scale_threshold = m_params.pruning_scale_threshold,
      prune_large = this_step() > m_params.reset_every,
      min_opacity = m_params.pruning_opacity_threshold
    ] __device__ (int idx) {
      bool not_large_ws = max(exp(scale[idx])) < pruning_scale_threshold * scene_scale;
      bool not_transparent = opacities[idx] > min_opacity;

      if (not_transparent && (not_large_ws || !prune_large)) {
        return 1;
      }
      return 0;
    }
  );

  const int num_kept = thrust::reduce(is_alive.begin(), is_alive.end());
  const int num_dead = num_gaussians - num_kept;
  if (num_dead <= 0) {
    log_debug("No gaussians to relocate");
    return;
  }
  log_debug("Relocating {} gaussians", num_dead);

  thrust::device_vector<int> alive_indices(num_kept);
  thrust::device_vector<int> dead_indices(num_dead);
  thrust::copy_if(
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_gaussians),
    is_alive.begin(),
    alive_indices.begin(),
    [] __device__ (int idx) { return idx == 1; }
  );

  thrust::copy_if(
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_gaussians),
    is_alive.begin(),
    dead_indices.begin(),
    [] __device__ (int idx) { return idx == 0; }
  );
  
  // Sample from alive Gaussians based on opacity
  thrust::device_vector<float> probs(num_kept);
  thrust::transform(  // probs = opacities.index_select(0, alive_indices);
    thrust::device,
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_kept),
    probs.begin(),
    [
      opacities = thrust::raw_pointer_cast(opacities.data()),
      alive_indices = thrust::raw_pointer_cast(alive_indices.data())
    ] __device__ (int idx) { return opacities[alive_indices[idx]]; }
  );
  auto sampled_idxs_local = multinomial_cuda_with_replacement(
    thrust::raw_pointer_cast(probs.data()),
    num_kept,
    num_dead,
    0 // TODO: replace with real seed.
  );
  thrust::device_vector<int> sampled_idxs(num_dead);
  thrust::transform(  // sampled_idxs = alive_indices.index_select(0, sampled_idxs_local);
    thrust::device,
    sampled_idxs_local.data(),
    sampled_idxs_local.data() + num_dead,
    sampled_idxs.begin(),
    [
      alive_indices = thrust::raw_pointer_cast(alive_indices.data())
    ] __device__ (int idx) { return alive_indices[idx]; }
  );

  // Get parameters for sampled Gaussians
  thrust::device_vector<float> sampled_opacities(num_dead);
  thrust::device_vector<vec3> sampled_scales(num_dead);
  thrust::transform( // sampled_opacities = opacities.index_select(0, sampled_idxs);
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_dead),
    sampled_opacities.begin(),
    [
      opacities = thrust::raw_pointer_cast(opacities.data()),
      sampled_idxs = thrust::raw_pointer_cast(sampled_idxs.data())
    ] __device__ (int idx) { return opacities[sampled_idxs[idx]]; }
  );
  thrust::transform( // sampled_scales = get_scale().index_select(0, sampled_idxs);
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_dead),
    sampled_scales.begin(),
    [
      scales = thrust::raw_pointer_cast(m_gaussians->scales().data()),
      sampled_idxs = thrust::raw_pointer_cast(sampled_idxs.data())
    ] __device__ (int idx) { return exp(scales[sampled_idxs[idx]]); }
  );

  // Count occurrences of each sampled index
  // Equivalent to:
  // auto ratios = torch::ones_like(opacities, torch::kInt32);
  // ratios.index_add_(0, sampled_idxs, torch::ones_like(sampled_idxs, torch::kInt32));
  thrust::device_vector<int> sampled_idxs_count(num_kept, 1);
  index_add_kernel<<<(num_dead + 255) / 256, 256>>>(
    thrust::raw_pointer_cast(sampled_idxs_count.data()),
    thrust::raw_pointer_cast(sampled_idxs.data()),
    1,
    num_dead
  );
  //* ratios = ratios.index_select(0, sampled_idxs).contiguous();
  thrust::device_vector<int> ratios(num_dead);
  thrust::transform(
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_dead),
    ratios.begin(),
    [
      cnt = thrust::raw_pointer_cast(sampled_idxs_count.data()),
      sampled_idxs = thrust::raw_pointer_cast(sampled_idxs.data())
    ] __device__ (int idx) { 
      return ::min(cnt[sampled_idxs[idx]], max_binom_size);
    }
  );

  // Call the CUDA relocation function from gsplat
  thrust::device_vector<float> new_opacities(num_dead);
  thrust::device_vector<vec3> new_scales(num_dead);
  relocation_kernel<<<(num_dead + 255) / 256, 256>>>(
    num_dead,
    thrust::raw_pointer_cast(sampled_opacities.data()),
    thrust::raw_pointer_cast(sampled_scales.data()), // scales in exponential space.
    thrust::raw_pointer_cast(ratios.data()),
    thrust::raw_pointer_cast(new_opacities.data()),
    thrust::raw_pointer_cast(new_scales.data()),
    m_params.pruning_opacity_threshold
  );

  // Update all the dead gaussians.
  thrust::for_each(
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_dead),
    [
      new_opacities = thrust::raw_pointer_cast(new_opacities.data()),
      new_scales = thrust::raw_pointer_cast(new_scales.data()), // exp space.
      sampled_idxs = sampled_idxs.data(), // copy from
      dead_idxs = dead_indices.data(),    // copy to
      means = m_gaussians->means().data(),
      opacities = m_gaussians->opacities().data(),
      scales = m_gaussians->scales().data(), // raw.
      rotations = m_gaussians->rotations().data(),
      sh_coefficient_0 = m_gaussians->sh_coefficient_0().data(),
      sh_coefficients_rest = m_gaussians->sh_coefficients_rest().data()
    ] __device__ (int idx) {
      int src = sampled_idxs[idx]; // Get the source index.
      int dst = dead_idxs[idx];    // Get the destination index.
      means[dst] = means[src];
      opacities[src] = opacities[dst] = logit(new_opacities[idx]);
      scales[src] = scales[dst] = log(new_scales[idx]);
      rotations[dst] = rotations[src];
      sh_coefficient_0[dst] = sh_coefficient_0[src];
      auto sh_coef_src = sh_coefficients_rest + src * (kMaxSphericalHarmonicsCoefficients - 1);
      auto sh_coef_dst = sh_coefficients_rest + dst * (kMaxSphericalHarmonicsCoefficients - 1);
      for (int i = 0; i < kMaxSphericalHarmonicsCoefficients- 1; i++) {
        sh_coef_dst[i] = sh_coef_src[i];
      }
    }
  );

  StrategyBase::on_reset(
    /* indices */ thrust::raw_pointer_cast(dead_indices.data()),
    /* num_indices */ num_dead);
}

} // namespace tinygs
