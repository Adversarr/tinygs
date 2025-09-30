#include <spdlog/spdlog.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <vector>

#include "tinygs/cuda/gpu_memory.hpp"
#include "tinygs/cuda/reduce.hpp"
#include "tinygs/orchestrator.hpp"
#include "tinygs/utils/file.hpp"
#include "tinygs/utils/scope_timer.hpp"
#include <cub/device/device_radix_sort.cuh>
#include "tinygs/cuda/common_device.cuh"
namespace tinygs {

static thrust::device_vector<uint> reorder(const vec3* positions, uint n, cudaStream_t stream) {
  thrust::device_vector<uint> idx_in(n), idx_out(n);
  thrust::device_vector<uint> enc_in(n), enc_out(n);

  thrust::copy(
    thrust::cuda::par.on(stream),
    thrust::make_counting_iterator<uint>(0),
    thrust::make_counting_iterator<uint>(n),
    idx_in.begin()
  );

  vec3 min_pos = thrust::reduce(
    thrust::cuda::par.on(stream),
    positions, positions + n,
    vec3(FLT_MAX, FLT_MAX, FLT_MAX),
    [] __host__ __device__ (const vec3& a, const vec3& b) -> vec3 { return vec3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)); }
  );

  vec3 max_pos = thrust::reduce(
    thrust::cuda::par.on(stream),
    positions, positions + n,
    vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX),
    [] __host__ __device__ (const vec3& a, const vec3& b) -> vec3 { return vec3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }
  );

  // Precompute inverse deltas on host to avoid per-element device computation
  const float inv_dx = 1.0f / std::max(max_pos.x - min_pos.x, 1e-8f);
  const float inv_dy = 1.0f / std::max(max_pos.y - min_pos.y, 1e-8f);
  const float inv_dz = 1.0f / std::max(max_pos.z - min_pos.z, 1e-8f);

  thrust::transform(
    thrust::cuda::par.on(stream),
    positions, positions + n,
    enc_in.begin(), [min_pos, inv_dx, inv_dy, inv_dz] __device__ (const vec3& p) -> uint {
      // Normalize position to [0,1] within the bounding box
      const float nx = fminf(fmaxf((p.x - min_pos.x) * inv_dx, 0.0f), 1.0f);
      const float ny = fminf(fmaxf((p.y - min_pos.y) * inv_dy, 0.0f), 1.0f);
      const float nz = fminf(fmaxf((p.z - min_pos.z) * inv_dz, 0.0f), 1.0f);

      // Map to 10-bit integer grid per axis and compute Morton code
      const uint32_t xi = static_cast<uint32_t>(nx * 1023.0f);
      const uint32_t yi = static_cast<uint32_t>(ny * 1023.0f);
      const uint32_t zi = static_cast<uint32_t>(nz * 1023.0f);
      return morton3D(xi, yi, zi);
    }
  );

  // sort.
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  // Query temporary storage size
  cub::DeviceRadixSort::SortPairs(
    d_temp_storage, temp_storage_bytes,
    thrust::raw_pointer_cast(enc_in.data()),
    thrust::raw_pointer_cast(enc_out.data()),
    thrust::raw_pointer_cast(idx_in.data()),
    thrust::raw_pointer_cast(idx_out.data()),
    n, 0, 30, stream
  );

  // Allocate temporary storage and perform sort
  thrust::device_vector<uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());
  cub::DeviceRadixSort::SortPairs(
    d_temp_storage, temp_storage_bytes,
    thrust::raw_pointer_cast(enc_in.data()),
    thrust::raw_pointer_cast(enc_out.data()),
    thrust::raw_pointer_cast(idx_in.data()),
    thrust::raw_pointer_cast(idx_out.data()),
    n, 0, 30, stream
  );
  

  return idx_out;
}

// TrainerConfig serialization methods
json OrchestratorConfig::to_json() const {
  json j;
  j["max_steps"] = max_steps;
  j["log_interval"] = log_interval;
  j["checkpoint_interval"] = checkpoint_interval;
  j["sh_degree_interval"] = sh_degree_interval;
  j["max_sh_degree"] = max_sh_degree;
  j["enable_early_stopping"] = enable_early_stopping;
  j["early_stopping_threshold"] = early_stopping_threshold;
  j["early_stopping_patience"] = early_stopping_patience;
  j["near_plane"] = near_plane;
  j["far_plane"] = far_plane;
  j["test_steps"] = test_steps;
  j["grad_scaler"] = grad_scaler;
  j["out_dir"] = out_dir;
  j["export_rasterized"] = export_rasterized;
  j["enable_progressive_resolution"] = enable_progressive_resolution;
  j["resolution_milestones"] = resolution_milestones;
  j["resolution_scales"] = resolution_scales;
  j["scene_scale_recompute_interval"] = scene_scale_recompute_interval;
  return j;
}

void OrchestratorConfig::from_json(const json& j) {
  if (j.contains("max_steps")) max_steps = j["max_steps"].get<int>();
  if (j.contains("log_interval")) log_interval = j["log_interval"].get<int>();
  if (j.contains("checkpoint_interval")) checkpoint_interval = j["checkpoint_interval"].get<int>();
  if (j.contains("sh_degree_interval")) sh_degree_interval = j["sh_degree_interval"].get<int>();
  if (j.contains("max_sh_degree")) max_sh_degree = j["max_sh_degree"].get<int>();
  if (j.contains("enable_early_stopping")) enable_early_stopping = j["enable_early_stopping"].get<bool>();
  if (j.contains("early_stopping_threshold")) early_stopping_threshold = j["early_stopping_threshold"].get<float>();
  if (j.contains("early_stopping_patience")) early_stopping_patience = j["early_stopping_patience"].get<int>();
  if (j.contains("near_plane")) near_plane = j["near_plane"].get<float>();
  if (j.contains("far_plane")) far_plane = j["far_plane"].get<float>();
  if (j.contains("test_steps")) {
    try {
      const json::array_t steps = j.at("test_steps");
      std::vector<size_t> new_steps;
      for (const auto& step : steps) {
        new_steps.push_back(step.get<size_t>());
      }
      test_steps = new_steps;
    } catch (const json::exception& e) {
      throw std::invalid_argument("Expect test_steps to be array of integers.");
    }
  }
  if (j.contains("grad_scaler")) grad_scaler = j["grad_scaler"].get<float>();
  if (j.contains("out_dir")) out_dir = j["out_dir"].get<std::string>();
  if (j.contains("export_rasterized")) export_rasterized = j["export_rasterized"].get<bool>();
  if (j.contains("enable_progressive_resolution")) enable_progressive_resolution = j["enable_progressive_resolution"].get<bool>();
  if (j.contains("resolution_milestones")) {
    try {
      const json::array_t milestones = j.at("resolution_milestones");
      std::vector<size_t> new_milestones;
      for (const auto& milestone : milestones) {
        new_milestones.push_back(milestone.get<size_t>());
      }
      resolution_milestones = new_milestones;
    } catch (const json::exception& e) {
      throw std::invalid_argument("Expect resolution_milestones to be array of integers.");
    }
  }
  if (j.contains("resolution_scales")) {
    try {
      const json::array_t scales = j.at("resolution_scales");
      std::vector<float> new_scales;
      for (const auto& scale : scales) {
        new_scales.push_back(scale.get<float>());
      }
      resolution_scales = new_scales;
    } catch (const json::exception& e) {
      throw std::invalid_argument("Expect resolution_scales to be array of floats.");
    }
  }
  if (j.contains("scene_scale_recompute_interval")) scene_scale_recompute_interval = j["scene_scale_recompute_interval"].get<size_t>();
}

void mean(const vec3* data, size_t size, vec3& out) {
  out = thrust::transform_reduce(
    thrust::device,
    data,
    data + size,
    [inv_s = 1.0f / static_cast<float>(size)] __device__ (const vec3& p) -> vec3 { return p * inv_s; },
    vec3{0.0f, 0.0f, 0.0f},
    thrust::plus<vec3>()
  );
}


void Orchestrator::recompute_scene_scale() {
  auto ds = m_dataloader->get_dataset();
  auto pc = m_gaussians->means();

  // avg_pc_mean
  vec3 avg_pc_mean;
  mean(thrust::raw_pointer_cast(pc.data()), pc.size(), avg_pc_mean);

  float scale = 0;
  for (auto c: ds->get_camera_loader().get_camera_extrinsics()) {
    auto c2w = c.get_c2w();
    vec3 cam_pos = c2w[3];
    scale = std::max(scale, glm::distance(cam_pos, avg_pc_mean));
  }
  m_gaussians->set_scene_scale(scale);
  log_info("Recompute scene scale: {}", scale);
}

Orchestrator::Orchestrator(const OrchestratorConfig& config) : m_config(config) {
  m_state.start_time = std::chrono::steady_clock::now();
  m_state.last_log_time = m_state.start_time;
}

void Orchestrator::set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians, 
                           std::shared_ptr<GPUGaussian3d> gradients) {
  m_gaussians = gaussians;
  m_gradients = gradients;
}

void Orchestrator::set_rasterizer(std::shared_ptr<RasterizerBase> rasterizer) {
  m_rasterizer = rasterizer;
  if (m_gaussians) {
    m_rasterizer->set_gaussians(m_gaussians);
  }
}

void Orchestrator::set_dataloader(std::shared_ptr<DataLoaderBase> dataloader) {
  m_dataloader = dataloader;
}

void Orchestrator::set_optimizer(std::shared_ptr<OptimizerBase> optimizer) {
  m_optimizer = optimizer;
}

void Orchestrator::set_strategy(std::shared_ptr<StrategyBase> strategy) {
  m_strategy = strategy;
}

void Orchestrator::add_loss(std::shared_ptr<LossBase> loss, float weight) {
  m_losses.push_back({loss, weight});
}

void Orchestrator::add_metric(std::shared_ptr<MetricBase> metric, const std::string& name) {
  m_metrics.push_back({metric, name});
}

void Orchestrator::set_pre_step_callback(PreStepCallback callback) {
  m_pre_step_callback = callback;
}

void Orchestrator::set_post_step_callback(PostStepCallback callback) {
  m_post_step_callback = callback;
}

void Orchestrator::set_checkpoint_callback(CheckpointCallback callback) {
  m_checkpoint_callback = callback;
}

TrainingState Orchestrator::train() {
  validate_setup();
  initialize();
  recompute_scene_scale();

  m_state.start_time = std::chrono::steady_clock::now();
  m_state.last_log_time = m_state.start_time;
  m_state.should_stop = false;

  while (!m_state.should_stop && m_state.current_step <= m_config.max_steps) {
    // Test step
    if (std::find(m_config.test_steps.begin(), m_config.test_steps.end(),
                  m_state.current_step) != m_config.test_steps.end()) {
      log_info("Test step at step {}", m_state.current_step);
      test_step();
    }

    train_step();

    // Check for early stopping
    if (m_config.enable_early_stopping && should_early_stop()) {
      log_info("Early stopping triggered at step {}", m_state.current_step);
      break;
    }
  }
  
  return m_state;
}

void Orchestrator::train_step() {
  NVTX3_FUNC_RANGE();
  // Check and update resolution for progressive training
  if (m_config.enable_progressive_resolution) {
    update_resolution(m_state.current_step);
  }

  // Pre-step callback
  if (m_pre_step_callback) {
    m_pre_step_callback(m_state);
  }

  // TODO: async, not in the major/default stream.
  // Clear gradients and buffers
  m_gradients->memset(0);
  m_loss_buffer->memset(0);
  m_image_grad_buffer->memset(0);

  // Get next batch of data
  auto data = m_dataloader->next();

  // Update rasterization context with current data
  m_rasterize_ctx.fwd_input = data.input;

  // Forward pass
  m_rasterizer->forward(m_rasterize_ctx);

  // Evaluate losses and accumulate gradients
  evaluate_losses(data);

  // Setup gradient output for backward pass
  m_rasterize_ctx.grad_output.image = m_loss_ctx.grad;

  // TODO: remove the alpha support, we only consider RGB image.
  // Create alpha gradient image
  ImageShape shape = m_rasterize_ctx.fwd_output.image.shape;
  m_rasterize_ctx.grad_output.alpha =
      Image({shape.width, shape.height, 1}, ImageDataType::Float32,
             m_loss_buffer->data() + shape.padded_size());

  // Backward pass
  m_rasterizer->backward(m_rasterize_ctx);

  // Step the learning rate scheduler if available
  if (m_lr_scheduler) {
    m_lr_scheduler->step();
  }

  // Learning rate is now managed by the scheduler-optimizer system

  // Optimizer step (learning rate already set by scheduler)
  //? the gradient scaler, since we are not supporting AMP, 1.0f is the default value.
  const float inv_grad_scale = 1.0f / m_config.grad_scaler;
  m_optimizer->step(inv_grad_scale, m_major_stream);

  // Strategy step (densification)
  if (m_strategy) {
    m_strategy->step(m_rasterize_ctx);
  }

  if (m_state.current_step > 0) {
    if (m_state.current_step % m_config.scene_scale_recompute_interval == 0) {
      recompute_scene_scale();
    }
    if (m_state.current_step % m_config.reorder_gaussians_interval == 0) {
      reorder_gaussians();
    }
  }

  // Update spherical harmonics degree
  update_sh_degree();

  // Post-step callback (for logging, visualization, etc.)
  if (m_post_step_callback) {
    m_post_step_callback(m_state);
  }

  // Checkpoint callback
  if (m_checkpoint_callback && m_state.current_step % m_config.checkpoint_interval == 0) {
    m_checkpoint_callback(m_state);
  }

  m_state.current_step++;
}

void Orchestrator::test_step() {
  NVTX3_FUNC_RANGE();

  // Store current resolution for restoration later
  ImageShape current_shape{m_rasterize_ctx.fwd_input.width, m_rasterize_ctx.fwd_input.height, 3};
  
  // Force full resolution for testing if progressive resolution is enabled
  if (m_config.enable_progressive_resolution) {
    ImageShape full_shape = m_dataloader->get_dataset()->image_shape();
    if (full_shape.width != current_shape.width || full_shape.height != current_shape.height) {
      log_info("Switching to full resolution {}x{} for testing", full_shape.width, full_shape.height);
      set_render_resolution({full_shape.width, full_shape.height, 1});
    }
  }

  m_dataloader->reset(); // reset the permutation.
  std::string out_dir = m_config.out_dir + "/" + std::to_string(m_state.current_step);

  ensure(out_dir);
  const auto total_samples = m_dataloader->get_dataset()->size();
  std::map<std::string, std::vector<float>> metrics;
  std::vector<uuid_t> timestamps;

  for (size_t idx = 0; idx < total_samples; ++ idx){
    auto data = m_dataloader->next();
    timestamps.push_back(data.input.timestamp);
    // Rasterize
    m_rasterize_ctx.fwd_input = data.input;
    m_rasterizer->forward(m_rasterize_ctx);

    // Export the rasterized image if enabled
    if (m_config.export_rasterized) {
      cv::Mat img = to_opencv();
      if (!img.empty()) {
        std::string rasterized_path = out_dir + "/" + std::to_string(data.input.timestamp) + ".png";
        if (!cv::imwrite(rasterized_path, img)) {
          log_error("Failed to export rasterized image to: {}", rasterized_path);
        }
      }
    }

    for (const auto &item : m_metrics) {
      auto value = item.metric->evaluate(m_rasterize_ctx.fwd_output.image,
                                         data.output.image);
      metrics[item.name].push_back(value);
    }
  }

  // Export the metrics to the out_dir in csv format
  const std::string csv_path = out_dir + "/metrics.csv";
  
  // Input validation: ensure data consistency
  if (timestamps.empty()) {
    log_warning("No timestamps available for CSV export");
    return;
  }
  
  // Validate that all metric vectors have the same size as timestamps
  for (const auto& metric_pair : metrics) {
    if (metric_pair.second.size() != timestamps.size()) {
      log_error(
          "Metric '{}' has {} values but {} timestamps - skipping CSV export",
          metric_pair.first, metric_pair.second.size(), timestamps.size());
      return;
    }
  }

  try {
    // Use RAII pattern with proper file stream management
    std::ofstream csv_file(csv_path);
    if (!csv_file.is_open()) {
      throw std::runtime_error("Failed to open file for writing: " + csv_path);
    }

    // Set precision for floating point numbers
    csv_file << std::fixed << std::setprecision(6);

    // Write header row (avoid trailing comma)
    csv_file << "timestamp";
    for (const auto &metric_pair : metrics) {
      csv_file << "," << metric_pair.first;
    }
    csv_file << "\n";

    // Write data rows
    for (size_t i = 0; i < timestamps.size(); ++i) {
      csv_file << timestamps[i];
      for (const auto &metric_pair : metrics) {
        csv_file << "," << metric_pair.second[i];
      }
      csv_file << "\n";

      // Check for write errors
      if (csv_file.fail()) {
        throw std::runtime_error("Error writing to CSV file: " + csv_path);
      }
    }

    log_info("Successfully exported {} metrics for {} samples to: {}",
             metrics.size(), timestamps.size(), csv_path);

  } catch (const std::exception &e) {
    log_error("Failed to export metrics to CSV: {}", e.what());
    throw; // Re-throw to allow caller to handle if needed
  }

  // Print the metrics statistics to stdout
  for (const auto &metric_pair : metrics) {
    float mean = std::accumulate(metric_pair.second.begin(), metric_pair.second.end(), 0.0f) / metric_pair.second.size();
    float std = std::sqrt(std::transform_reduce(metric_pair.second.begin(), metric_pair.second.end(), 0.0f,
                                                std::plus<float>(),
                                                [mean](float x) { return (x - mean) * (x - mean); }) 
                         / metric_pair.second.size());
    // log_info("Metric {}: mean = {:.6f}, std = {:.6f}", metric_pair.first, mean, std);
    std::cout << fmt::format("Metric {}: mean = {:.6f}, std = {:.6f}\n", metric_pair.first, mean, std);
  }
  
  // Restore training resolution if progressive resolution is enabled
  if (m_config.enable_progressive_resolution) {
    ImageShape training_shape = scale_image_shape(m_dataloader->get_dataset()->image_shape(), 
                                                  calculate_resolution_scale(m_state.current_step));
    if (training_shape.width != current_shape.width || training_shape.height != current_shape.height) {
      log_info("Restoring training resolution {}x{} after testing", training_shape.width, training_shape.height);
      set_render_resolution({training_shape.width, training_shape.height, 1});
    }
  }
}

float Orchestrator::accumulate_loss() {
  if (!m_loss_buffer) {
    return 0.0f;
  }

  ImageShape shape = m_rasterize_ctx.fwd_output.image.shape;
  //? the unused pixels in the padded area are set to zero during loss computation
  //! fix the shape is not compatible with the tile-based design.
  return gpu_sum(m_loss_buffer->data(), shape.width * shape.height * 3);
}

void Orchestrator::stop_training() {
  m_state.should_stop = true;
}

void Orchestrator::reset() {
  m_state.current_step = 0;
  m_state.current_loss = 0.0f;

  m_state.should_stop = false;
  m_state.start_time = std::chrono::steady_clock::now();
  m_state.last_log_time = m_state.start_time;
  
  if (m_dataloader) {
    m_dataloader->reset();
  }
  
  if (m_strategy) {
    m_strategy->reset();
  }
  
  if (m_lr_scheduler) {
    m_lr_scheduler->reset();
  }
  
  // Learning rate is now managed by the scheduler-optimizer system
}

void Orchestrator::update_config(const OrchestratorConfig& config) {
  m_config = config;
}

void Orchestrator::initialize() {
  if (m_major_stream) {
    CUDA_CHECK_THROW(cudaStreamDestroy(m_major_stream));
  }
  CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&m_major_stream, cudaStreamNonBlocking));

  m_rasterize_ctx.stream = m_major_stream;

  // Get image dimensions from the first data sample
  auto base_shape = m_dataloader->get_dataset()->image_shape();
  
  // Always allocate buffers for full resolution to avoid reallocations during training
  uint32_t full_pad_width = base_shape.padded_width();
  uint32_t full_pad_height = base_shape.padded_height();
  size_t full_buffer_size = full_pad_width * full_pad_height * 4;  // RGBA
  
  // Initialize GPU memory buffers with full resolution size
  m_loss_buffer = std::make_unique<GPUMemory<float>>(full_buffer_size);
  m_render_buffer = std::make_unique<GPUMemory<float>>(full_buffer_size);
  m_image_grad_buffer = std::make_unique<GPUMemory<float>>(full_buffer_size);
  
  log_info("Allocated GPU buffers for full resolution {}x{} (size: {} MB)", 
           base_shape.width, base_shape.height, 
           (full_buffer_size * sizeof(float) * 3) / (1024 * 1024));
  
  // Determine initial training resolution
  ImageShape training_shape = base_shape;
  if (m_config.enable_progressive_resolution) {
    float initial_scale = calculate_resolution_scale(0);  // Get scale for step 0
    training_shape = scale_image_shape(base_shape, initial_scale);
    log_info("Starting with progressive resolution {}x{} (scale: {:.2f})", 
             training_shape.width, training_shape.height, initial_scale);
  } else {
    log_info("Start from full resolution {}x{}", training_shape.width, training_shape.height);
  }
  set_render_resolution({training_shape.width, training_shape.height, 1});

  uint32_t width = training_shape.width;
  uint32_t height = training_shape.height;

  ImageShape rgb_shape{width, height, 3};
  ImageShape alpha_shape{width, height, 1};
  Image render_rgb = Image(rgb_shape, ImageDataType::Float32, m_render_buffer->data());
  Image render_alpha = Image(alpha_shape, ImageDataType::Float32, m_render_buffer->data() + rgb_shape.padded_size());
  Image grad_rgb = Image(rgb_shape, ImageDataType::Float32, m_image_grad_buffer->data());
  Image grad_alpha = Image(alpha_shape, ImageDataType::Float32, m_image_grad_buffer->data() + rgb_shape.padded_size());

  // Setup rasterization context
  m_rasterize_ctx.inference = false; // Training mode
  m_rasterize_ctx.fwd_input.width = width;
  m_rasterize_ctx.fwd_input.height = height;
  m_rasterize_ctx.fwd_input.near = m_config.near_plane;
  m_rasterize_ctx.fwd_input.far = m_config.far_plane;
  m_rasterize_ctx.grad_scaler = m_config.grad_scaler;

  // Setup output images
  m_rasterize_ctx.fwd_output.image = render_rgb;
  m_rasterize_ctx.fwd_output.alpha = render_alpha;
  // ... gradient to output image
  m_rasterize_ctx.grad_output.image = grad_rgb;
  m_rasterize_ctx.grad_output.alpha = grad_alpha;
  // ... gradient to gaussian parameters
  m_rasterize_ctx.gaussians_grad = m_gradients;

  // Setup loss context
  m_loss_ctx.loss = Image(rgb_shape, ImageDataType::Float32, m_loss_buffer->data());
  // TODO: alpha is ignored for now
  m_loss_ctx.pred = render_rgb;
  m_loss_ctx.grad = grad_rgb;
  log_info("Setup trainer buffers with image shape: {}", to_string(rgb_shape));

  // Setup output folder
  ensure(m_config.out_dir);
  reorder_gaussians();
}

float Orchestrator::compute_learning_rate() const {
  // Learning rate is now controlled by the scheduler through the optimizer
  if (m_optimizer) {
    return m_optimizer->get_lr();
  }
  return 0.0f;
}

void Orchestrator::set_lr_scheduler(std::shared_ptr<LrSchedulerBase> scheduler) {
  m_lr_scheduler = scheduler;
  if (m_lr_scheduler) {
    m_lr_scheduler->reset();
  }
}

std::shared_ptr<LrSchedulerBase> Orchestrator::get_lr_scheduler() const {
  return m_lr_scheduler;
}

std::shared_ptr<OptimizerBase> Orchestrator::get_optimizer() const {
  return m_optimizer;
}

void Orchestrator::update_sh_degree() {
  if (m_state.current_step % m_config.sh_degree_interval == 0) {
    size_t new_degree = std::min(
      m_state.current_step / m_config.sh_degree_interval,
      m_config.max_sh_degree
    );
    if (new_degree != m_gaussians->get_sh_degree()) {
      log_info("Updating SH degree to {}", new_degree);
    }
    m_gaussians->set_sh_degree(static_cast<int>(new_degree));
  }
}

void Orchestrator::evaluate_losses(const GPUBatchInputOutput& data) {
  m_loss_ctx.target = data.output.image;
  m_loss_ctx.pred = m_rasterize_ctx.fwd_output.image;
  for (const auto& loss_component : m_losses) {
    // Apply gradient scaler to loss weight
    const float w = loss_component.weight * m_config.grad_scaler;
    loss_component.loss->evaluate(m_loss_ctx, w);
  }
}

std::vector<float> Orchestrator::evaluate_metrics() {
  std::vector<float> metric_values;
  metric_values.reserve(m_metrics.size());
  
  for (const auto& metric_component : m_metrics) {
    float value = metric_component.metric->evaluate(
      m_loss_ctx.pred,
      m_loss_ctx.target
    );
    metric_values.push_back(value);
  }
  
  return metric_values;
}

bool Orchestrator::should_early_stop() const {
  // Simple early stopping based on loss threshold
  // More sophisticated implementations could track loss history
  return m_state.current_loss < m_config.early_stopping_threshold;
}

void Orchestrator::set_params(const json& j) {
  m_config.from_json(j);
}

json Orchestrator::get_params() const {
  return m_config.to_json();
}

void Orchestrator::validate_setup() const {
  if (!m_gaussians) {
    throw std::runtime_error("Gaussians not set. Call set_gaussians() before training.");
  }
  if (!m_gradients) {
    throw std::runtime_error("Gradients not set. Call set_gaussians() before training.");
  }
  if (!m_rasterizer) {
    throw std::runtime_error("Rasterizer not set. Call set_rasterizer() before training.");
  }
  if (!m_dataloader) {
    throw std::runtime_error("Dataloader not set. Call set_dataloader() before training.");
  }
  if (!m_optimizer) {
    throw std::runtime_error("Optimizer not set. Call set_optimizer() before training.");
  }
  if (m_losses.empty()) {
    throw std::runtime_error("No loss functions added. Call add_loss() before training.");
  }
  
  // Validate progressive resolution configuration
  if (m_config.enable_progressive_resolution) {
    if (m_config.resolution_milestones.empty()) {
      throw std::runtime_error("Progressive resolution enabled but no milestones specified.");
    }
    if (m_config.resolution_scales.empty()) {
      throw std::runtime_error("Progressive resolution enabled but no scales specified.");
    }
    if (m_config.resolution_milestones.size() != m_config.resolution_scales.size()) {
      throw std::runtime_error("Resolution milestones and scales must have the same size.");
    }
    
    // Check that milestones are in ascending order
    for (size_t i = 1; i < m_config.resolution_milestones.size(); ++i) {
      if (m_config.resolution_milestones[i] <= m_config.resolution_milestones[i-1]) {
        throw std::runtime_error("Resolution milestones must be in ascending order.");
      }
    }
    
    // Check that scales are valid (between 0 and 1)
    for (float scale : m_config.resolution_scales) {
      if (scale <= 0.0f || scale > 1.0f) {
        throw std::runtime_error("Resolution scales must be between 0 and 1.");
      }
    }
    
    log_info("Progressive resolution validation passed: {} milestones, scales from {:.2f} to {:.2f}", 
             m_config.resolution_milestones.size(), 
             m_config.resolution_scales.front(), 
             m_config.resolution_scales.back());
  }
}

cv::Mat Orchestrator::to_opencv() const {
  if (m_rasterize_ctx.fwd_output.image.data == nullptr) {
    log_warning("Orchestrator::to_opencv() - No data in current rasterizer context");
    return cv::Mat();
  }
  
  auto shape = m_rasterize_ctx.fwd_output.image.shape;
  int width = shape.width, height = shape.height;
  auto pad_width = shape.padded_width();
  auto channel_stride = shape.padded_height() * pad_width;

  // Copy GPU rendered image to CPU for visualization
  std::vector<float> cpu_image(shape.padded_size());
  CUDA_CHECK_THROW(
      cudaMemcpy(cpu_image.data(), 
        m_rasterize_ctx.fwd_output.image.data,
        // m_loss_ctx.target.data,
        shape.padded_size() * sizeof(float), cudaMemcpyDeviceToHost));
  // Convert float RGB to 8-bit BGR for OpenCV
  cv::Mat img(height, width, CV_8UC3);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      // Convert from HWC (RGB) to HWC (BGR)
      // const int pixel_offset = (y * width + x) * 3;
      const int pixel_offset = get_linear_index(y, x, pad_width);
      const int r_idx = pixel_offset + 0 * channel_stride; // R channel offset
      const int g_idx = pixel_offset + 1 * channel_stride; // G channel offset 
      const int b_idx = pixel_offset + 2 * channel_stride; // B channel offset
      
      img.at<cv::Vec3b>(y, x)[0] = static_cast<uint8_t>(std::clamp(cpu_image[b_idx] * 255.0f, 0.0f, 255.0f));  // B
      img.at<cv::Vec3b>(y, x)[1] = static_cast<uint8_t>(std::clamp(cpu_image[g_idx] * 255.0f, 0.0f, 255.0f));  // G
      img.at<cv::Vec3b>(y, x)[2] = static_cast<uint8_t>(std::clamp(cpu_image[r_idx] * 255.0f, 0.0f, 255.0f));  // R
    }
  }
  return img;
}

float Orchestrator::calculate_resolution_scale(size_t current_step) const {
  if (!m_config.enable_progressive_resolution || m_config.resolution_milestones.empty()) {
    return 1.0f;  // Full resolution if disabled
  }
  
  // Find the appropriate scale for the current step
  float current_scale = 1.0f;
  for (size_t i = 0; i < m_config.resolution_milestones.size(); ++i) {
    if (current_step >= m_config.resolution_milestones[i]) {
      if (i < m_config.resolution_scales.size()) {
        current_scale = m_config.resolution_scales[i];
      }
    } else {
      break;
    }
  }
  
  return current_scale;
}

ImageShape Orchestrator::scale_image_shape(const ImageShape& original_shape, float scale) {
  if (scale <= 0.0f || scale > 1.0f) {
    throw std::invalid_argument("Resolution scale must be in range (0, 1]");
  }
  if (fabs(scale - 1.0f) < 1e-6) {
    return original_shape;  // No scaling needed, also no rounding is needed.
  }

  // Calculate scaled dimensions, ensuring they're at least 1
  uint32_t scaled_width_tile = std::clamp(kImageTile * (static_cast<uint32_t>(original_shape.width * (scale / kImageTile))),
                                          kImageTile, original_shape.width);
  uint32_t scaled_height_tile = std::clamp(kImageTile * (static_cast<uint32_t>(original_shape.height * (scale / kImageTile))),
                                           kImageTile, original_shape.height);

  return ImageShape{scaled_width_tile, scaled_height_tile, original_shape.channel};
}

void Orchestrator::set_render_resolution(const ImageShape& new_shape) {
  // Calculate padded dimensions for the new shape
  uint32_t padded_width = new_shape.padded_width();
  uint32_t padded_height = new_shape.padded_height();
  uint32_t channel_stride = padded_width * padded_height;

  if (new_shape.width  > m_dataloader->get_dataset()->image_shape().width ||
      new_shape.height > m_dataloader->get_dataset()->image_shape().height) {
    throw std::invalid_argument("Not a valid buffer shape.");
  }

  // Create new image objects with the reallocated buffers
  ImageShape rgb_shape{new_shape.width, new_shape.height, 3};
  ImageShape alpha_shape{new_shape.width, new_shape.height, 1};

  Image render_rgb = Image(rgb_shape, ImageDataType::Float32, m_render_buffer->data());
  Image render_alpha = Image(alpha_shape, ImageDataType::Float32, m_render_buffer->data() + channel_stride * 3);
  Image grad_rgb = Image(rgb_shape, ImageDataType::Float32, m_image_grad_buffer->data());
  Image grad_alpha = Image(alpha_shape, ImageDataType::Float32, m_image_grad_buffer->data() + channel_stride * 3);

  // Update rasterization context with new dimensions and images
  m_rasterize_ctx.fwd_input.width = new_shape.width;
  m_rasterize_ctx.fwd_input.height = new_shape.height;

  // Update output images
  m_rasterize_ctx.fwd_output.image = render_rgb;
  m_rasterize_ctx.fwd_output.alpha = render_alpha;
  m_rasterize_ctx.grad_output.image = grad_rgb;
  m_rasterize_ctx.grad_output.alpha = grad_alpha;

  // Update loss context
  m_loss_ctx.loss = Image(rgb_shape, ImageDataType::Float32, m_loss_buffer->data());
  m_loss_ctx.pred = render_rgb;
  m_loss_ctx.grad = grad_rgb;

  m_dataloader->set_output_shape(rgb_shape);
}

void Orchestrator::update_resolution(size_t current_step) {
  // Calculate current resolution scale
  float scale = calculate_resolution_scale(current_step);
  
  // Get base shape from dataset
  ImageShape base_shape = m_dataloader->get_dataset()->image_shape();
  
  // Scale the image shape
  ImageShape new_shape = scale_image_shape(base_shape, scale);
  
  // Check if resolution has changed
  if (new_shape.width != m_rasterize_ctx.fwd_input.width || 
      new_shape.height != m_rasterize_ctx.fwd_input.height) {
    
    log_info("Updating resolution from {}x{} to {}x{} at step {}", 
             m_rasterize_ctx.fwd_input.width, m_rasterize_ctx.fwd_input.height,
             new_shape.width, new_shape.height, current_step);
    
    // Reallocate buffers for new resolution
    set_render_resolution({new_shape.width, new_shape.height, 1});
  }
}

static __global__ void densification_update(const float *__restrict__ old_info,
                                            float *__restrict__ new_info,
                                            uint n, uint *old_idx) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

#pragma unroll
  for (uint coef = 0; coef < 2u; ++ coef) {
    float old_density = old_info[old_idx[i] + coef * n];
    new_info[i + coef * n] = old_density;
  }
}

void Orchestrator::reorder_gaussians() {
  NVTX3_FUNC_RANGE();
  log_info("Reordering gaussians by Morton code for better spatial locality...");
  auto& pos = m_gaussians->means();
  uint n = m_gaussians->size();

  // Reorder gaussians by Morton code
  auto idx = reorder(thrust::raw_pointer_cast(pos.data()), n, nullptr);
  
  // Apply reordering to gaussians
  m_gaussians->reorder(thrust::raw_pointer_cast(idx.data()));
  m_gradients->reorder(thrust::raw_pointer_cast(idx.data()));
  m_optimizer->reorder(thrust::raw_pointer_cast(idx.data()));

  if (m_rasterize_ctx.densification_info) {
    auto new_info = std::make_shared<GPUBuffer<float>>(n * 2);
    densification_update<<<(n + 255) / 256, 256>>>(
      m_rasterize_ctx.densification_info->data(),
      new_info->data(),
      n,
      thrust::raw_pointer_cast(idx.data()));
    m_rasterize_ctx.densification_info = new_info;
  }
}

}  // namespace tinygs