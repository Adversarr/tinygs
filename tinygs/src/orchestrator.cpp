#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <vector>

#include "tinygs/core/pointcloud.hpp"
#include "tinygs/platform/backend_build.hpp"
#include "tinygs/platform/runtime_factory.hpp"
#include "tinygs/platform/buffer_utils.hpp"
#include "tinygs/cuda/reduce.hpp"
#include "tinygs/orchestrator.hpp"
#include "tinygs/utils/file.hpp"
#include "tinygs/utils/scope_timer.hpp"
#include "tinygs/core/gaussian.hpp"
#include "tinygs/utils/image_format.hpp"
#include "tinygs/common.hpp"

namespace tinygs {

static inline void backend_check_throw(const BackendError& status, const char* operation) {
  if (status.ok()) {
    return;
  }
  throw std::runtime_error(
      "Runtime operation failed (" + std::string(operation) + "): " + to_string(status));
}

json OrchestratorConfig::to_json() const {
  json j;
  j["max_steps"] = max_steps;
  j["means_accumulate_grad_steps"] = means_accumulate_grad_steps;
  j["shs_accumulate_grad_steps"] = shs_accumulate_grad_steps;
  j["opacities_accumulate_grad_steps"] = opacities_accumulate_grad_steps;
  j["scales_accumulate_grad_steps"] = scales_accumulate_grad_steps;
  j["rotations_accumulate_grad_steps"] = rotations_accumulate_grad_steps;
  j["max_seconds"] = max_seconds;
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
  j["export_full_features"] = export_full_features;
  j["record_trajectory"] = record_trajectory;
  j["start_pose_opt"] = start_pose_opt;
  j["scene_scale_recompute_interval"] = scene_scale_recompute_interval;
  j["reorder_gaussians_interval"] = reorder_gaussians_interval;
  j["train_data_type"] = to_string(train_data_type);
  j["eval_data_type"] = to_string(eval_data_type);
  j["debug_cuda_check_each_stage"] = debug_cuda_check_each_stage;
  j["debug_cuda_sync_each_stage"] = debug_cuda_sync_each_stage;
  j["debug_cuda_check_every"] = debug_cuda_check_every;
  j["debug_cuda_log_each_stage"] = debug_cuda_log_each_stage;
  return j;
}

void OrchestratorConfig::from_json(const json& j) {
  if (j.contains("max_steps")) max_steps = j["max_steps"].get<int>();
  if (j.contains("means_accumulate_grad_steps")) means_accumulate_grad_steps = j["means_accumulate_grad_steps"].get<size_t>();
  if (j.contains("shs_accumulate_grad_steps")) shs_accumulate_grad_steps = j["shs_accumulate_grad_steps"].get<size_t>();
  if (j.contains("opacities_accumulate_grad_steps")) opacities_accumulate_grad_steps = j["opacities_accumulate_grad_steps"].get<size_t>();
  if (j.contains("scales_accumulate_grad_steps")) scales_accumulate_grad_steps = j["scales_accumulate_grad_steps"].get<size_t>();
  if (j.contains("rotations_accumulate_grad_steps")) rotations_accumulate_grad_steps = j["rotations_accumulate_grad_steps"].get<size_t>();
  if (j.contains("max_seconds")) max_seconds = j["max_seconds"].get<int>();
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
  if (j.contains("export_full_features")) export_full_features = j["export_full_features"].get<bool>();
  if (j.contains("record_trajectory")) record_trajectory = j["record_trajectory"].get<bool>();
  if (j.contains("start_pose_opt")) start_pose_opt = j["start_pose_opt"].get<size_t>();

  if (j.contains("scene_scale_recompute_interval")) scene_scale_recompute_interval = j["scene_scale_recompute_interval"].get<size_t>();
  if (j.contains("reorder_gaussians_interval")) reorder_gaussians_interval = j["reorder_gaussians_interval"].get<size_t>();
  if (j.contains("train_data_type")) train_data_type = from_string<DataType>(j["train_data_type"].get<std::string>());
  if (j.contains("eval_data_type")) eval_data_type = from_string<DataType>(j["eval_data_type"].get<std::string>());
  if (j.contains("debug_cuda_check_each_stage")) debug_cuda_check_each_stage = j["debug_cuda_check_each_stage"].get<bool>();
  if (j.contains("debug_cuda_sync_each_stage")) debug_cuda_sync_each_stage = j["debug_cuda_sync_each_stage"].get<bool>();
  if (j.contains("debug_cuda_check_every")) debug_cuda_check_every = j["debug_cuda_check_every"].get<size_t>();
  if (j.contains("debug_cuda_log_each_stage")) debug_cuda_log_each_stage = j["debug_cuda_log_each_stage"].get<bool>();
}


void Orchestrator::recompute_scene_scale() {
  auto ds = m_dataloader->get_dataset();
  auto pc = m_gaussians->means();

  vec3 avg_pc_mean;
  gpu_mean_vec3(pc.data(), static_cast<int>(pc.size()), avg_pc_mean, m_major_queue.get());

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
  if (m_strategy) m_strategy->set_rasterizer(m_rasterizer);
}

void Orchestrator::set_dataloader(std::shared_ptr<DataLoaderBase> dataloader) {
  m_dataloader = dataloader;
  if (m_strategy) m_strategy->set_dataloader(m_dataloader);
}

void Orchestrator::set_test_dataloader(std::shared_ptr<DataLoaderBase> dataloader) {
  m_test_dataloader = dataloader;
}

std::shared_ptr<DataLoaderBase> Orchestrator::get_test_dataloader() const {
  return m_test_dataloader;
}

void Orchestrator::set_optimizer(std::shared_ptr<OptimizerBase> optimizer) {
  m_optimizer = optimizer;
}

void Orchestrator::set_pose_opt(std::shared_ptr<PoseOptBase> pose_opt) {
  m_pose_opt = pose_opt;
}

void Orchestrator::set_strategy(std::shared_ptr<StrategyBase> strategy) {
  m_strategy = strategy;
  // Wire rasterizer and dataloader so strategies like FastGS can render
  // additional views for multi-view metric scoring.
  if (m_strategy) {
    if (m_rasterizer) m_strategy->set_rasterizer(m_rasterizer);
    if (m_dataloader) m_strategy->set_dataloader(m_dataloader);
  }
}

void Orchestrator::set_backend_runtime(std::shared_ptr<BackendRuntime> backend_runtime) {
  m_backend_runtime = std::move(backend_runtime);
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

  while (!m_state.should_stop && m_state.current_step < m_config.max_steps) {
    // Time-based stopping
    if (m_config.max_seconds > 0) {
      auto now = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - m_state.start_time).count();
      if (elapsed >= static_cast<long>(m_config.max_seconds)) {
        log_info("Max seconds ({}) reached at step {}", m_config.max_seconds, m_state.current_step);
        m_state.should_stop = true;
        test_step();
        break;
      }
    }
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
  const bool means_cycle_start = (m_state.current_step % group_accumulate_steps(OptimParamGroup::Means)) == 0;
  const bool shs_cycle_start = (m_state.current_step % group_accumulate_steps(OptimParamGroup::Shs)) == 0;
  const bool opacities_cycle_start = (m_state.current_step % group_accumulate_steps(OptimParamGroup::Opacities)) == 0;
  const bool scales_cycle_start = (m_state.current_step % group_accumulate_steps(OptimParamGroup::Scales)) == 0;
  const bool rotations_cycle_start = (m_state.current_step % group_accumulate_steps(OptimParamGroup::Rotations)) == 0;
  const bool any_cycle_start = means_cycle_start || shs_cycle_start || opacities_cycle_start ||
                               scales_cycle_start || rotations_cycle_start;
  if (any_cycle_start && m_pre_step_callback) {
    m_pre_step_callback(m_state);
  }

  auto clear_group_gradients = [this](OptimParamGroup group) {
    switch (group) {
      case OptimParamGroup::Means:
        fill_buffer_zero_async(m_backend_runtime, m_major_queue, m_gradients->means().view());
        break;
      case OptimParamGroup::Shs:
        fill_buffer_zero_async(m_backend_runtime, m_major_queue, m_gradients->sh0().view());
        fill_buffer_zero_async(m_backend_runtime, m_major_queue, m_gradients->sh1().view());
        fill_buffer_zero_async(m_backend_runtime, m_major_queue, m_gradients->sh2().view());
        fill_buffer_zero_async(m_backend_runtime, m_major_queue, m_gradients->sh3().view());
        break;
      case OptimParamGroup::Opacities:
        fill_buffer_zero_async(m_backend_runtime, m_major_queue, m_gradients->opacities().view());
        break;
      case OptimParamGroup::Scales:
        fill_buffer_zero_async(m_backend_runtime, m_major_queue, m_gradients->scales().view());
        break;
      case OptimParamGroup::Rotations:
        fill_buffer_zero_async(m_backend_runtime, m_major_queue, m_gradients->rotations().view());
        break;
    }
  };
  if (means_cycle_start) clear_group_gradients(OptimParamGroup::Means);
  if (shs_cycle_start) clear_group_gradients(OptimParamGroup::Shs);
  if (opacities_cycle_start) clear_group_gradients(OptimParamGroup::Opacities);
  if (scales_cycle_start) clear_group_gradients(OptimParamGroup::Scales);
  if (rotations_cycle_start) clear_group_gradients(OptimParamGroup::Rotations);
  fill_buffer_zero_async(m_backend_runtime, m_major_queue, m_loss_buffer);
  fill_buffer_zero_async(m_backend_runtime, m_major_queue, m_image_grad_buffer);

  auto data = m_dataloader->next();

  m_rasterize_ctx.fwd_input = data.input;

  if (m_pose_opt) {
    mat4x4 w2c = data.input.w2c;
    const uuid_t timestamp = m_rasterize_ctx.fwd_input.timestamp;
    w2c = m_pose_opt->query(timestamp, w2c);
    m_rasterize_ctx.fwd_input.w2c = w2c;
  }

  m_rasterizer->forward(m_rasterize_ctx);

  evaluate_losses(data);

  m_rasterize_ctx.grad_output.image = m_loss_ctx.grad;

  m_rasterizer->backward(m_rasterize_ctx);
  auto grad_w2c = m_rasterize_ctx.grad_input.w2c;
  if (m_pose_opt && m_state.current_step >= m_config.start_pose_opt) {
    float lr = m_optimizer ? m_optimizer->get_lr(OptimParamGroup::Means) : 1.0f;
    m_pose_opt->update(m_rasterize_ctx.fwd_input.timestamp, grad_w2c, lr);
  }

  const bool step_means = should_step_group(OptimParamGroup::Means, m_state.current_step);
  const bool step_shs = should_step_group(OptimParamGroup::Shs, m_state.current_step);
  const bool step_opacities = should_step_group(OptimParamGroup::Opacities, m_state.current_step);
  const bool step_scales = should_step_group(OptimParamGroup::Scales, m_state.current_step);
  const bool step_rotations = should_step_group(OptimParamGroup::Rotations, m_state.current_step);

  if (step_means || step_shs || step_opacities || step_scales || step_rotations) {
    if (step_means && m_means_lr_scheduler) m_means_lr_scheduler->step();
    if (step_shs && m_shs_lr_scheduler) m_shs_lr_scheduler->step();
    if (step_opacities && m_opacities_lr_scheduler) m_opacities_lr_scheduler->step();
    if (step_scales && m_scales_lr_scheduler) m_scales_lr_scheduler->step();
    if (step_rotations && m_rotations_lr_scheduler) m_rotations_lr_scheduler->step();

    const float inv_grad_scale = 1.0f / m_config.grad_scaler;
    GroupStepConfig step_cfg;
    step_cfg.update_means = step_means;
    step_cfg.update_shs = step_shs;
    step_cfg.update_opacities = step_opacities;
    step_cfg.update_scales = step_scales;
    step_cfg.update_rotations = step_rotations;
    step_cfg.means_scale = inv_grad_scale / static_cast<float>(group_accumulate_steps(OptimParamGroup::Means));
    step_cfg.shs_scale = inv_grad_scale / static_cast<float>(group_accumulate_steps(OptimParamGroup::Shs));
    step_cfg.opacities_scale = inv_grad_scale / static_cast<float>(group_accumulate_steps(OptimParamGroup::Opacities));
    step_cfg.scales_scale = inv_grad_scale / static_cast<float>(group_accumulate_steps(OptimParamGroup::Scales));
    step_cfg.rotations_scale = inv_grad_scale / static_cast<float>(group_accumulate_steps(OptimParamGroup::Rotations));
    m_optimizer->step(step_cfg, m_major_queue.get());
  }

  if ((step_means || step_shs || step_opacities || step_scales || step_rotations) && m_strategy) {
    m_strategy->step(m_rasterize_ctx);
  }

  if (m_state.current_step > 0) {
    if (m_config.scene_scale_recompute_interval > 0 &&
        m_state.current_step % m_config.scene_scale_recompute_interval == 0) {
      recompute_scene_scale();
    }
    if (m_config.reorder_gaussians_interval > 0 &&
        m_state.current_step % m_config.reorder_gaussians_interval == 0) {
      reorder_gaussians();
    }
  }

  update_sh_degree();

  if (m_post_step_callback) {
    m_post_step_callback(m_state);
  }

  if (m_checkpoint_callback && m_state.current_step % m_config.checkpoint_interval == 0) {
    m_checkpoint_callback(m_state);
  }

  m_state.current_step++;
}

void Orchestrator::test_step() {
  DataLoaderBase* use_loader = m_test_dataloader ? m_test_dataloader.get() : m_dataloader.get();
  eval(use_loader);
}

std::unordered_map<std::string, float> Orchestrator::eval(DataLoaderBase* loader) {
  const ImageShape current_shape{m_rasterize_ctx.fwd_input.width, m_rasterize_ctx.fwd_input.height, 3};
  const DataType current_dtype = m_active_data_type;

  m_active_data_type = m_config.eval_data_type;
  DataLoaderBase* effective_loader = loader ? loader : m_dataloader.get();

  // Use the effective dataset resolution (train/test can differ).
  ImageShape eval_shape = effective_loader->get_dataset()->image_shape();

  // Update rasterizer context buffers for the eval resolution
  set_render_resolution(eval_shape);

  // Switch dataloader output dtype for eval
  effective_loader->set_params(json{{"data_type", to_string(m_active_data_type)}});
  effective_loader->reset();

  std::string out_dir = m_config.out_dir + "/" + std::to_string(m_state.current_step);
  ensure(out_dir);

  const auto total_samples = effective_loader->get_dataset()->size();
  std::map<std::string, std::vector<float>> metrics;
  std::vector<uuid_t> timestamps;
  log_info("Start Evaluation on {} samples, DataType={}",
           total_samples, to_string(m_active_data_type));
  auto start = std::chrono::high_resolution_clock::now();
  for (size_t idx = 0; idx < total_samples; ++idx) {
    auto data = effective_loader->next();
    timestamps.push_back(data.input.timestamp);

    // Rasterize
    m_rasterize_ctx.fwd_input = data.input;
    if (m_pose_opt) {
      mat4x4 w2c = data.input.w2c;
      const uuid_t timestamp = m_rasterize_ctx.fwd_input.timestamp;
      w2c = m_pose_opt->query(timestamp, w2c);
      m_rasterize_ctx.fwd_input.w2c = w2c;
    }
    m_rasterizer->forward(m_rasterize_ctx);

    // Wait for the rasterization to finish
    backend_check_throw(m_backend_runtime->synchronize_queue(m_major_queue), "eval synchronize");

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
      auto value = item.metric->evaluate(m_rasterize_ctx.fwd_output.image, data.output.image);
      metrics[item.name].push_back(value);
    }
  }

  // Export the metrics to the out_dir in csv format
  const std::string csv_path = out_dir + "/metrics.csv";

  if (timestamps.empty()) {
    log_warning("No timestamps available for CSV export");
  } else {
    for (const auto& metric_pair : metrics) {
      if (metric_pair.second.size() != timestamps.size()) {
        log_error("Metric '{}' has {} values but {} timestamps - skipping CSV export",
                  metric_pair.first, metric_pair.second.size(), timestamps.size());
        metrics.clear();
        break;
      }
    }
  }

  try {
    std::ofstream csv_file(csv_path);
    if (csv_file.is_open()) {
      csv_file << std::fixed << std::setprecision(6);
      csv_file << "timestamp";
      for (const auto &metric_pair : metrics) {
        csv_file << "," << metric_pair.first;
      }
      csv_file << "\n";
      for (size_t i = 0; i < timestamps.size(); ++i) {
        csv_file << timestamps[i];
        for (const auto &metric_pair : metrics) {
          csv_file << "," << metric_pair.second[i];
        }
        csv_file << "\n";
      }
      log_info("Successfully exported {} metrics for {} samples to: {}",
               metrics.size(), timestamps.size(), csv_path);
    } else {
      log_error("Failed to open file for writing: {}", csv_path);
    }
  } catch (const std::exception &e) {
    log_error("Failed to export metrics to CSV: {}", e.what());
  }

  // Print the metrics statistics to stdout
  for (const auto &metric_pair : metrics) {
    float mean = std::accumulate(metric_pair.second.begin(), metric_pair.second.end(), 0.0f) / metric_pair.second.size();
    float std = std::sqrt(std::transform_reduce(metric_pair.second.begin(), metric_pair.second.end(), 0.0f,
                                                std::plus<float>(),
                                                [mean](float x) { return (x - mean) * (x - mean); }) 
                         / metric_pair.second.size());
    printf("[Step %zu] Metric %s: mean = %.6f, std = %.6f\n",
         m_state.current_step,
         metric_pair.first.c_str(),
         mean,
         std);
  }

  // Restore training dtype and resolution
  m_active_data_type = current_dtype;
  set_render_resolution({current_shape.width, current_shape.height, 3});

  // Restore dataloader dtype
  effective_loader->set_params(json{{"data_type", to_string(m_active_data_type)}});
  effective_loader->reset();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  log_info("Evaluation finished in {} ms", duration.count());

  // Export PLY at eval end
  Gaussian3d gs_host;
  m_gaussians->copy_to_host_async(gs_host, m_major_queue);
  backend_check_throw(m_backend_runtime->synchronize_queue(m_major_queue), "eval export points");
  save_ply(out_dir + "/points.ply", gs_host, m_config.export_full_features || m_state.should_stop);

  // Return mean metrics
  std::unordered_map<std::string, float> result;
  for (const auto &metric_pair : metrics) {
    if (!metric_pair.second.empty()) {
      float mean = std::accumulate(metric_pair.second.begin(), metric_pair.second.end(), 0.0f) / metric_pair.second.size();
      result[metric_pair.first] = mean;
    }
  }
  return result;
}

float Orchestrator::accumulate_loss() {
  if (!m_loss_buffer) {
    return 0.0f;
  }
  backend_check_throw(
      m_backend_runtime->synchronize_queue(m_major_queue), "accumulate_loss synchronize");
  ImageShape shape = m_rasterize_ctx.fwd_output.image.shape;
  //? the unused pixels in the padded area are set to zero during loss computation
  //! fix the shape is not compatible with the tile-based design.
  return gpu_sum(buffer_data<float>(m_loss_buffer), shape.padded_size());
}

void Orchestrator::stop_training() {
  m_state.should_stop = true;
}

bool Orchestrator::is_stop_requested() const {
  return m_state.should_stop;
}

void Orchestrator::reset() {
  m_state.current_step = 0;
  m_state.current_loss = 0.0f;

  m_state.should_stop = false;
  m_state.start_time = std::chrono::steady_clock::now();
  m_state.last_log_time = m_state.start_time;

  // Reset early stopping tracking
  m_best_loss = -1.0f;
  m_best_loss_step = 0;
  
  if (m_dataloader) {
    m_dataloader->reset();
  }
  
  if (m_strategy) {
    m_strategy->reset();
  }

  if (m_means_lr_scheduler) m_means_lr_scheduler->reset();
  if (m_shs_lr_scheduler) m_shs_lr_scheduler->reset();
  if (m_opacities_lr_scheduler) m_opacities_lr_scheduler->reset();
  if (m_scales_lr_scheduler) m_scales_lr_scheduler->reset();
  if (m_rotations_lr_scheduler) m_rotations_lr_scheduler->reset();
  
  // Learning rate is now managed by the scheduler-optimizer system
}

void Orchestrator::update_config(const OrchestratorConfig& config) {
  m_config = config;
}

void Orchestrator::initialize() {
  if (!m_backend_runtime) {
    BackendConfig backend_config;
    backend_config.type = compiled_backend_type();
    const auto runtime_result = create_backend_runtime(backend_config);
    backend_check_throw(runtime_result.error(), "initialize create backend runtime");
    m_backend_runtime = runtime_result.value();
  }
  CHECK_THROW(m_backend_runtime != nullptr);

  m_major_queue.reset();
  QueueDesc queue_desc;
  queue_desc.non_blocking = true;
  queue_desc.debug_name = "orchestrator_major";
  const auto queue_result = m_backend_runtime->create_queue(queue_desc);
  backend_check_throw(queue_result.error(), "initialize create major queue");
  m_major_queue = queue_result.value();
  CHECK_THROW(m_major_queue != nullptr);

  m_loss_ctx.queue = m_major_queue;
  m_rasterize_ctx.queue = m_major_queue;
  m_rasterize_ctx.runtime = m_backend_runtime;

  
  // Use dataset-owned image resolutions (train and optional test may differ).
  auto train_shape = m_dataloader->get_dataset()->image_shape();
  auto full_shape = train_shape;
  if (m_test_dataloader) {
    const auto test_shape = m_test_dataloader->get_dataset()->image_shape();
    full_shape.width = std::max(full_shape.width, test_shape.width);
    full_shape.height = std::max(full_shape.height, test_shape.height);
  }
  m_max_render_shape = full_shape;
  
  // Always allocate buffers for full resolution to avoid reallocations during training
  uint32_t full_pad_width = full_shape.padded_width();
  uint32_t full_pad_height = full_shape.padded_height();
  size_t full_buffer_size = full_pad_width * full_pad_height * 3;  // RGB elements count
  
  // Initialize GPU memory buffers with full resolution size
  const size_t full_buffer_bytes = full_buffer_size * sizeof(float);
  m_loss_buffer = create_device_buffer(m_backend_runtime, full_buffer_bytes, "loss_buffer");
  m_render_buffer = create_device_buffer(m_backend_runtime, full_buffer_bytes, "render_buffer");
  m_image_grad_buffer = create_device_buffer(m_backend_runtime, full_buffer_bytes, "image_grad_buffer");

  const double mb = static_cast<double>(full_buffer_size) * sizeof(float) / (1024.0 * 1024.0);
  log_info("Allocated GPU buffers for full resolution {}x{} (size: {:.2f} MB)", 
           full_shape.width, full_shape.height, mb);

  ImageShape training_shape = train_shape;
  log_info("Training resolution: {}x{}", training_shape.width, training_shape.height);
  log_info("Camera intrinsics: {}", to_string(m_dataloader->get_dataset()
                                                  ->get_camera_loader()
                                                  .get_camera_intrinsics()[0]));

  // Set active dtype for training
  m_active_data_type = m_config.train_data_type;
  log_info("Rasterize Precision: {}", to_string(m_active_data_type));

  set_render_resolution(training_shape);

  uint32_t width = training_shape.width;
  uint32_t height = training_shape.height;

  ImageShape rgb_shape{width, height, 3};
  Image render_rgb = Image(rgb_shape, m_active_data_type, buffer_data<float>(m_render_buffer));
  Image grad_rgb = Image(rgb_shape, m_active_data_type, buffer_data<float>(m_image_grad_buffer));

  // Setup rasterization context
  m_rasterize_ctx.inference = false; // Training mode
  m_rasterize_ctx.runtime = m_backend_runtime;
  m_rasterize_ctx.queue = m_major_queue;
  m_rasterize_ctx.fwd_input.width = width;
  m_rasterize_ctx.fwd_input.height = height;
  m_rasterize_ctx.fwd_input.near = m_config.near_plane;
  m_rasterize_ctx.fwd_input.far = m_config.far_plane;
  m_rasterize_ctx.grad_scaler = m_config.grad_scaler;

  // Setup output images
  m_rasterize_ctx.fwd_output.image = render_rgb;
  // ... gradient to output image
  m_rasterize_ctx.grad_output.image = grad_rgb;
  // ... gradient to gaussian parameters
  m_rasterize_ctx.gaussians_grad = m_gradients;

  // Setup loss context
  m_loss_ctx.loss = Image(rgb_shape, m_active_data_type, buffer_data<float>(m_loss_buffer));
  // TODO: alpha is ignored for now
  m_loss_ctx.pred = render_rgb;
  m_loss_ctx.grad = grad_rgb;
  log_info("Setup trainer buffers with image shape: {}", to_string(rgb_shape));

  // Setup output folder
  ensure(m_config.out_dir);
  if (m_config.reorder_gaussians_interval > 0) {
    reorder_gaussians();
  }

  if (m_config.train_data_type == DataType::Float16) {
    log_info("Using Float16 precision for training.");
  }
}

float Orchestrator::compute_learning_rate() const {
  // Means-group learning rate is used as representative scalar.
  if (m_optimizer) {
    return m_optimizer->get_lr(OptimParamGroup::Means);
  }
  return 0.0f;
}

void Orchestrator::set_lr_scheduler(OptimParamGroup group, std::shared_ptr<LrSchedulerBase> scheduler) {
  switch (group) {
    case OptimParamGroup::Means:
      m_means_lr_scheduler = scheduler;
      break;
    case OptimParamGroup::Shs:
      m_shs_lr_scheduler = scheduler;
      break;
    case OptimParamGroup::Opacities:
      m_opacities_lr_scheduler = scheduler;
      break;
    case OptimParamGroup::Scales:
      m_scales_lr_scheduler = scheduler;
      break;
    case OptimParamGroup::Rotations:
      m_rotations_lr_scheduler = scheduler;
      break;
  }
  auto target = group_scheduler(group);
  if (target) {
    target->reset();
  }
}

std::shared_ptr<LrSchedulerBase> Orchestrator::get_lr_scheduler(OptimParamGroup group) const {
  return group_scheduler(group);
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
  if (m_loss_ctx.pred.shape !=  m_loss_ctx.target.shape) {
    throw std::runtime_error("Prediction and target image shapes do not match in loss evaluation.");
  }

  float last_accum_loss = 0;
  std::map<std::string, float> loss_values;
  for (const auto& loss_component : m_losses) {
    // Apply gradient scaler to loss weight
    const float w = loss_component.weight * m_config.grad_scaler;
    loss_component.loss->evaluate(m_loss_ctx, w);

    if (m_config.record_trajectory || m_config.enable_early_stopping) {
      float accum_loss = accumulate_loss();
      loss_values[loss_component.loss->name()] = accum_loss - last_accum_loss;
      last_accum_loss = accum_loss;
    }
  }

  // Always update current_loss so that early stopping can observe it
  if (m_config.record_trajectory || m_config.enable_early_stopping) {
    m_state.current_loss = last_accum_loss;

    // Track best loss for patience-based early stopping
    if (m_best_loss < 0.0f || last_accum_loss < m_best_loss - m_config.early_stopping_threshold) {
      m_best_loss = last_accum_loss;
      m_best_loss_step = m_state.current_step;
    }
  }

  if (m_config.record_trajectory) {
    // Write per-step loss breakdown to CSV file
    const auto openmode = m_state.current_step == 0 ? std::ios::out : std::ios::app;
    std::ofstream loss_file(m_config.out_dir + "/loss.csv", openmode);
    if (!loss_file.is_open()) {
      log_error("Failed to open {}/loss.csv for writing.", m_config.out_dir);
    }

    if (m_state.current_step == 0) {
      loss_file << "step,timestamp,";
      for (const auto& loss_name : loss_values) {
        loss_file << loss_name.first << ",";
      }
      loss_file << "total_loss" << std::endl;
    } else {
      loss_file << m_state.current_step << "," << m_rasterize_ctx.fwd_input.timestamp << ",";
      for (const auto& loss_name : loss_values) {
        loss_file << loss_name.second << ",";
      }
      loss_file << last_accum_loss << std::endl;
    }
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
  // Patience-based early stopping: stop when loss has not improved by at least
  // early_stopping_threshold over the last early_stopping_patience steps.
  if (m_state.current_step < m_config.early_stopping_patience) {
    return false;  // Not enough history yet
  }

  // Use the best-seen loss tracked in m_best_loss (updated below in train_step flow).
  // If the current loss is still above (best + threshold), patience counter in the
  // caller will handle it. Here we use a simple check: if the loss hasn't meaningfully
  // decreased from its value patience-steps ago, stop.
  if (m_best_loss < 0.0f) {
    return false;  // No valid loss recorded yet
  }

  const float improvement = m_best_loss - m_state.current_loss;
  if (improvement < m_config.early_stopping_threshold &&
      m_state.current_step - m_best_loss_step >= m_config.early_stopping_patience) {
    return true;
  }
  return false;
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

  // Validate gradient accumulation configuration
  if (m_config.means_accumulate_grad_steps < 1 || m_config.shs_accumulate_grad_steps < 1 ||
      m_config.opacities_accumulate_grad_steps < 1 || m_config.scales_accumulate_grad_steps < 1 ||
      m_config.rotations_accumulate_grad_steps < 1) {
    throw std::runtime_error("All <group>_accumulate_grad_steps must be >= 1");
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
  CHECK_THROW(m_backend_runtime != nullptr);
  CHECK_THROW(m_major_queue != nullptr);
  std::vector<float> cpu_image(shape.padded_size());
  const void* src_ptr = nullptr;
  std::shared_ptr<BackendBuffer> fp32_tmp;
  if (m_rasterize_ctx.fwd_output.image.data_type == DataType::Float16) {
    // Convert FP16 buffer to FP32 on GPU before host copy
    fp32_tmp = create_device_buffer_for<float>(m_backend_runtime, shape.padded_size(), "fp32_tmp");
    half_to_float_gpu(buffer_data<float>(fp32_tmp),
                      reinterpret_cast<const float16_t*>(m_rasterize_ctx.fwd_output.image.data),
                      shape.padded_size(),
                      m_major_queue.get());
    src_ptr = buffer_data<float>(fp32_tmp);
  } else if (m_rasterize_ctx.fwd_output.image.data_type == DataType::Float32) {
    src_ptr = m_rasterize_ctx.fwd_output.image.data;
  } else {
    throw std::runtime_error("to_opencv expects float32 or float16 image data");
  }
  backend_check_throw(
      m_backend_runtime->copy_device_to_host_async(
          m_major_queue,
          cpu_image.data(),
          src_ptr,
          shape.padded_size() * sizeof(float)),
      "to_opencv copy_device_to_host_async");
  backend_check_throw(
      m_backend_runtime->synchronize_queue(m_major_queue),
      "to_opencv synchronize_queue");

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

void Orchestrator::set_render_resolution(const ImageShape& new_shape) {
  if (new_shape.width > m_max_render_shape.width ||
      new_shape.height > m_max_render_shape.height) {
    throw std::invalid_argument("Not a valid buffer shape.");
  }

  // Create new image objects with the reallocated buffers
  ImageShape rgb_shape{new_shape.width, new_shape.height, 3};
  Image render_rgb = Image(rgb_shape, m_active_data_type, buffer_data<float>(m_render_buffer));
  Image grad_rgb = Image(rgb_shape, m_active_data_type, buffer_data<float>(m_image_grad_buffer));

  // Update rasterization context with new dimensions and images
  m_rasterize_ctx.fwd_input.width = new_shape.width;
  m_rasterize_ctx.fwd_input.height = new_shape.height;

  // Update output images
  m_rasterize_ctx.fwd_output.image = render_rgb;
  m_rasterize_ctx.grad_output.image = grad_rgb;

  // Update loss context
  m_loss_ctx.loss = Image(rgb_shape, m_active_data_type, buffer_data<float>(m_loss_buffer));
  m_loss_ctx.pred = render_rgb;
  m_loss_ctx.grad = grad_rgb;
}

void Orchestrator::reorder_gaussians() {
  log_info("Reordering gaussians by Morton code for better spatial locality...");
  uint n = m_gaussians->size();

  auto idx_buffer = m_gaussians->compute_morton_order_indices(m_major_queue.get());
  if (!idx_buffer) return;

  uint* indices = buffer_data<uint>(idx_buffer);

  m_gaussians->reorder(indices, m_major_queue.get());
  m_gradients->reorder(indices, m_major_queue.get());
  m_optimizer->reorder(indices, m_major_queue);

  if (m_rasterize_ctx.densification_info) {
    m_rasterize_ctx.densification_info = reorder_densification_info(
        m_rasterize_ctx.densification_info,
        indices,
        n,
        m_backend_runtime,
        m_major_queue.get());
  }
}

size_t Orchestrator::group_accumulate_steps(OptimParamGroup group) const {
  switch (group) {
    case OptimParamGroup::Means:
      return m_config.means_accumulate_grad_steps;
    case OptimParamGroup::Shs:
      return m_config.shs_accumulate_grad_steps;
    case OptimParamGroup::Opacities:
      return m_config.opacities_accumulate_grad_steps;
    case OptimParamGroup::Scales:
      return m_config.scales_accumulate_grad_steps;
    case OptimParamGroup::Rotations:
      return m_config.rotations_accumulate_grad_steps;
  }
  return 1;
}

std::shared_ptr<LrSchedulerBase> Orchestrator::group_scheduler(OptimParamGroup group) const {
  switch (group) {
    case OptimParamGroup::Means:
      return m_means_lr_scheduler;
    case OptimParamGroup::Shs:
      return m_shs_lr_scheduler;
    case OptimParamGroup::Opacities:
      return m_opacities_lr_scheduler;
    case OptimParamGroup::Scales:
      return m_scales_lr_scheduler;
    case OptimParamGroup::Rotations:
      return m_rotations_lr_scheduler;
  }
  return nullptr;
}

bool Orchestrator::should_step_group(OptimParamGroup group, size_t step) const {
  const size_t k = group_accumulate_steps(group);
  return ((step + 1) % k) == 0;
}

}  // namespace tinygs
