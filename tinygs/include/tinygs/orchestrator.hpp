#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>
#include <unordered_map>

#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/dataloader/dataloader.hpp"
#include "tinygs/loss/loss.hpp"
#include "tinygs/optim/lr_scheduler.hpp"
#include "tinygs/optim/optim.hpp"
#include "tinygs/pose_opt/pose_opt.hpp"
#include "tinygs/rasterizer/rasterizer.hpp"
#include "tinygs/strategy/strategy.hpp"
#include "tinygs/platform/backend_types.hpp"
#include "tinygs/platform/runtime.hpp"

namespace tinygs {

/// @brief Configuration parameters for training
struct OrchestratorConfig {
  // Training parameters
  size_t max_steps = 30000;
  // Deprecated: ignored at runtime, kept only for backward config compatibility.
  size_t accumulate_grad_steps = 1;
  size_t means_accumulate_grad_steps = 1;
  size_t shs_accumulate_grad_steps = 1;
  size_t opacities_accumulate_grad_steps = 1;
  size_t scales_accumulate_grad_steps = 1;
  size_t rotations_accumulate_grad_steps = 1;
  // Time-based stopping (0 disables)
  size_t max_seconds = 0;

  // Logging and visualization
  size_t log_interval = 100;
  size_t checkpoint_interval = 1000;

  // Spherical harmonics progression
  size_t sh_degree_interval = 1000;
  size_t max_sh_degree = 3;

  // Early stopping
  bool enable_early_stopping = false;
  float early_stopping_threshold = 1e-6f;
  size_t early_stopping_patience = 1000;

  // Rasterization parameters
  float near_plane = 0.01f;
  float far_plane = 100.0f;

  // Validations
  std::vector<size_t> test_steps{7'000, 30'000};
  std::string out_dir;
  bool export_rasterized = false;
  bool export_full_features = false;
  bool record_trajectory = false;

  float grad_scaler = 1.0f;

  // Strategy parameters
  size_t scene_scale_recompute_interval = 1000;     ///< Interval for recomputing scene scale in strategy steps
  size_t reorder_gaussians_interval = 1000;         ///< Interval for reordering gaussians in strategy steps
  size_t start_pose_opt = 500;                     ///< Step to start pose optimization

  // Rasterizer
  DataType train_data_type = DataType::Float32;
  DataType eval_data_type = DataType::Float32;

  // CUDA debugging
  bool debug_cuda_check_each_stage = false;  ///< Check CUDA error state after each train_step stage
  bool debug_cuda_sync_each_stage = false;   ///< Synchronize stream after each train_step stage
  size_t debug_cuda_check_every = 0;         ///< Check cadence in steps (0 = disabled)
  bool debug_cuda_log_each_stage = false;    ///< Log stage names when debug checks run

  /// @brief Convert config to JSON
  json to_json() const;
  /// @brief Load config from JSON
  void from_json(const json& j);
};

/// @brief Training state information
struct TrainingState {
  size_t current_step = 0;
  float current_loss = 0.0f;

  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point last_log_time;
  bool should_stop = false;
};

/// @brief Callback function types for training events
using PreStepCallback = std::function<void(const TrainingState&)>;
using PostStepCallback = std::function<void(const TrainingState&)>;
using CheckpointCallback = std::function<void(const TrainingState&)>;

/// @brief Main trainer class for 3D Gaussian Splatting
class Orchestrator {
public:
  /// @brief Construct a new Trainer object
  /// @param config Training configuration parameters
  explicit Orchestrator(const OrchestratorConfig& config = OrchestratorConfig{});

  ~Orchestrator() = default;

  ////////////////////////////// Core setup methods //////////////////////////////

  /// @brief Set the gaussians data and gradients
  void set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gradients);

  /// @brief Set the rasterizer for rendering
  void set_rasterizer(std::shared_ptr<RasterizerBase> rasterizer);

  /// @brief Set the data loader for training data
  void set_dataloader(std::shared_ptr<DataLoaderBase> dataloader);

  void set_test_dataloader(std::shared_ptr<DataLoaderBase> dataloader);
  std::shared_ptr<DataLoaderBase> get_test_dataloader() const;

  /// @brief Set the optimizer for parameter updates
  void set_optimizer(std::shared_ptr<OptimizerBase> optimizer);

  /// @brief Set the pose optimizer for camera pose updates
  void set_pose_opt(std::shared_ptr<PoseOptBase> pose_opt);

  /// @brief Set the densification strategy
  void set_strategy(std::shared_ptr<StrategyBase> strategy);

  /// @brief Set backend runtime used for queue/event/buffer operations.
  void set_backend_runtime(std::shared_ptr<BackendRuntime> backend_runtime);

  /// @brief Set the learning rate scheduler
  void set_lr_scheduler(OptimParamGroup group, std::shared_ptr<LrSchedulerBase> scheduler);

  /// @brief Get the current learning rate scheduler for a parameter group
  std::shared_ptr<LrSchedulerBase> get_lr_scheduler(OptimParamGroup group) const;

  /// @brief Get the current optimizer
  std::shared_ptr<OptimizerBase> get_optimizer() const;

  // Loss and metrics management

  /// @brief Add a loss function with weight
  void add_loss(std::shared_ptr<LossBase> loss, float weight = 1.0f);

  /// @brief Add a metric for evaluation
  void add_metric(std::shared_ptr<MetricBase> metric, const std::string& name);

  // Callback registration

  /// @brief Register pre-step callback
  void set_pre_step_callback(PreStepCallback callback);

  /// @brief Register post-step callback
  void set_post_step_callback(PostStepCallback callback);

  /// @brief Register checkpoint callback
  void set_checkpoint_callback(CheckpointCallback callback);

  // Training control

  /// @brief Start the training loop
  TrainingState train();

  /// @brief Execute a single training step
  void train_step();

  /// @brief Execute a test step
  void test_step();

  /// @brief Run full evaluation over dataset
  std::unordered_map<std::string, float> eval(DataLoaderBase* loader = nullptr);

  /// @brief Accumulate the loss in current loss buffer.
  float accumulate_loss();

  /// @brief Stop training
  void stop_training();

  /// @brief Check if a stop has been requested
  bool is_stop_requested() const;

  /// @brief Reset trainer state for new training session
  void reset();

  // State access

  /// @brief Get current training state
  const TrainingState& get_state() const { return m_state; }

  /// @brief Get training configuration
  const OrchestratorConfig& get_config() const { return m_config; }

  /// @brief Update training configuration
  void update_config(const OrchestratorConfig& config);

  /// @brief Get the rasterize context
  const RasterizeContext& get_rasterize_context() const { return m_rasterize_ctx; }

  /// @brief Get the loss context
  const LossContext& get_loss_context() const { return m_loss_ctx; }

  /// @brief Evaluate all metrics
  std::vector<float> evaluate_metrics();

  /// @brief Set parameters from JSON
  void set_params(const json& j);
  /// @brief Get parameters as JSON
  json get_params() const;

  /// @brief convert current rasterizer result to opencv mat
  cv::Mat to_opencv() const;

  /// @brief Get the current Gaussian splatting model
  std::shared_ptr<GPUGaussian3d> get_gaussians() const { return m_gaussians; }

  /// @brief Get the current Gaussian gradients
  std::shared_ptr<GPUGaussian3d> get_gradients() const { return m_gradients; }

private:
  // Core training components
  std::shared_ptr<GPUGaussian3d> m_gaussians;
  std::shared_ptr<GPUGaussian3d> m_gradients;
  std::shared_ptr<RasterizerBase> m_rasterizer;
  std::shared_ptr<DataLoaderBase> m_dataloader;
  std::shared_ptr<DataLoaderBase> m_test_dataloader;
  std::shared_ptr<OptimizerBase> m_optimizer;
  std::shared_ptr<StrategyBase> m_strategy;
  std::shared_ptr<PoseOptBase> m_pose_opt;
  std::shared_ptr<LrSchedulerBase> m_means_lr_scheduler;
  std::shared_ptr<LrSchedulerBase> m_shs_lr_scheduler;
  std::shared_ptr<LrSchedulerBase> m_opacities_lr_scheduler;
  std::shared_ptr<LrSchedulerBase> m_scales_lr_scheduler;
  std::shared_ptr<LrSchedulerBase> m_rotations_lr_scheduler;

  // Loss functions and metrics
  struct LossComponent {
    std::shared_ptr<LossBase> loss;
    float weight;
  };
  std::vector<LossComponent> m_losses;

  struct MetricComponent {
    std::shared_ptr<MetricBase> metric;
    std::string name;
  };
  std::vector<MetricComponent> m_metrics;

  // Training state and configuration
  OrchestratorConfig m_config;
  TrainingState m_state;

  // Backend runtime and major queue for orchestration-stage synchronization.
  std::shared_ptr<BackendRuntime> m_backend_runtime;
  std::shared_ptr<BackendQueue> m_major_queue;

  // Callbacks
  PreStepCallback m_pre_step_callback;
  PostStepCallback m_post_step_callback;
  CheckpointCallback m_checkpoint_callback;

  // Internal GPU memory management (backend-agnostic)
  std::shared_ptr<BackendBuffer> m_loss_buffer;
  std::shared_ptr<BackendBuffer> m_render_buffer;
  std::shared_ptr<BackendBuffer> m_image_grad_buffer;
  RasterizeContext m_rasterize_ctx;
  LossContext m_loss_ctx;

  // Backend stream for training, do not block.
  BackendStream m_major_stream = nullptr;

  // Active render/loss data type used for buffers
  DataType m_active_data_type = DataType::Float32;

  // Maximum allocated render shape used as a guard for render/loss buffers.
  ImageShape m_max_render_shape{};

  // Early stopping state: track best loss and the step it was achieved
  float m_best_loss = -1.0f;        ///< Best (lowest) loss seen so far, -1 means unset
  size_t m_best_loss_step = 0;      ///< Step at which m_best_loss was recorded

  ////////////////////////////// Helper methods //////////////////////////////

  /// @brief Initialize GPU memory buffers
  void initialize();

  /// @brief Compute current learning rate
  float compute_learning_rate() const;

  /// @brief Update spherical harmonics degree
  void update_sh_degree();

  /// @brief Evaluate all loss functions and accumulate gradients
  /// @param data Current training data batch
  void evaluate_losses(const GPUBatchInputOutput& data);

  /// @brief Check if early stopping criteria are met
  bool should_early_stop() const;

  /// @brief Validate that all required components are set
  void validate_setup() const;

  /// @brief Recompute the scene scale
  void recompute_scene_scale();

  /// @brief Reorder gaussians to encourage spatial-storage continuity (Morton)
  void reorder_gaussians();

  /// @brief Reallocate GPU buffers and update contexts for a new render resolution.
  /// @param new_shape New image shape (must not exceed full dataset resolution)
  void set_render_resolution(const ImageShape& new_shape);

  size_t group_accumulate_steps(OptimParamGroup group) const;
  std::shared_ptr<LrSchedulerBase> group_scheduler(OptimParamGroup group) const;
  bool should_step_group(OptimParamGroup group, size_t step) const;
};

}  // namespace tinygs
