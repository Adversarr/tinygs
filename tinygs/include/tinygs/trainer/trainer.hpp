#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/cuda/gpu_memory.hpp"
#include "tinygs/dataloader/dataloader.hpp"
#include "tinygs/loss/loss.hpp"
#include "tinygs/optim/optim.hpp"
#include "tinygs/rasterizer/rasterizer.hpp"
#include "tinygs/strategy/strategy.hpp"

namespace tinygs {

/**
 * @brief Configuration parameters for training
 */
struct TrainerConfig {
  // Training parameters
  size_t max_steps = 30000;
  float initial_learning_rate = 1.0f;
  float final_learning_rate = 0.01f;

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
};

/**
 * @brief Training state information
 */
struct TrainingState {
  size_t current_step = 0;
  float current_loss = 0.0f;
  float current_learning_rate = 0.0f;
  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point last_log_time;
  bool should_stop = false;
};

/**
 * @brief Callback function types for training events
 */
using PreStepCallback = std::function<void(const TrainingState&)>;
using PostStepCallback = std::function<void(const TrainingState&, float loss, const std::vector<float>& metrics)>;
using CheckpointCallback = std::function<void(const TrainingState&, std::shared_ptr<GPUGaussian3d>)>;

/**
 * @brief Main trainer class for 3D Gaussian Splatting
 *
 * This class encapsulates the training loop and provides a flexible interface
 * for training 3D Gaussian Splatting models with various optimizers, loss functions,
 * and densification strategies.
 */
class Trainer {
public:
  /**
   * @brief Construct a new Trainer object
   * @param config Training configuration parameters
   */
  explicit Trainer(const TrainerConfig& config = TrainerConfig{});

  ~Trainer() = default;

  // Core setup methods

  /**
   * @brief Set the gaussians data and gradients
   * @param gaussians Shared pointer to GPU gaussians
   * @param gradients Shared pointer to gaussians gradients
   */
  void set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gradients);

  /**
   * @brief Set the rasterizer for rendering
   * @param rasterizer Shared pointer to rasterizer
   */
  void set_rasterizer(std::shared_ptr<RasterizerBase> rasterizer);

  /**
   * @brief Set the data loader for training data
   * @param dataloader Shared pointer to data loader
   */
  void set_dataloader(std::shared_ptr<DataLoaderBase> dataloader);

  /**
   * @brief Set the optimizer for parameter updates
   * @param optimizer Shared pointer to optimizer
   */
  void set_optimizer(std::shared_ptr<OptimizerBase> optimizer);

  /**
   * @brief Set the densification strategy
   * @param strategy Shared pointer to densification strategy
   */
  void set_strategy(std::shared_ptr<StrategyBase> strategy);

  // Loss and metrics management

  /**
   * @brief Add a loss function with weight
   * @param loss Shared pointer to loss function
   * @param weight Weight for this loss component
   */
  void add_loss(std::shared_ptr<LossBase> loss, float weight = 1.0f);

  /**
   * @brief Add a metric for evaluation (no gradient computation)
   * @param metric Shared pointer to metric
   * @param name Name of the metric for logging
   */
  void add_metric(std::shared_ptr<MetricBase> metric, const std::string& name);

  // Callback registration

  /**
   * @brief Register callback to be called before each training step
   * @param callback Pre-step callback function
   */
  void set_pre_step_callback(PreStepCallback callback);

  /**
   * @brief Register callback to be called after each training step
   * @param callback Post-step callback function
   */
  void set_post_step_callback(PostStepCallback callback);

  /**
   * @brief Register callback to be called at checkpoint intervals
   * @param callback Checkpoint callback function
   */
  void set_checkpoint_callback(CheckpointCallback callback);

  // Training control

  /**
   * @brief Start the training loop
   * @return Final training state
   */
  TrainingState train();

  /**
   * @brief Execute a single training step
   */
  void step();

  /**
   * @brief Accumulate the loss
   * @return the value
   */
  float accumulate_loss();

  /**
   * @brief Stop training (can be called from callbacks)
   */
  void stop_training();

  /**
   * @brief Check if a stop has been requested
   * @return True if training should stop
   */
  bool is_stop_requested() const;

  /**
   * @brief Reset trainer state for new training session
   */
  void reset();

  // State access

  /**
   * @brief Get current training state
   * @return Current training state
   */
  const TrainingState& get_state() const { return m_state; }

  /**
   * @brief Get training configuration
   * @return Training configuration
   */
  const TrainerConfig& get_config() const { return m_config; }

  /**
   * @brief Update training configuration
   * @param config New configuration
   */
  void update_config(const TrainerConfig& config);

  /**
   * @brief Get the rasterize context for accessing rendered output
   * @return Reference to the rasterize context
   */
  const RasterizeContext& get_rasterize_context() const { return m_rasterize_ctx; }

  /**
   * @brief Get the loss context for accessing loss computation data
   * @return Reference to the loss context
   */
  const LossContext& get_loss_context() const { return m_loss_ctx; }

private:
  // Core training components
  std::shared_ptr<GPUGaussian3d> m_gaussians;
  std::shared_ptr<GPUGaussian3d> m_gradients;
  std::shared_ptr<RasterizerBase> m_rasterizer;
  std::shared_ptr<DataLoaderBase> m_dataloader;
  std::shared_ptr<OptimizerBase> m_optimizer;
  std::shared_ptr<StrategyBase> m_strategy;

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
  TrainerConfig m_config;
  TrainingState m_state;

  // Callbacks
  PreStepCallback m_pre_step_callback;
  PostStepCallback m_post_step_callback;
  CheckpointCallback m_checkpoint_callback;

  // Internal GPU memory management
  std::unique_ptr<GPUMemory<float>> m_loss_buffer;
  std::unique_ptr<GPUMemory<float>> m_render_buffer;
  std::unique_ptr<GPUMemory<float>> m_image_grad_buffer;
  RasterizeContext m_rasterize_ctx;
  LossContext m_loss_ctx;

  // Helper methods

  /**
   * @brief Initialize GPU memory buffers based on image dimensions
   */
  void initialize_buffers();

  /**
   * @brief Compute current learning rate based on step and schedule
   * @return Current learning rate
   */
  float compute_learning_rate() const;

  /**
   * @brief Update spherical harmonics degree based on training progress
   */
  void update_sh_degree();

  /**
   * @brief Evaluate all loss functions and accumulate gradients
   * @param data Current training data batch
   * @return Total loss value
   */
  float evaluate_losses(const GPUBatchInputOutput& data);

  /**
   * @brief Evaluate all metrics for logging
   * @return Vector of metric values
   */
  std::vector<float> evaluate_metrics();

  /**
   * @brief Check if early stopping criteria are met
   * @return True if training should stop early
   */
  bool should_early_stop() const;

  /**
   * @brief Validate that all required components are set
   * @throws std::runtime_error if any required component is missing
   */
  void validate_setup() const;
};

}  // namespace tinygs