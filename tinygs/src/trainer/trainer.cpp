#include "tinygs/trainer/trainer.hpp"

#include <stdexcept>
#include <spdlog/spdlog.h>

#include "tinygs/cuda/reduce.hpp"
#include "tinygs/cuda/gpu_memory.hpp"

namespace tinygs {

Trainer::Trainer(const TrainerConfig& config) : m_config(config) {
  m_state.start_time = std::chrono::steady_clock::now();
  m_state.last_log_time = m_state.start_time;
}

void Trainer::set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians, 
                           std::shared_ptr<GPUGaussian3d> gradients) {
  m_gaussians = gaussians;
  m_gradients = gradients;
}

void Trainer::set_rasterizer(std::shared_ptr<RasterizerBase> rasterizer) {
  m_rasterizer = rasterizer;
  if (m_gaussians) {
    m_rasterizer->set_gaussians(m_gaussians);
  }
}

void Trainer::set_dataloader(std::shared_ptr<DataLoaderBase> dataloader) {
  m_dataloader = dataloader;
}

void Trainer::set_optimizer(std::shared_ptr<OptimizerBase> optimizer) {
  m_optimizer = optimizer;
}

void Trainer::set_strategy(std::shared_ptr<StrategyBase> strategy) {
  m_strategy = strategy;
}

void Trainer::add_loss(std::shared_ptr<LossBase> loss, float weight) {
  m_losses.push_back({loss, weight});
}

void Trainer::add_metric(std::shared_ptr<MetricBase> metric, const std::string& name) {
  m_metrics.push_back({metric, name});
}

void Trainer::set_pre_step_callback(PreStepCallback callback) {
  m_pre_step_callback = callback;
}

void Trainer::set_post_step_callback(PostStepCallback callback) {
  m_post_step_callback = callback;
}

void Trainer::set_checkpoint_callback(CheckpointCallback callback) {
  m_checkpoint_callback = callback;
}

TrainingState Trainer::train() {
  validate_setup();
  initialize_buffers();

  m_state.start_time = std::chrono::steady_clock::now();
  m_state.last_log_time = m_state.start_time;
  m_state.should_stop = false;
  
  while (!m_state.should_stop && m_state.current_step < m_config.max_steps) {
    step();
    
    // Check for early stopping
    if (m_config.enable_early_stopping && should_early_stop()) {
      spdlog::info("Early stopping triggered at step {}", m_state.current_step);
      break;
    }
  }
  
  return m_state;
}

void Trainer::step() {
  // Pre-step callback
  if (m_pre_step_callback) {
    m_pre_step_callback(m_state);
  }
  
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
  
  // Create alpha gradient image
  ImageShape shape = m_rasterize_ctx.fwd_output.image.shape;
  m_rasterize_ctx.grad_output.alpha = Image(
    {shape.width, shape.height, 1}, 
    ImageFormat::CHW, 
    ImageDataType::Float32,
    m_loss_buffer->data() + shape.width * shape.height * 3
  );
  
  // Backward pass
  m_rasterizer->backward(m_rasterize_ctx);
  
  // Update learning rate
  m_state.current_learning_rate = compute_learning_rate();
  
  // Optimizer step
  m_optimizer->step(m_state.current_learning_rate);
  
  // Strategy step (densification)
  if (m_strategy) {
    m_strategy->step(m_rasterize_ctx);
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

float Trainer::accumulate_loss() {
  if (!m_loss_buffer) {
    return 0.0f;
  }

  ImageShape shape = m_rasterize_ctx.fwd_output.image.shape;
  return gpu_sum(m_loss_buffer->data(), shape.width * shape.height * 3);
}

void Trainer::stop_training() {
  m_state.should_stop = true;
}

void Trainer::reset() {
  m_state.current_step = 0;
  m_state.current_loss = 0.0f;
  m_state.current_learning_rate = m_config.initial_learning_rate;
  m_state.should_stop = false;
  m_state.start_time = std::chrono::steady_clock::now();
  m_state.last_log_time = m_state.start_time;
  
  if (m_dataloader) {
    m_dataloader->reset();
  }
  
  if (m_strategy) {
    m_strategy->reset();
  }
}

void Trainer::update_config(const TrainerConfig& config) {
  m_config = config;
}

void Trainer::initialize_buffers() {
  m_dataloader->reset();
  // Get image dimensions from the first data sample
  auto shape = m_dataloader->get_dataset()->image_shape();

  uint32_t width = shape.width;
  uint32_t height = shape.height;
  
  // Initialize GPU memory buffers
  m_loss_buffer = std::make_unique<GPUMemory<float>>(width * height * 4);
  m_render_buffer = std::make_unique<GPUMemory<float>>(width * height * 4);
  m_image_grad_buffer = std::make_unique<GPUMemory<float>>(width * height * 4);

  ImageShape rgb_shape{width, height, 3};
  ImageShape alpha_shape{width, height, 1};
  Image render_rgb = Image(rgb_shape, ImageFormat::CHW, ImageDataType::Float32, m_render_buffer->data());
  Image render_alpha = Image(alpha_shape, ImageFormat::CHW, ImageDataType::Float32, m_render_buffer->data() + width * height * 3);
  Image grad_rgb = Image(rgb_shape, ImageFormat::CHW, ImageDataType::Float32, m_image_grad_buffer->data());
  Image grad_alpha = Image(alpha_shape, ImageFormat::CHW, ImageDataType::Float32, m_image_grad_buffer->data() + width * height * 3);

  // Setup rasterization context
  m_rasterize_ctx.inference = false; // Training mode
  m_rasterize_ctx.fwd_input.width = width;
  m_rasterize_ctx.fwd_input.height = height;
  // TODO: near and far should be configurable.
  m_rasterize_ctx.fwd_input.near = 0.01f;
  m_rasterize_ctx.fwd_input.far = 100.0f;
  
  // Setup output images
  m_rasterize_ctx.fwd_output.image = render_rgb;
  m_rasterize_ctx.fwd_output.alpha = render_alpha;
  // ... gradient to output image
  m_rasterize_ctx.grad_output.image = grad_rgb;
  m_rasterize_ctx.grad_output.alpha = grad_alpha;
  // ... gradient to gaussian parameters
  m_rasterize_ctx.gaussians_grad = m_gradients;

  // Setup loss context
  m_loss_ctx.loss = Image(shape, ImageFormat::CHW, ImageDataType::Float32, m_loss_buffer->data());
  // TODO: alpha is ignored for now
  m_loss_ctx.pred = render_rgb;
  m_loss_ctx.grad = grad_rgb;
  log_info("Setup trainer buffers with image shape: {}", to_string(shape));
}

float Trainer::compute_learning_rate() const {
  float progress = static_cast<float>(m_state.current_step) / static_cast<float>(m_config.max_steps);
  return std::pow(m_config.final_learning_rate / m_config.initial_learning_rate, progress) * m_config.initial_learning_rate;
}

void Trainer::update_sh_degree() {
  if (m_state.current_step % m_config.sh_degree_interval == 0) {
    size_t new_degree = std::min(
      m_state.current_step / m_config.sh_degree_interval,
      m_config.max_sh_degree
    );
    m_gaussians->set_sh_degree(static_cast<int>(new_degree));
  }
}

void Trainer::evaluate_losses(const GPUBatchInputOutput& data) {
  m_loss_ctx.target = data.output.image;
  m_loss_ctx.pred = m_rasterize_ctx.fwd_output.image;
  for (const auto& loss_component : m_losses) {
    m_loss_ctx.scale = loss_component.weight;
    loss_component.loss->evaluate(m_loss_ctx);
  }
}

std::vector<float> Trainer::evaluate_metrics() {
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

bool Trainer::should_early_stop() const {
  // Simple early stopping based on loss threshold
  // More sophisticated implementations could track loss history
  return m_state.current_loss < m_config.early_stopping_threshold;
}

void Trainer::validate_setup() const {
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
}

}  // namespace tinygs