# Orchestrator Module

The orchestrator is the central training coordinator that manages the entire training pipeline.

## File

| File | Description |
|------|-------------|
| `orchestrator.hpp` | Main trainer class |
| `orchestrator.cu` | Implementation |

---

## OrchestratorConfig

Training configuration parameters:

```cpp
struct OrchestratorConfig {
    // Training limits
    size_t max_steps = 30000;              // Maximum training steps
    size_t accumulate_grad_steps = 1;       // Gradient accumulation
    size_t max_seconds = 0;                 // Time limit (0 = disabled)
    
    // Logging
    size_t log_interval = 100;              // Logging frequency
    size_t checkpoint_interval = 1000;      // Checkpoint frequency
    
    // Spherical harmonics
    size_t sh_degree_interval = 1000;       // SH degree increase interval
    size_t max_sh_degree = 3;               // Maximum SH degree
    
    // Early stopping
    bool enable_early_stopping = false;
    float early_stopping_threshold = 1e-6f;
    size_t early_stopping_patience = 1000;
    
    // Rendering
    float near_plane = 0.01f;
    float far_plane = 100.0f;
    float grad_scaler = 1.0f;               // Gradient scaler (128 for fp16)
    
    // Evaluation
    std::vector<size_t> test_steps{7000, 30000};
    std::string out_dir;
    
    // Strategy
    size_t scene_scale_recompute_interval = 1000;
    size_t reorder_gaussians_interval = 1000;
    size_t start_pose_opt = 500;            // Start pose optimization step
    
    // Data types
    DataType train_data_type = DataType::Float32;
    DataType eval_data_type = DataType::Float32;
    
    // Export options
    bool export_rasterized = false;
    bool export_full_features = false;
    bool record_trajectory = false;
};
```

---

## TrainingState

Current training state:

```cpp
struct TrainingState {
    size_t current_step = 0;
    float current_loss = 0.0f;
    
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point last_log_time;
    bool should_stop = false;
};
```

---

## Orchestrator Class

```cpp
class Orchestrator {
public:
    explicit Orchestrator(const OrchestratorConfig& config = {});
    
    // Component setup
    void set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians,
                       std::shared_ptr<GPUGaussian3d> gradients);
    void set_rasterizer(std::shared_ptr<RasterizerBase> rasterizer);
    void set_dataloader(std::shared_ptr<DataLoaderBase> dataloader);
    void set_test_dataloader(std::shared_ptr<DataLoaderBase> dataloader);
    void set_optimizer(std::shared_ptr<OptimizerBase> optimizer);
    void set_pose_opt(std::shared_ptr<PoseOptBase> pose_opt);
    void set_strategy(std::shared_ptr<StrategyBase> strategy);
    void set_lr_scheduler(std::shared_ptr<LrSchedulerBase> scheduler);
    
    // Loss and metrics
    void add_loss(std::shared_ptr<LossBase> loss, float weight = 1.0f);
    void add_metric(std::shared_ptr<MetricBase> metric, const std::string& name);
    
    // Callbacks
    using PreStepCallback = std::function<void(const TrainingState&)>;
    using PostStepCallback = std::function<void(const TrainingState&)>;
    using CheckpointCallback = std::function<void(const TrainingState&)>;
    
    void set_pre_step_callback(PreStepCallback callback);
    void set_post_step_callback(PostStepCallback callback);
    void set_checkpoint_callback(CheckpointCallback callback);
    
    // Training control
    TrainingState train();
    void train_step();
    void test_step();
    std::unordered_map<std::string, float> eval(DataLoaderBase* loader = nullptr);
    void stop_training();
    bool is_stop_requested() const;
    void reset();
    
    // State access
    const TrainingState& get_state() const;
    const OrchestratorConfig& get_config() const;
    std::shared_ptr<GPUGaussian3d> get_gaussians() const;
    std::shared_ptr<OptimizerBase> get_optimizer() const;
    std::shared_ptr<LrSchedulerBase> get_lr_scheduler() const;
    
    // Configuration
    void set_params(const json& j);
    json get_params() const;
};
```

---

## Training Loop

The main training loop:

```cpp
TrainingState Orchestrator::train() {
    initialize();
    m_state.start_time = std::chrono::steady_clock::now();
    
    while (m_state.current_step < m_config.max_steps && !m_state.should_stop) {
        // Pre-step callback
        if (m_pre_step_callback) m_pre_step_callback(m_state);
        
        // Execute training step
        train_step();
        
        // Post-step callback
        if (m_post_step_callback) m_post_step_callback(m_state);
        
        // Logging
        if (m_state.current_step % m_config.log_interval == 0) {
            log_progress();
        }
        
        // Evaluation
        if (should_eval(m_state.current_step)) {
            test_step();
        }
        
        m_state.current_step++;
        
        // Time limit check
        if (m_config.max_seconds > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - m_state.start_time
            ).count();
            if (elapsed >= m_config.max_seconds) {
                m_state.should_stop = true;
            }
        }
    }
    
    return m_state;
}
```

---

## Single Training Step

```cpp
void Orchestrator::train_step() {
    // Get next batch
    auto batch = m_dataloader->next();
    
    // Apply pose optimization
    if (m_state.current_step >= m_config.start_pose_opt && m_pose_opt) {
        batch.input.w2c = m_pose_opt->query(
            batch.input.timestamp,
            batch.input.w2c
        );
    }
    
    // Setup rasterization context
    m_rasterize_ctx.fwd_input = batch.input;
    m_rasterize_ctx.stream = m_major_stream;
    m_rasterize_ctx.grad_scaler = m_config.grad_scaler;
    
    // Forward pass
    m_rasterizer->forward(m_rasterize_ctx);
    
    // Evaluate losses
    m_loss_ctx.pred = m_rasterize_ctx.fwd_output.image;
    m_loss_ctx.target = batch.output.image;
    m_loss_ctx.stream = m_major_stream;
    evaluate_losses(batch);
    
    // Backward pass
    m_rasterize_ctx.grad_output.image = m_loss_ctx.grad;
    m_rasterizer->backward(m_rasterize_ctx);
    
    // Update learning rate
    float lr = m_lr_scheduler->step();
    
    // Optimization step
    m_optimizer->step(lr, m_major_stream);
    
    // Update SH degree
    update_sh_degree();
    
    // Strategy step (densification)
    if (m_strategy) {
        m_strategy->step(m_rasterize_ctx);
    }
    
    // Update pose
    if (m_state.current_step >= m_config.start_pose_opt && m_pose_opt) {
        if (m_rasterize_ctx.grad_input.has_value()) {
            m_pose_opt->update(
                batch.input.timestamp,
                m_rasterize_ctx.grad_input->w2c,
                lr
            );
        }
    }
}
```

---

## Resolution Configuration

Resolution is now dataset-owned. Configure `resolution` and `resolution_scale` under
`dataset` / `test_dataset`; orchestrator uses the loaded dataset shapes directly.

---

## Spherical Harmonics Progression

Gradually increase SH degree during training:

```cpp
void Orchestrator::update_sh_degree() {
    int new_degree = std::min<int>(
        m_state.current_step / m_config.sh_degree_interval,
        m_config.max_sh_degree
    );
    
    if (new_degree != m_gaussians->get_sh_degree()) {
        m_gaussians->set_sh_degree(new_degree);
    }
}
```

**Progression:**
- Step 0-999: Degree 0 (DC only)
- Step 1000-1999: Degree 1
- Step 2000-2999: Degree 2
- Step 3000+: Degree 3

---

## Loss Evaluation

```cpp
void Orchestrator::evaluate_losses(const GPUBatchInputOutput& data) {
    // Zero gradients
    cudaMemsetAsync(m_loss_ctx.grad.data, 0, ...);
    
    // Accumulate weighted losses
    for (const auto& [loss, weight] : m_losses) {
        loss->evaluate(m_loss_ctx, weight);
    }
}
```

---

## Evaluation

```cpp
std::unordered_map<std::string, float> Orchestrator::eval(DataLoaderBase* loader) {
    if (!loader) loader = m_test_dataloader.get();
    
    loader->reset();
    std::unordered_map<std::string, float> results;
    
    for (size_t i = 0; i < loader->get_dataset()->size(); ++i) {
        auto batch = loader->next();
        
        // Render
        m_rasterize_ctx.fwd_input = batch.input;
        m_rasterize_ctx.inference = true;
        m_rasterizer->forward(m_rasterize_ctx);
        
        // Evaluate metrics
        for (const auto& [metric, name] : m_metrics) {
            float value = metric->evaluate(
                m_rasterize_ctx.fwd_output.image,
                batch.output.image
            );
            results[name] += value;
        }
    }
    
    // Average
    for (auto& [name, value] : results) {
        value /= loader->get_dataset()->size();
    }
    
    return results;
}
```

---

## Usage Example

```cpp
// Create components
auto dataset = create_dataset("image");
dataset->set_params({
    {"root_path", "outputs/scene/train/"},
    {"extension", "png"},
    {"resolution", -1},
    {"resolution_scale", 1.0f}
});
dataset->load();

auto dataloader = create_dataloader("async", dataset);
auto initializer = create_initialization("knn");
auto rasterizer = create_rasterizer("fastgs");
auto optimizer = create_optimizer("adam", gaussians, gradients);
auto scheduler = create_lr_scheduler("exponential", optimizer);
auto strategy = create_strategy("mcmc", gaussians, gradients, optimizer);
auto pose_opt = create_pose_opt("adamw");

// Initialize Gaussians
auto pc_opt = dataset->get_point_cloud();
if (!pc_opt.has_value()) {
    throw std::runtime_error("Dataset does not provide points3d.ply");
}
initializer->initialize(pc_opt.value());
gaussians->copy_from_host(initializer->gaussians());

// Setup orchestrator
OrchestratorConfig orch_config;
orch_config.from_json(config["trainer"]);

Orchestrator orchestrator(orch_config);
orchestrator.set_gaussians(gaussians, gradients);
orchestrator.set_rasterizer(rasterizer);
orchestrator.set_dataloader(dataloader);
orchestrator.set_optimizer(optimizer);
orchestrator.set_lr_scheduler(scheduler);
orchestrator.set_strategy(strategy);
orchestrator.set_pose_opt(pose_opt);
orchestrator.add_loss(create_loss("l1"), 0.8f);
orchestrator.add_loss(create_loss("fused_ssim"), 0.2f);
orchestrator.add_metric(create_metric("psnr"), "psnr");

// Train
auto state = orchestrator.train();
log_info("Training completed at step {}", state.current_step);

// Evaluate
auto metrics = orchestrator.eval();
log_info("Final PSNR: {:.2f} dB", metrics["psnr"]);
```

---

## Callbacks

Use callbacks for custom behavior:

```cpp
orchestrator.set_post_step_callback([](const TrainingState& state) {
    if (state.current_step % 1000 == 0) {
        save_checkpoint(state.current_step);
    }
});

orchestrator.set_checkpoint_callback([](const TrainingState& state) {
    log_info("Checkpoint at step {}", state.current_step);
});
```

---

## Logging Output

```
STEP 100] PSNR=18.234 | TPUT=1234ms/100step | LR=1.0e+0 | N-Gs: 15234 | T= 12s
STEP 200] PSNR=20.156 | TPUT=1180ms/100step | LR=9.8e-1 | N-Gs: 18456 | T= 24s
STEP 300] PSNR=22.891 | TPUT=1156ms/100step | LR=9.6e-1 | N-Gs: 22341 | T= 36s
...
```