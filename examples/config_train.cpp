#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <cxxopts.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "tinygs/core/pointcloud.hpp"
#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/initialization/initialization.hpp"
#include "tinygs/optim/lr_scheduler.hpp"
#include "tinygs/optim/optim.hpp"
#include "tinygs/rasterizer/rasterizer.hpp"
#include "tinygs/orchestrator.hpp"
#include "tinygs/dataloader/dataloader.hpp"
using namespace tinygs;

std::shared_ptr<Orchestrator> build(const std::string& config_path);

void train(std::shared_ptr<Orchestrator> orchestrator, bool visualize = false);

int main(int argc, char** argv) {
  auto opts = cxxopts::Options("config_train", "Train model with config file.");

  opts.add_options()
    ("h,help", "Print help")
    ("v,visualize", "Visualize training process", cxxopts::value<bool>()->default_value("false"))
    ("c,config", "Config file path", cxxopts::value<std::string>());

  auto result = opts.parse(argc, argv);

  if (result.count("help")) {
    std::cout << opts.help() << std::endl;
    return 0;
  }

  auto visualize = result["visualize"].as<bool>();

  if (result.count("config")) {
    auto orchestrator = build(result["config"].as<std::string>());
    train(orchestrator, visualize);
  } else {
    std::cout << "Config file path is required." << std::endl;
    std::cout << opts.help() << std::endl;
    return 0;
  }
  return 0;
}

std::shared_ptr<Orchestrator> build(const std::string& config_path) {
  nlohmann::json config;
  {
    std::ifstream config_file(config_path);
    config_file >> config;
    if (!config_file) {
      throw std::runtime_error("Failed to read config file.");
    } else if (!config.is_object()) {
      throw std::runtime_error("Config file must be a JSON object.");
    }
  }

  // orchestrator
  auto orchestrator = std::make_shared<Orchestrator>();
  if (auto trainer_config = config.at("trainer"); trainer_config.is_object()) {
    orchestrator->set_params(trainer_config);
  }

  // Initial points
  auto pointcloud_filepath = config.at("input_pc_file").get<std::string>();
  auto pointcloud = load_point_cloud(pointcloud_filepath);

  // dataset
  std::shared_ptr<DatasetBase> dataset;
  if (auto dataset_config = config.at("dataset"); dataset_config.is_object()) {
    dataset = create_dataset(dataset_config.at("type").get<std::string>());
    dataset->set_params(dataset_config);
    dataset->load();
  } else {
    throw std::runtime_error("Dataset config is required.");
  }

  // dataloader
  std::shared_ptr<DataLoaderBase> dataloader;
  if (auto dataloader_config = config.at("dataloader"); dataloader_config.is_object()) {
    dataloader = create_dataloader(dataloader_config.at("type").get<std::string>(), dataset);
    dataloader->set_params(dataloader_config);
  } else {
    throw std::runtime_error("Dataloader config is required.");
  }
  orchestrator->set_dataloader(dataloader);
  log_info("All loaders done.");

  // initializer
  std::shared_ptr<InitializationBase> initializer;
  if (auto initializer_config = config.at("initializer"); initializer_config.is_object()) {
    initializer = create_initialization(initializer_config.at("type").get<std::string>());
    initializer->set_params(initializer_config);
  } else {
    throw std::runtime_error("Initializer config is required.");
  }

  // Init the point clouds.
  initializer->initialize(pointcloud);
  const auto& init_result = initializer->gaussians();
  auto gs3d = std::make_shared<GPUGaussian3d>();
  gs3d->copy_from_host(init_result);
  std::shared_ptr<GPUGaussian3d> grads = gs3d->clone();
  grads->memset(0);
  gs3d->set_sh_degree(0);
  orchestrator->set_gaussians(gs3d, grads);

  // rasterizer
  std::shared_ptr<RasterizerBase> rasterizer;
  if (auto rasterizer_config = config.at("rasterizer"); rasterizer_config.is_object()) {
    rasterizer = create_rasterizer(rasterizer_config.at("type").get<std::string>());
    rasterizer->set_params(rasterizer_config);
    orchestrator->set_rasterizer(rasterizer);
  } else {
    throw std::runtime_error("Rasterizer config is required.");
  }

  // optimizer
  std::shared_ptr<OptimizerBase> optimizer;
  if (auto optimizer_config = config.at("optimizer"); optimizer_config.is_object()) {
    optimizer = create_optimizer(optimizer_config.at("type").get<std::string>(), gs3d, grads);
    optimizer->set_params(optimizer_config);
    orchestrator->set_optimizer(optimizer);
  } else {
    throw std::runtime_error("Optimizer config is required.");
  }

  // lr_scheduler
  std::shared_ptr<LrSchedulerBase> lr_scheduler;
  if (auto lr_scheduler_config = config.at("lr_scheduler"); lr_scheduler_config.is_object()) {
    lr_scheduler = create_lr_scheduler(lr_scheduler_config.at("type").get<std::string>(), optimizer);
    lr_scheduler->set_params(lr_scheduler_config);
    orchestrator->set_lr_scheduler(lr_scheduler);
  } else {
    throw std::runtime_error("LR scheduler config is required.");
  }

  // Losses
  if (auto losses_config = config.at("losses"); losses_config.is_array()) {
    for (const auto& loss_config : losses_config) {
      if (loss_config.is_object()) {
        orchestrator->add_loss(create_loss(loss_config.at("type").get<std::string>()),
                               loss_config.at("weight").get<float>());
      } else {
        throw std::runtime_error("Loss item must be an object.");
      }
    }
  } else {
    throw std::runtime_error("Losses config is required.");
  }

  // metrics
  if (auto metrics_config = config.at("metrics"); metrics_config.is_array()) {
    for (const auto& metric_config : metrics_config) {
      if (metric_config.is_string()) {
        auto name = metric_config.get<std::string>();
        orchestrator->add_metric(create_metric(name), name);
      } else {
        throw std::runtime_error("Metric item must be an string.");
      }
    }
  } else {
    throw std::runtime_error("Metrics config is required.");
  }

  // strategy
  std::shared_ptr<StrategyBase> strategy;
  if (auto strategy_config = config.at("strategy"); strategy_config.is_object()) {
    strategy = create_strategy(strategy_config.at("type").get<std::string>(), gs3d, grads, optimizer);
    strategy->set_params(strategy_config);
    orchestrator->set_strategy(strategy);
  } else {
    throw std::runtime_error("Strategy config is required.");
  }

  return orchestrator;
}

void train(std::shared_ptr<Orchestrator> orchestrator, bool visualize) {
  auto gs3d = orchestrator->get_optimizer()->get_gaussians();
  auto grads = orchestrator->get_optimizer()->get_gaussians_grad();
  
  orchestrator->set_post_step_callback([orchestrator, gs3d, grads, visualize] (const TrainingState& state) {
    if (state.current_step % 100 != 0) {
      return;
    }
    
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - state.last_log_time);
    auto loss = orchestrator->accumulate_loss();
    auto metrics = orchestrator->evaluate_metrics();
    float psnr = metrics.empty() ? 0.0f : metrics[0];
    auto lr = orchestrator->get_optimizer()->get_lr();

    log_info("step {} loss: {:.3e} psnr: {:.3f} time: {:.1f}ms/100step current_lr: {:.3e}", state.current_step, loss,
             psnr, duration.count() / (state.current_step / 100.0), lr);

    // Visualize RGB - copy rendered image from trainer's internal buffers
    if (visualize) {
      cv::Mat img = orchestrator->to_opencv();
      if (!img.empty()) {
        cv::imshow("render", img);

        if (char key = cv::waitKey(3); key == 27) {
          orchestrator->stop_training();
          std::cout << "ESC pressed - stopping training..." << std::endl;
        }
      }
    }
  });

  auto final_state = orchestrator->train();
  
  log_info("Training completed after {} steps", final_state.current_step);
  if (visualize) {
    cv::waitKey(0);
    cv::destroyAllWindows();
  }
}