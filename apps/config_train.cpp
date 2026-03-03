#include <chrono>
#include <cxxopts.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "tinygs/core/pointcloud.hpp"
#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/dataset/dataset.hpp"
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
    ("d,debug", "Enable debug mode", cxxopts::value<bool>()->default_value("false"))
    ("c,config", "Config file path", cxxopts::value<std::string>())
    ("l,log_level", "Verbose level of logger (debug, info, warn, error)", cxxopts::value<std::string>()->default_value("warn"));

  auto result = opts.parse(argc, argv);
  if(result["debug"].as<bool>()) {
    spdlog::set_level(spdlog::level::debug);
  } else if (result["log_level"].as<std::string>() == "info") {
    spdlog::set_level(spdlog::level::info);
  } else if (result["log_level"].as<std::string>() == "warn") {
    spdlog::set_level(spdlog::level::warn);
  } else {
    spdlog::set_level(spdlog::level::err);
  }

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
  if (config.contains("trainer") && config.at("trainer").is_object()) {
    const auto& trainer_cfg = config.at("trainer");
    if (trainer_cfg.contains("resolution") || trainer_cfg.contains("resolution_scale")) {
      throw std::runtime_error(
          "trainer.resolution and trainer.resolution_scale are no longer supported. "
          "Please move them to dataset/test_dataset sections.");
    }
    orchestrator->set_params(config.at("trainer"));
  }

  // dataset
  std::shared_ptr<DatasetBase> train_dataset;
  if (auto dataset_config = config.at("dataset"); dataset_config.is_object()) {
    train_dataset = create_dataset(dataset_config.at("type").get<std::string>());
    train_dataset->set_params(dataset_config);
    train_dataset->load();
  } else {
    throw std::runtime_error("Dataset config is required.");
  }

  // Initial point cloud from dataset
  auto pointcloud_opt = train_dataset->get_point_cloud();
  if (!pointcloud_opt.has_value()) {
    throw std::runtime_error("Dataset does not provide an initial point cloud "
                             "(missing points3d.ply in dataset root).");
  }
  auto pointcloud = std::move(pointcloud_opt.value());

  // dataloader
  std::shared_ptr<DataLoaderBase> dataloader;
  if (auto dataloader_config = config.at("dataloader"); dataloader_config.is_object()) {
    dataloader = create_dataloader(dataloader_config.at("type").get<std::string>(), train_dataset);
    dataloader->set_params(dataloader_config);
  } else {
    throw std::runtime_error("Dataloader config is required.");
  }
  orchestrator->set_dataloader(dataloader);
  log_info("Training samples: {}", train_dataset->size());

  if (config.contains("test_dataset") && config.at("test_dataset").is_object()) {
    const auto &test_dataset_cfg = config.at("test_dataset");
    std::shared_ptr<DatasetBase> test_dataset = create_dataset(test_dataset_cfg.at("type").get<std::string>());
    test_dataset->set_params(test_dataset_cfg);
    test_dataset->load();

    std::shared_ptr<DataLoaderBase> test_loader;
    if (config.contains("test_dataloader") && config.at("test_dataloader").is_object()) {
      const auto &test_dataloader_cfg = config.at("test_dataloader");
      test_loader = create_dataloader(test_dataloader_cfg.at("type").get<std::string>(), test_dataset);
      test_loader->set_params(test_dataloader_cfg);
    } else {
      const auto &train_dataloader_cfg = config.at("dataloader");
      test_loader = create_dataloader(train_dataloader_cfg.at("type").get<std::string>(), test_dataset);
      test_loader->set_params(train_dataloader_cfg);
    }
    orchestrator->set_test_dataloader(test_loader);
    log_info("Test dataset configured: {} samples.", test_dataset->size());
  } else {
    log_info("No test dataset configured. Training dataset will be used for evaluation.");
  }

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

  std::shared_ptr<PoseOptBase> pose_opt;
  if (config.contains("pose_opt") && config.at("pose_opt").is_object()) {
    const auto& pose_opt_config = config.at("pose_opt");
    pose_opt = create_pose_opt(pose_opt_config.at("type").get<std::string>());
    pose_opt->set_params(pose_opt_config);
    orchestrator->set_pose_opt(pose_opt);
  }

  auto maybe_set_scheduler = [&](OptimParamGroup group, const json& scheduler_config) {
    auto scheduler_unique = create_lr_scheduler(scheduler_config.at("type").get<std::string>(), optimizer, group);
    scheduler_unique->set_params(scheduler_config);
    std::shared_ptr<LrSchedulerBase> scheduler = std::move(scheduler_unique);
    orchestrator->set_lr_scheduler(group, scheduler);
  };

  if (config.contains("lr_schedulers") && config.at("lr_schedulers").is_object()) {
    const auto& s = config.at("lr_schedulers");
    if (s.contains("means") && s.at("means").is_object()) maybe_set_scheduler(OptimParamGroup::Means, s.at("means"));
    if (s.contains("shs") && s.at("shs").is_object()) maybe_set_scheduler(OptimParamGroup::Shs, s.at("shs"));
    if (s.contains("opacities") && s.at("opacities").is_object()) maybe_set_scheduler(OptimParamGroup::Opacities, s.at("opacities"));
    if (s.contains("scales") && s.at("scales").is_object()) maybe_set_scheduler(OptimParamGroup::Scales, s.at("scales"));
    if (s.contains("rotations") && s.at("rotations").is_object()) maybe_set_scheduler(OptimParamGroup::Rotations, s.at("rotations"));
  } else if (config.contains("lr_scheduler") && config.at("lr_scheduler").is_object()) {
    const auto& s = config.at("lr_scheduler");
    maybe_set_scheduler(OptimParamGroup::Means, s);
    maybe_set_scheduler(OptimParamGroup::Shs, s);
    maybe_set_scheduler(OptimParamGroup::Opacities, s);
    maybe_set_scheduler(OptimParamGroup::Scales, s);
    maybe_set_scheduler(OptimParamGroup::Rotations, s);
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

void eval(std::shared_ptr<Orchestrator> orchestrator) {
  auto output_dir = orchestrator->get_config().out_dir;
  auto test_loader = orchestrator->get_test_dataloader();
  auto test_set_result = orchestrator->eval(test_loader.get());
  log_warning("Evaluation into {}, len={}", output_dir, test_loader->get_dataset()->size());
  const float test_psnr = test_set_result.at("psnr");
  log_info("Evaluation on test set done.");

  json result_json;
  result_json["psnr"] = test_psnr;
  result_json["time"] = orchestrator->get_config().max_seconds;
  std::cout << result_json << std::endl;
  std::ofstream out(fmt::format("{}/stats.json", output_dir));
  out << result_json;
}

void train(std::shared_ptr<Orchestrator> orchestrator, bool visualize) {
  auto gs3d = orchestrator->get_optimizer()->get_gaussians();
  auto last_log_time = std::chrono::steady_clock::now();
  orchestrator->set_post_step_callback([orchestrator, gs3d, visualize, last_log_time] (const TrainingState& state) mutable {
    if (state.current_step % 100 != 0) {
      return;
    }

    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log_time);
    auto loss = orchestrator->accumulate_loss();
    auto metrics = orchestrator->evaluate_metrics();
    float psnr = metrics.empty() ? 0.0f : metrics[0];
    auto lr = orchestrator->get_optimizer()->get_lr();

    constexpr int LOG_WIDTH = 80;  // Adjust based on terminal width
    auto duration_from_start = std::chrono::duration_cast<std::chrono::seconds>(now - state.last_log_time);

    auto log_string = fmt::format("STEP {:4d}] PSNR={:.3f} | TPUT={:4d}ms/100step | LR={:.1e} | N-Gs: {:7d} | T={:3d}s",
                                  state.current_step, psnr, duration.count(), lr, gs3d->size(), duration_from_start.count());

    std::cout << "\r" << std::left << std::setw(LOG_WIDTH) << log_string << std::flush;


    // Visualize RGB - copy rendered image from trainer's internal buffers
    if (visualize) {
      cv::Mat img = orchestrator->to_opencv();
      if (!img.empty()) {
        cv::imshow("render", img);

        if (char key = cv::waitKey(0); key == 27) {
          orchestrator->stop_training();
          std::cout << "ESC pressed - stopping training..." << std::endl;
        }
      }
    }
    last_log_time = now;
  });
  auto final_state = orchestrator->train();
  log_info("Training completed after {} steps", final_state.current_step);
  if (visualize) {
    cv::waitKey(0);
    cv::destroyAllWindows();
  }
  std::cout << std::endl;
  eval(orchestrator);
}
