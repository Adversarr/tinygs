#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <tinygs/dataloader/dataloader.hpp>
#include <tinygs/dataset/dataset.hpp>

#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/initialization/initialization.hpp"
#include "tinygs/optim/lr_scheduler.hpp"
#include "tinygs/optim/optim.hpp"
#include "tinygs/platform/runtime_factory.hpp"
#include "tinygs/rasterizer/rasterizer.hpp"
#include "tinygs/orchestrator.hpp"

using namespace tinygs;

int main(int argc, char* argv[]) {
  cxxopts::Options options("export_default", "Export default model json config.");
  options.add_options()
    ("h,help", "Print help")
    ("o,output", "Output file path", cxxopts::value<std::string>());

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  // Create a backend runtime for parameter extraction
  BackendConfig backend_config;
  auto runtime_result = create_backend_runtime(backend_config);
  if (!runtime_result.ok()) {
    std::cerr << "Failed to create backend runtime: " << to_string(runtime_result.error()) << std::endl;
    return 1;
  }
  auto runtime = runtime_result.value();

  json j;

  std::shared_ptr<DatasetBase> dataset = create_dataset("image", *runtime);
  j["dataset"] = dataset->get_params();

  std::shared_ptr<DataLoaderBase> dataloader = create_dataloader("simple", *runtime, dataset);
  j["dataloader"] = dataloader->get_params();

  std::shared_ptr<InitializationBase> initializer = create_initialization("knn");
  j["initializer"] = initializer->get_params();

  std::shared_ptr<GPUGaussian3d> gs3d = std::make_shared<GPUGaussian3d>(*runtime);
  std::shared_ptr<RasterizerBase> rasterizer = create_rasterizer("fastgs", *runtime);
  j["rasterizer"] = rasterizer->get_params();

  std::shared_ptr<OptimizerBase> opt = create_optimizer("adam", *runtime, gs3d, gs3d);
  j["optimizer"] = opt->get_params();

  std::shared_ptr<LrSchedulerBase> lr_scheduler =
      create_lr_scheduler("exponential", opt, OptimParamGroup::Means);
  j["lr_scheduler"] = lr_scheduler->get_params();

  std::shared_ptr<StrategyBase> strategy = create_strategy("default", *runtime, gs3d, gs3d, opt);
  j["strategy"] = strategy->get_params();

  j["losses"] = json::array({
    json::object({
      {"type", "l1"}, {"weight", 0.8},
    }),
    json::object({
      {"type", "fused_ssim"}, {"weight", 0.2},
    }),
  });
  j["metrics"] = json::array({"psnr"});

  Orchestrator t;
  j["trainer"] = t.get_params();

  if (result.count("output")) {
    std::ofstream output_file(result["output"].as<std::string>());
    output_file << j.dump(4) << std::endl;
  } else {
    std::cout << j.dump(4) << std::endl;
  }
  return 0;
}