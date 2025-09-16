#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/initialization/initialization.hpp"
#include "tinygs/optim/lr_scheduler.hpp"
#include "tinygs/optim/optim.hpp"
#include "tinygs/rasterizer/rasterizer.hpp"
#include "tinygs/trainer/trainer.hpp"
#include <cxxopts.hpp>
#include <iostream>
#include <fstream>
#include <tinygs/dataset/dataset.hpp>
#include <tinygs/dataloader/dataloader.hpp>

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

  json j;

  std::shared_ptr<DatasetBase> dataset = create_dataset("png_folder");
  j["dataset"] = dataset->get_params();

  std::shared_ptr<DataLoaderBase> dataloader = create_dataloader("simple", dataset);
  j["dataloader"] = dataloader->get_params();

  std::shared_ptr<InitializationBase> initializer = create_initialization("knn");
  j["initializer"] = initializer->get_params();

  std::shared_ptr<GPUGaussian3d> gs3d = std::make_shared<GPUGaussian3d>();
  std::shared_ptr<RasterizerBase> rasterizer = create_rasterizer("default");
  j["rasterizer"] = rasterizer->get_params();

  std::shared_ptr<OptimizerBase> opt = create_optimizer("adam", gs3d, gs3d);
  j["optimizer"] = opt->get_params();

  std::shared_ptr<LrSchedulerBase> lr_scheduler = create_lr_scheduler("exponential", opt);
  j["lr_scheduler"] = lr_scheduler->get_params();

  j["loss"] = json::array({
    json::object({
      {"type", "l1"}, {"weight", 0.8},
    }),
    json::object({
      {"type", "fused_ssim"}, {"weight", 0.2},
    }),
  });
  j["metric"] = json::array({"psnr"});
  j["input_pc_file"] = "YOUR_POINT_CLOUD.txt";

  Trainer t;
  j["trainer"] = t.get_params();

  if (result.count("output")) {
    std::ofstream output_file(result["output"].as<std::string>());
    output_file << j.dump(4) << std::endl;
  } else {
    std::cout << j.dump(4) << std::endl;
  }
  return 0;
}