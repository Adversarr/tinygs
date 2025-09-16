#include "cxxopts.hpp"
#include "tinygs/orchestrator.hpp"

void train(const std::string& config_path);

int main(int argc, char** argv) {
  auto opts = cxxopts::Options("config_train", "Train model with config file.");

  opts.add_options()
    ("h,help", "Print help")
    ("c,config", "Config file path", cxxopts::value<std::string>());

  auto result = opts.parse(argc, argv);

  if (result.count("help")) {
    std::cout << opts.help() << std::endl;
    return 0;
  }

  if (result.count("config")) {
    train(result["config"].as<std::string>());
  } else {
    std::cout << "Config file path is required." << std::endl;
    std::cout << opts.help() << std::endl;
    return 0;
  }
  return 0;
}