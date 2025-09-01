#include <iostream>
#include <tinygs/core/camera.hpp>

#include "tinygs/cuda/common_host.hpp"
#include "tinygs/dataset/png_folder.hpp"
#include "tinygs/utils/file.hpp"
#include "tinygs/dataloader/simple.hpp"

std::string DATA_PATH = "/data/accgs/1747834320424/";

int main() {
  spdlog::set_level(spdlog::level::debug);
  try {
    std::string camera_intrinsics_path = DATA_PATH + "inputs/slam/cameras.txt";
    std::string camera_extrinsics_path = DATA_PATH + "inputs/traj_full.txt.bak";
    std::vector<std::string> camera_intrinsics_lines = tinygs::readlines(camera_intrinsics_path);
    std::vector<std::string> camera_extrinsics_lines = tinygs::readlines(camera_extrinsics_path);

    auto intrisics = tinygs::CameraIntrinsics::parse(camera_intrinsics_lines.at(0));
    std::cout << tinygs::to_string(intrisics.get_K()) << std::endl;

    // first 5 extrinsics
    for (int i = 0; i < 5; i++) {
      auto extrinsics = tinygs::CameraExtrinsics::parse(camera_extrinsics_lines.at(i));
      std::cout << tinygs::to_string(extrinsics.get_w2c()) << std::endl;
    }

    int width = 480, height = 640;
    tinygs::ImageShape shape;
    shape.width = width;
    shape.height = height;
    shape.channels = 3;

    auto start = std::chrono::system_clock::now();
    std::shared_ptr<tinygs::PngFolderDataset> dataset = std::make_shared<tinygs::PngFolderDataset>(
        DATA_PATH + "inputs/images_480x640_1", camera_extrinsics_path,
        camera_intrinsics_path, shape);
    auto end = std::chrono::system_clock::now();
    std::cout << "Load image cost: " << (end - start).count() << std::endl;

    tinygs::SimpleDataLoader loader(dataset);
    auto data = loader.next();
    std::cout << tinygs::to_string(data.output.image.shape) << std::endl;
    std::cout << tinygs::to_string(data.input.w2c) << std::endl;
    std::cout << tinygs::to_string(data.input.K) << std::endl;

    return EXIT_SUCCESS;
  } catch (const std::exception &e) {
    log_error("Error: {}", e.what());
    return EXIT_FAILURE;
  }
}