#include "tinygs/cuda/common_host.hpp"
#include "tinygs/utils/file.hpp"
#include <iostream>
#include <tinygs/core/camera.hpp>

std::string DATA_PATH = "/data/accgs/1747834320424/";

int main() {
  try {
    std::string camera_intrinsics_path = DATA_PATH + "inputs/slam/cameras.txt";
    std::string camera_extrinsics_path = DATA_PATH + "inputs/traj_full.txt.bak";
    // std::string camera_extrinsics_path = DATA_PATH + "inputs/slam/images.txt";
    std::vector<std::string> camera_intrinsics_lines =
        tinygs::readlines(camera_intrinsics_path);
    std::vector<std::string> camera_extrinsics_lines =
        tinygs::readlines(camera_extrinsics_path);

    auto intrisics =
        tinygs::CameraIntrinsics::parse(camera_intrinsics_lines.at(0));
    std::cout << tinygs::to_string(intrisics.get_K()) << std::endl;

    // first 5 extrinsics
    for (int i = 0; i < 5; i++) {
      auto extrinsics = tinygs::CameraExtrinsics::parse(camera_extrinsics_lines.at(i));
      std::cout << tinygs::to_string(extrinsics.get_w2c()) << std::endl;
    }
    

    return EXIT_SUCCESS;
  } catch (const std::exception &e) {
    log_error("Error: {}", e.what());
    return EXIT_FAILURE;
  }
}