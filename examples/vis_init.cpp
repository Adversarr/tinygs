#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <tinygs/core/camera.hpp>

#include "glm/gtx/string_cast.hpp"
#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/core/pointcloud.hpp"
#include "tinygs/cuda/common_host.hpp"
#include "tinygs/dataloader/simple.hpp"
#include "tinygs/dataset/png_folder.hpp"
#include "tinygs/initialization/knn.hpp"
#include "tinygs/rasterizer/default.hpp"
#include "tinygs/rasterizer/fastgs.hpp"
#include "tinygs/utils/file.hpp"
#include "tinygs/utils/scope_timer.hpp"

std::string DATA_PATH = "/data/accgs/1747834320424/";

int main() {
  spdlog::set_level(spdlog::level::debug);
  try {
    std::string camera_intrinsics_path = DATA_PATH + "inputs/slam/cameras.txt";
    std::string camera_extrinsics_path = DATA_PATH + "inputs/traj_full.txt.bak";
    std::vector<std::string> camera_intrinsics_lines = tinygs::readlines(camera_intrinsics_path);
    std::vector<std::string> camera_extrinsics_lines = tinygs::readlines(camera_extrinsics_path);

    // Example usage of PointCloud
    auto pc = tinygs::load_from_colmap_file(DATA_PATH + "inputs/slam/points3D.txt");
    log_info("#points: {}", pc.points.size());

    tinygs::KnnInitialization knn;
    knn.initialize(pc);
    auto init_result = knn.gaussians();
    auto intrinsics = tinygs::CameraIntrinsics::parse(camera_intrinsics_lines.at(0));
    std::cout << "Camera Intrinsics K matrix:" << std::endl;
    std::cout << glm::to_string(intrinsics.to_mat3()) << std::endl;

    int width = 480, height = 640;
    tinygs::ImageShape shape;
    shape.width = width;
    shape.height = height;
    shape.channel = 3;
    std::shared_ptr<tinygs::PngFolderDataset> dataset = std::make_shared<tinygs::PngFolderDataset>(
        DATA_PATH + "inputs/images_480x640_1", camera_extrinsics_path,
        camera_intrinsics_path, shape);

    tinygs::SimpleDataLoader loader(dataset);

    // Prepare Render data.
    auto gs3d = std::make_shared<tinygs::GPUGaussian3d>();
    gs3d->copy_from_host(init_result);

    tinygs::GPUBatchInputOutput io;
    io.input.width = width;
    io.input.height = height;
    io.input.near = 0.001f;
    io.input.far = 10000.0f;

    tinygs::GPUMemory<float> out_image(width * height * 3);
    tinygs::GPUMemory<float> out_alpha(width * height * 1);
    io.output.image.shape.width = io.output.alpha.shape.width = width;
    io.output.image.shape.height = io.output.alpha.shape.height = height;
    io.output.image.shape.channel = 3;
    io.output.alpha.shape.channel = 1;
    io.output.image.format = io.output.alpha.format = tinygs::ImageFormat::HWC;
    io.output.image.data = out_image.data();
    io.output.alpha.data = out_alpha.data();
    tinygs::RasterizeContext params;
    params.inference = true;
    params.fwd_input = io.input;
    params.fwd_output = io.output;

    // Rendering.
    tinygs::DefaultRasterizer rasterizer;
    // tinygs::FastGSRasterizer rasterizer;
    rasterizer.set_gaussians(gs3d);
    
    std::cout << "Press 'q' to quit, any other key to load next image" << std::endl;
    
    int frame_count = 0;
    while (true) {
      auto data = loader.next();
      
      log_info("Frame {}: K={}", frame_count, to_string(data.input.K));
      log_info("Frame {}: w2c={}", frame_count, to_string(data.input.w2c));
      
      // Update camera parameters for this frame
      io.input.K = data.input.K;
      io.input.w2c = data.input.w2c;
      params.fwd_input = io.input;

      params.gaussians_grad = gs3d; // TODO: impl
      
      // Render the frame
      rasterizer.forward(params);

      params.grad_output.image = params.fwd_output.image;
      params.grad_output.alpha = params.fwd_output.alpha;

      // rasterizer.backward(params);
      
      // Visualize RGB
      std::vector<float> h_img(width * height * 3);
      out_image.copy_to_host(h_img); // CHW format 
      
      std::vector<uint8_t> h_img_hwc(width * height * 3);
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          for (int c = 0; c < 3; c++) {
            // CHW format: data is stored as [C0H0W0, C0H0W1, ..., C0H1W0, ..., C1H0W0, ...]
            int chw_idx = c * height * width + h * width + w;
            // HWC format: data is stored as [H0W0C0, H0W0C1, H0W0C2, H0W1C0, ...]
            int hwc_idx = h * width * 3 + w * 3 + c;
            h_img_hwc[hwc_idx] = static_cast<uint8_t>(h_img[chw_idx] * 255.0f);
          }
        }
      }

      cv::Mat vis_image(height, width, CV_8UC3, h_img_hwc.data());
      
      // Add frame counter to the image
      std::string frame_text = "Frame: " + std::to_string(frame_count);
      cv::putText(vis_image, frame_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
      
      cv::imshow("Visualization", vis_image);
      
      // Wait for key press
      char key = cv::waitKey(0);
      if (key == 'q' || key == 'Q') {
        break;
      }
      
      frame_count++;
    }


    cv::destroyAllWindows();
    tinygs::GlobalTimerRegistry::get_instance().print_all_stats();
    return EXIT_SUCCESS;

  } catch (const std::exception &e) {
    log_error("Error: {}", e.what());
    return EXIT_FAILURE;
  }
}