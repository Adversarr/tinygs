#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <tinygs/core/camera.hpp>
#include <cuda_runtime.h>

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

    auto intrinsics = tinygs::CameraIntrinsics::parse(camera_intrinsics_lines.at(0));
    std::cout << "Camera Intrinsics K matrix:" << std::endl;
    std::cout << tinygs::to_string(intrinsics.get_K()) << std::endl;

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
    
    std::cout << "Press 'q' to quit, any other key to load next image" << std::endl;
    
    int frame_count = 0;
    while (true) {
      // data is on GPU
      auto data = loader.next();

      // Print camera information
      std::cout << "\n--- Frame " << frame_count << " ---" << std::endl;
      std::cout << "Image shape: " << tinygs::to_string(data.output.image.shape) << std::endl;
      std::cout << "Camera extrinsics (w2c):" << std::endl;
      std::cout << tinygs::to_string(data.input.w2c) << std::endl;
      std::cout << "Camera intrinsics (K):" << std::endl;
      std::cout << tinygs::to_string(data.input.K) << std::endl;
      
      // Copy image data from GPU to host
      // The image data is in CHW format with float values (0-1)
      if (data.output.image.data != nullptr) {
        // Allocate host memory for the image data
        std::vector<float> host_image_data(width * height * 3);
        
        // Copy from GPU to host
        cudaMemcpy(host_image_data.data(), data.output.image.data, 
                   width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Convert from CHW float (0-1) to HWC uint8 (0-255) for OpenCV
        cv::Mat image(height, width, CV_8UC3);
        
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            for (int c = 0; c < 3; ++c) {
              // CHW format: data is stored as [C0H0W0, C0H0W1, ..., C0H1W0, ..., C1H0W0, ...]
              int chw_idx = c * height * width + h * width + w;
              // HWC format: data is stored as [H0W0C0, H0W0C1, H0W0C2, H0W1C0, ...]
              int hwc_idx = h * width * 3 + w * 3 + c;
              
              // Convert float (0-1) to uint8 (0-255) and clamp values
              float pixel_value = host_image_data[chw_idx];
              pixel_value = std::max(0.0f, std::min(1.0f, pixel_value)); // Clamp to [0,1]
              image.data[hwc_idx] = static_cast<uint8_t>(pixel_value * 255.0f);
            }
          }
        }
        
        // Convert from RGB to BGR for OpenCV display
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        
        // Add text overlay with frame information
        std::string frame_text = "Frame: " + std::to_string(frame_count);
        cv::putText(image, frame_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        
        // Display the image
        cv::imshow("TinyGS Data Visualization", image);
        
        // Wait for key press
        char key = cv::waitKey(0);
        if (key == 'q' || key == 'Q') {
          break;
        }
      } else {
        std::cout << "Warning: Image data is null for frame " << frame_count << std::endl;
        break;
      }
      
      frame_count++;
    }
    
    cv::destroyAllWindows();
    return EXIT_SUCCESS;
    
  } catch (const std::exception &e) {
    log_error("Error: {}", e.what());
    return EXIT_FAILURE;
  }
}