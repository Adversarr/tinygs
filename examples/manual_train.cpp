#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <stdexcept>
#include <tinygs/core/camera.hpp>

#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/core/pointcloud.hpp"
#include "tinygs/cuda/common_host.hpp"
#include "tinygs/cuda/reduce.hpp"
#include "tinygs/dataloader/simple.hpp"
#include "tinygs/dataset/png_folder.hpp"
#include "tinygs/dataset/video.hpp"
#include "tinygs/initialization/knn.hpp"
#include "tinygs/loss/fused_ssim.hpp"
#include "tinygs/loss/l1.hpp"
#include "tinygs/loss/psnr.hpp"
#include "tinygs/optim/adamw.hpp"
#include "tinygs/optim/sgd.hpp"
#include "tinygs/rasterizer/default.hpp"
#include "tinygs/rasterizer/fastgs.hpp"
#include "tinygs/strategy/default.hpp"
#include "tinygs/strategy/mcmc.hpp"
#include "tinygs/trainer/trainer.hpp"
#include "tinygs/utils/file.hpp"
#include "tinygs/utils/inspect_change.hpp"
#include "tinygs/utils/scope_timer.hpp"
#include "tinygs/utils/stbi/stbi_wrapper.h"

using namespace tinygs;

std::vector<vec3> skybox(
  const DatasetBase& ds, size_t count
) {
  std::default_random_engine rng;
  std::normal_distribution<float> dist(0.0f, 1.0f);

  vec3 cam_pos_min = vec3(FLT_MAX);
  vec3 cam_pos_max = vec3(-FLT_MAX);

  for (int i = 0; i < ds.size(); i++) {
    auto cam = glm::inverse(ds[i].w2c);
    vec3 pos = cam[3];
    cam_pos_min = glm::min(cam_pos_min, pos);
    cam_pos_max = glm::max(cam_pos_max, pos);
  }

  vec3 center = (cam_pos_min + cam_pos_max) * 0.5f;
  float radius = max(cam_pos_max - cam_pos_min) * 5.f;
  auto rand_on_sphere = [&]() {
    vec3 p = vec3(dist(rng), dist(rng), dist(rng));
    p = glm::normalize(p);
    p = center + radius * p;
    return p;
  };
  // randomly sample points in the sphere
  std::vector<vec3> points;
  for (int i = 0; i < count; i++) {
    vec3 p = rand_on_sphere();
    points.push_back(p);
  }

  return points;
}

int main() {
  spdlog::set_level(spdlog::level::debug);
  std::string data_path = "/data/accgs/1747834320424/";
  std::string camera_intrinsics_path = data_path + "inputs/slam/cameras.txt";
  std::string camera_extrinsics_path = data_path + "inputs/traj_full.txt.bak";

  
  // Setup dataset and dataloader
  // std::shared_ptr<PngFolderDataset> dataset = std::make_shared<PngFolderDataset>(
  //     data_path + "inputs/images_480x640_1",
  //     camera_extrinsics_path,
  //     camera_intrinsics_path);
  std::shared_ptr<VideoDataset> dataset = std::make_shared<VideoDataset>(
    data_path + "1747834320424_flip.mp4",
    camera_extrinsics_path,
    camera_intrinsics_path);
  auto dataloader = std::make_shared<SimpleDataLoader>(dataset);


  ImageShape shape = dataset->image_shape();
  int width = shape.width, height = shape.height;

  // Load and initialize point cloud
  auto pc = load_from_colmap_file(data_path + "inputs/slam/points3D.txt");
  log_info("#points: {}", pc.points.size());

  // Extend with skybox points
  {
    auto p_sky = skybox(*dataset, 10000);
    for (auto& p : p_sky) {
      pc.points.push_back(p);
      pc.colors.push_back(vec3(0.7f));
    }
  }

  // Initialize gaussians
  KnnInitialization knn;
  knn.initialize(pc);
  auto init_result = knn.gaussians();

  auto gs3d = std::make_shared<GPUGaussian3d>();
  gs3d->copy_from_host(init_result);
  std::shared_ptr<GPUGaussian3d> grads(gs3d->clone().release());
  gs3d->set_sh_degree(0);

  // Setup trainer configuration
  TrainerConfig config;
  config.max_steps = 30000;
  config.initial_learning_rate = 1.0f;
  config.final_learning_rate = 0.01f;
  config.log_interval = 100;
  config.sh_degree_interval = 1000;
  config.max_sh_degree = 3;
  
  Trainer trainer(config);
  
  // Setup trainer components
  trainer.set_gaussians(gs3d, grads);
  trainer.set_dataloader(dataloader);
  
  auto rasterizer = std::make_shared<DefaultRasterizer>();
  trainer.set_rasterizer(rasterizer);
  
  auto optimizer = std::make_shared<AdamW>(gs3d, grads);
  trainer.set_optimizer(optimizer);
  
  auto strategy = std::make_shared<DefaultStrategy>(gs3d, grads, optimizer);
  trainer.set_strategy(strategy);
  
  // Add loss functions
  auto l1_loss = std::make_shared<L1Loss>();
  auto ssim_loss = std::make_shared<FusedSSIMLoss>();
  trainer.add_loss(l1_loss, 0.8f);
  trainer.add_loss(ssim_loss, 0.2f);
  
  // Add metrics
  auto psnr_metric = std::make_shared<PsnrMetric>();
  trainer.add_metric(psnr_metric, "PSNR");
  
  // Setup visualization callback
  GPUMemory<float> out_image(width * height * 3);
  auto visualization_callback = [&](const TrainingState& state, float loss, const std::vector<float>& metrics) {
    if (state.current_step % 100 == 0) {
      auto now = std::chrono::steady_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - state.last_log_time);
      
      log_info(
          "step {} loss: {:.6f} time: {}ms/100step",
          state.current_step,
          loss,
          duration.count());
      
      // Visualize RGB - copy rendered image from trainer's internal buffers
      const auto& rasterize_ctx = trainer.get_rasterize_context();
      if (rasterize_ctx.fwd_output.image.data != nullptr) {
        // Copy GPU rendered image to CPU for visualization
        std::vector<float> cpu_image(height * width * 3);
        cudaMemcpy(cpu_image.data(), rasterize_ctx.fwd_output.image.data, 
                   height * width * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Convert float RGB to 8-bit BGR for OpenCV
        cv::Mat img(height, width, CV_8UC3);
        for (int y = 0; y < height; ++y) {
          for (int x = 0; x < width; ++x) {
            // Convert from CHW (RGB) to HWC (BGR)
            int r_idx = y * width + x;                    // R channel offset
            int g_idx = (height * width) + r_idx;        // G channel offset
            int b_idx = (2 * height * width) + r_idx;    // B channel offset
            
            img.at<cv::Vec3b>(y, x)[0] = static_cast<uint8_t>(std::clamp(cpu_image[b_idx] * 255.0f, 0.0f, 255.0f));  // B
            img.at<cv::Vec3b>(y, x)[1] = static_cast<uint8_t>(std::clamp(cpu_image[g_idx] * 255.0f, 0.0f, 255.0f));  // G
            img.at<cv::Vec3b>(y, x)[2] = static_cast<uint8_t>(std::clamp(cpu_image[r_idx] * 255.0f, 0.0f, 255.0f));  // R
          }
        }
        cv::imshow("render", img);
      }
      
      if (char key = cv::waitKey(1); key == 27) {
        trainer.stop_training();
        std::cout << "ESC pressed - stopping training..." << std::endl;
      }
    }
  };
  
  trainer.set_post_step_callback(visualization_callback);
  
  // Start training
  auto final_state = trainer.train();
  
  log_info("Training completed after {} steps", final_state.current_step);
  
  cv::waitKey(0);
  cv::destroyAllWindows();
  GlobalTimerRegistry::get_instance().print_all_stats();
  return EXIT_SUCCESS;
}
