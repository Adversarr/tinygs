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
#include "tinygs/utils/file.hpp"
#include "tinygs/utils/inspect_change.hpp"
#include "tinygs/utils/scope_timer.hpp"
#include "tinygs/utils/stbi/stbi_wrapper.h"

using namespace tinygs;

std::vector<vec3> skybox(
  const PngFolderDataset& ds, size_t count
) {
  std::default_random_engine rng;
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  
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
  std::string data_path = "/data/accgs/1748422612463/";
  std::string camera_intrinsics_path = data_path + "inputs/slam/cameras.txt";
  std::string camera_extrinsics_path = data_path + "inputs/traj_full.txt.bak";

  int width = 480, height = 640;
  ImageShape shape;
  shape.width = width;
  shape.height = height;
  shape.channel = 3;
  std::shared_ptr<PngFolderDataset> dataset = std::make_shared<PngFolderDataset>(  //
      data_path + "inputs/images_480x640_1",                                                       //
      camera_extrinsics_path,                                                                      //
      camera_intrinsics_path,                                                                      //
      shape);

  SimpleDataLoader loader(dataset);

  auto pc = load_from_colmap_file(data_path + "inputs/slam/points3D.txt");
  log_info("#points: {}", pc.points.size());

  // Extend
  {
    auto p_sky = skybox(*dataset, 10000);
    for (auto& p : p_sky) {
      pc.points.push_back(p);
      pc.colors.push_back(vec3(0.7f));
    }
  }

  KnnInitialization knn;
  knn.initialize(pc);
  auto init_result = knn.gaussians();


  // Prepare Render data.
  auto gs3d = std::make_shared<GPUGaussian3d>();
  gs3d->copy_from_host(init_result);
  std::shared_ptr<GPUGaussian3d> grads = gs3d->clone();
  // gs3d->set_scene_scale(knn.get_scene_scale());
  gs3d->set_sh_degree(0);

  GPUBatchInputOutput io;
  io.input.width = width;
  io.input.height = height;
  io.input.batch_size = 1;
  io.input.near = 0.001f;
  io.input.far = 10000.0f;

  GPUMemory<float> out_image(width * height * 3);
  GPUMemory<float> out_alpha(width * height * 1);
  io.output.image.shape.width = io.output.alpha.shape.width = width;
  io.output.image.shape.height = io.output.alpha.shape.height = height;
  io.output.image.shape.channel = 3;
  io.output.alpha.shape.channel = 1;
  io.output.image.format = io.output.alpha.format = ImageFormat::HWC;
  io.output.image.data = out_image.data();
  io.output.alpha.data = out_alpha.data();
  RasterizeContext rasterize_ctx;
  rasterize_ctx.inference = true;
  rasterize_ctx.fwd_input = io.input;
  rasterize_ctx.fwd_output = io.output;
  rasterize_ctx.gaussians_grad = grads;

  // Rendering.
  DefaultRasterizer rasterizer;
  // FastGSRasterizer rasterizer;
  rasterizer.set_gaussians(gs3d);

  PsnrMetric psnr;

  // Optimizer
  auto optimizer = std::make_unique<AdamW>(gs3d, grads);

  // Loss
  GPUBuffer<float> loss_buffer = GPUBuffer<float>(shape.width * shape.height * 4);
  GPUMemory<float> out_image_grad(width * height * 3);
  auto l1_loss = std::make_unique<L1Loss>();
  auto ssim_loss = std::make_unique<FusedSSIMLoss>();
  LossContext loss_ctx;
  loss_ctx.loss = Image<float>(shape, ImageFormat::CHW, loss_buffer.data());
  loss_ctx.pred = rasterize_ctx.fwd_output.image;
  loss_ctx.grad = Image<float>(shape, ImageFormat::CHW, out_image_grad.data());

  // Densification Strategy
  auto strategy = std::make_unique<DefaultStrategy>(gs3d);
  strategy->set_remove_callback([&](char* kept_flag, int num_kept) {
    if (num_kept == gs3d->size()) return;
    optimizer->remove(kept_flag, num_kept);
    gs3d->remove(kept_flag, num_kept);
    grads->remove(kept_flag, num_kept);
  });

  strategy->set_duplicate_callback([&](int* src, int* dst, int num_duplications) {
    if (num_duplications <= 0) return;
    gs3d->append(num_duplications); // It is strategy's responsibility to update the gaussians.
    grads->append(num_duplications);
    optimizer->duplicate(src, dst, num_duplications);
  });

  strategy->set_reset_callback([&](int* indices, int num_reset) {
    if (num_reset <= 0) return;
    optimizer->reset(indices, num_reset);
  });

  strategy->set_reset_opacity_callback([&] {
    optimizer->reset_opacity();
  });

  int frame_count = 0;
  bool should_stop = false;
  auto beg = std::chrono::steady_clock::now();
  auto last = beg;

  rasterize_ctx.densification_info = std::make_shared<GPUBuffer<float>>(gs3d->size() * 2);
  rasterize_ctx.densification_info->memset(0);

  while (!should_stop) {
    grads->memset(0);
    loss_buffer.memset(0);
    out_image_grad.memset(0);

    auto data = loader.next();
    io.input.K = data.input.K;
    io.input.w2c = data.input.w2c;
    rasterize_ctx.fwd_input = io.input;
    loss_ctx.target = data.output.image;
    rasterizer.forward(rasterize_ctx);
    loss_ctx.scale = 0.8f;
    l1_loss->evaluate(loss_ctx);
    loss_ctx.scale = 0.2f;
    ssim_loss->evaluate(loss_ctx);
    
    rasterize_ctx.grad_output.image = loss_ctx.grad;
    rasterize_ctx.grad_output.alpha = Image<float>(  //
      shape, ImageFormat::CHW,                     //
        loss_buffer.data() + shape.width * shape.height * 3);
    rasterizer.backward(rasterize_ctx);

    if (frame_count % 100 == 0) {
      auto now = std::chrono::steady_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - beg);

      auto metric_psnr = psnr.evaluate(loss_ctx.pred, loss_ctx.target);
      log_info(
          "step {} loss: {} psnr: {} time: {}ms/100step, {}s elapsed",
          frame_count,
          gpu_sum(loss_ctx.loss.data, shape.width * shape.height * 3),
          metric_psnr, duration.count(),
          std::chrono::duration_cast<std::chrono::seconds>(now - last).count());
      beg = now;

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
            // h_img_hwc[hwc_idx] = static_cast<uint8_t>(std::clamp(h_img[chw_idx], 0.f, 1.f) * 255.0f);
          }
        }
      }

      cv::Mat img(height, width, CV_8UC3, h_img_hwc.data());
      // cv::imwrite("render.png", img);
      cv::imshow("render", img);

      if (char key = cv::waitKey(1); key == 27) {
        should_stop = true;
      }
    }

    
    float current_step_size = 1.0f;
    // float current_step_size = std::powf(0.01f, frame_count / 30000.0f);
    optimizer->step(current_step_size);
    strategy->step(rasterize_ctx);
    gs3d->set_sh_degree(frame_count / 1000);

    frame_count++;
    if (frame_count > 30000) {
      should_stop = true;
    }
  }

  cv::waitKey(0);
  cv::destroyAllWindows();
  GlobalTimerRegistry::get_instance().print_all_stats();
  return EXIT_SUCCESS;
}
