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
#include "tinygs/utils/image_format.hpp"

int main() {
  using namespace tinygs;
  Gaussian3d gaussian;
  gaussian.means.push_back({0.0f, 0.0f, 0.0f});
  gaussian.scales.push_back({0.1f, 0.1f, 0.1f});
  gaussian.rotations.push_back({0.0f, 0.0f, 0.0f, 1.0f});
  // Use partial opacity to allow blending
  gaussian.opacities.push_back(0.6f);
  gaussian.sh_coefficient_0.push_back({0.0f, 0.0f, 1.0f});
  for (int i = 0; i < 15; ++i) {
    gaussian.sh_coefficients_rest.push_back({0.0f, 0.0f, 0.0f});
  }

  // Second Gaussian: slightly behind the first and different color
  gaussian.means.push_back({0.3f, 0.3f, 0.2f});
  gaussian.scales.push_back({0.1f, 0.1f, 0.1f});
  gaussian.rotations.push_back({0.0f, 0.0f, 0.0f, 1.0f});
  gaussian.opacities.push_back(0.6f);
  gaussian.sh_coefficient_0.push_back({1.0f, 0.0f, 0.0f});
  for (int i = 0; i < 15; ++i) {
    gaussian.sh_coefficients_rest.push_back({0.0f, 0.0f, 0.0f});
  }
  auto gpu_gaussian = std::make_shared<GPUGaussian3d>();
  gpu_gaussian->copy_from_host(gaussian);

  int width = 1280;
  int height = 720;

  tinygs::GPUBatchInputOutput io;
  io.input.width = width;
  io.input.height = height;
  io.input.near = 0.001f;
  io.input.far = 10000.0f;
  CameraIntrinsics intrinsics{
    0,
    tinygs::CameraModel::Pinhole,
    width,
    height,
    /*fx=*/1000.0f,
    /*fy=*/1000.0f,
    /*cx=*/width / 2.0f,
    /*cy=*/height / 2.0f,
  };
  io.input.K = intrinsics.to_mat3();

  // Place the camera so the origin has positive z in camera space
  CameraExtrinsics extrinsics{quat(1.0f, 0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 3.0f), 0, 0};

  io.input.w2c = extrinsics.get_w2c();

  tinygs::GPUMemory<float> out_image(width * height * 3);
  io.output.image.shape.width = width;
  io.output.image.shape.height = height;
  io.output.image.shape.channel = 3;
  io.output.image.data = out_image.data();

  tinygs::RasterizeContext params;
  params.inference = true;
  params.fwd_input = io.input;
  params.fwd_output = io.output;

  FastGSRasterizer fastgs_rasterizer;
  fastgs_rasterizer.set_gaussians(gpu_gaussian);
  params.fwd_input = io.input;
  params.fwd_output = io.output;

  fastgs_rasterizer.forward(params);

  cv::Mat image(height, width, CV_8UC3);
  std::vector<float> image_host(width * height * 3);
  out_image.copy_to_host(image_host);
  std::vector<uchar> image_host_8uc3(width * height * 3);
  for (int i = 0; i < width * height * 3; ++i) {
    image_host_8uc3[i] = static_cast<uchar>(image_host[i] * 255.0f);
  }

  tinygs::ImageShape shape;
  shape.width = static_cast<uint32_t>(width);
  shape.height = static_cast<uint32_t>(height);
  shape.channel = 3;
  to_cv2(image.data, image_host_8uc3.data(), shape);

  cv::imshow("image", image);
  cv::waitKey(0);
  return 0;
}