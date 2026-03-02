#include <cuda_runtime.h>

#include <algorithm>
#include <cxxopts.hpp>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <tinygs/core/camera.hpp>

#include "glm/gtx/string_cast.hpp"
#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/cuda/gpu_memory.hpp"
#include "tinygs/random/pcg32.hpp"
#include "tinygs/rasterizer/rasterizer.hpp"
#include "tinygs/utils/image_format.hpp"

int main(int argc, char** argv) {
  cxxopts::Options options("single_gs", "Single Gaussian Splatting");
  options.add_options()
    ("h,help", "Print help")
    ("r,rasterizer", "Rasterizer to use", cxxopts::value<std::string>()->default_value("default"))
    ("o1,opacity1", "Opacity of the Gaussian 1", cxxopts::value<float>()->default_value("0.6"))
    ("o2,opacity2", "Opacity of the Gaussian 2", cxxopts::value<float>()->default_value("0.6"))
    ("s1,scale1", "Scale of the Gaussian 1", cxxopts::value<float>()->default_value("0.1"))
    ("s2,scale2", "Scale of the Gaussian 2", cxxopts::value<float>()->default_value("0.1"));

  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  float opacity1 = result["opacity1"].as<float>();
  float opacity2 = result["opacity2"].as<float>();
  float scale1 = result["scale1"].as<float>();
  float scale2 = result["scale2"].as<float>();
  std::string rasterizer = result["rasterizer"].as<std::string>();

  std::cout<< "opacity1: " << opacity1 << std::endl;
  std::cout<< "opacity2: " << opacity2 << std::endl;
  std::cout<< "scale1: " << scale1 << std::endl;
  std::cout<< "scale2: " << scale2 << std::endl;
  std::cout<< "rasterizer: " << rasterizer << std::endl;

  using namespace tinygs;
  Gaussian3d gaussian;
  gaussian.means.push_back({0.0f, 0.0f, 0.0f});
  gaussian.scales.push_back({scale1, -scale1, 0.3f * scale1});
  gaussian.rotations.push_back(/* glm::normalize */(vec4{-1.0f, 0.0f, 1.0f, 1.0f} * 3.0f));
  // Use partial opacity to allow blending
  gaussian.opacities.push_back(opacity1);
  gaussian.sh0.push_back({0.0f, 0.0f, 1.0f});
  // Initialize per-degree SH rest to zero (3 + 5 + 7 = 15 coefficients)
  for (int i = 0; i < kSHDegreeNumCoeffs[1]; ++i) gaussian.sh1.push_back({0.0f, 0.0f, 0.0f});
  for (int i = 0; i < kSHDegreeNumCoeffs[2]; ++i) gaussian.sh2.push_back({0.0f, 0.0f, 0.0f});
  for (int i = 0; i < kSHDegreeNumCoeffs[3]; ++i) gaussian.sh3.push_back({0.0f, 0.0f, 0.0f});

  // Second Gaussian: slightly behind the first and different color
  gaussian.means.push_back({0.5f, 0.3f, 0.1f});
  gaussian.scales.push_back({scale2, -scale2, 0.3f * scale2});
  gaussian.rotations.push_back(/* glm::normalize */(vec4{1.0f, 1.0f, 0.0f, -1.0f} * 3.0f));
  gaussian.opacities.push_back(opacity2);
  gaussian.sh0.push_back({1.0f, 0.0f, 0.0f});
  for (int i = 0; i < kSHDegreeNumCoeffs[1]; ++i) gaussian.sh1.push_back({0.0f, 0.0f, 0.0f});
  for (int i = 0; i < kSHDegreeNumCoeffs[2]; ++i) gaussian.sh2.push_back({0.0f, 0.0f, 0.0f});
  for (int i = 0; i < kSHDegreeNumCoeffs[3]; ++i) gaussian.sh3.push_back({0.0f, 0.0f, 0.0f});
  auto gpu_gaussian = std::make_shared<GPUGaussian3d>();
  gpu_gaussian->copy_from_host(gaussian);

  int width = 480;
  int height = 360;

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
    /*fx=*/375.0f,
    /*fy=*/375.0f,
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
  params.densification_info = std::make_shared<GPUBuffer<DensificationInfo>>(gpu_gaussian->size());

  auto rast = create_rasterizer(rasterizer);
  rast->set_gaussians(gpu_gaussian);
  params.fwd_input = io.input;
  params.fwd_output = io.output;
  rast->forward(params);

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

  // Compute comprehensive image statistics for each channel
  float max_r = 0.0f, max_g = 0.0f, max_b = 0.0f;
  float min_r = 1.0f, min_g = 1.0f, min_b = 1.0f;
  float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
  float sum_sq_r = 0.0f, sum_sq_g = 0.0f, sum_sq_b = 0.0f;
  
  for (int i = 0; i < width * height; ++i) {
    float r = image_host[i * 3 + 0];
    float g = image_host[i * 3 + 1];
    float b = image_host[i * 3 + 2];
    
    max_r = std::max(max_r, r);
    max_g = std::max(max_g, g);
    max_b = std::max(max_b, b);
    
    min_r = std::min(min_r, r);
    min_g = std::min(min_g, g);
    min_b = std::min(min_b, b);
    
    sum_r += r;
    sum_g += g;
    sum_b += b;
    
    sum_sq_r += r * r;
    sum_sq_g += g * g;
    sum_sq_b += b * b;
  }
  
  int total_pixels = width * height;
  float avg_r = sum_r / total_pixels;
  float avg_g = sum_g / total_pixels;
  float avg_b = sum_b / total_pixels;
  
  float std_r = std::sqrt(sum_sq_r / total_pixels - avg_r * avg_r);
  float std_g = std::sqrt(sum_sq_g / total_pixels - avg_g * avg_g);
  float std_b = std::sqrt(sum_sq_b / total_pixels - avg_b * avg_b);
  
  std::cout << "Max color: R=" << max_r << " G=" << max_g << " B=" << max_b << std::endl;
  std::cout << "Min color: R=" << min_r << " G=" << min_g << " B=" << min_b << std::endl;
  std::cout << "Avg color: R=" << avg_r << " G=" << avg_g << " B=" << avg_b << std::endl;
  std::cout << "Std color: R=" << std_r << " G=" << std_g << " B=" << std_b << std::endl;

  // backward pass
  GPUMemory<float> out_image_grad(width * height * 3);
  std::vector<float> out_image_grad_host(width * height * 3);
  pcg32 rng(0, 1u);
  for (int i = 0; i < width * height * 3; ++i) {
    out_image_grad_host[i] = (static_cast<float>(rng.next_uint(256)) / 255.0f) / total_pixels;
  }
  out_image_grad.copy_from_host(out_image_grad_host);
  std::shared_ptr<GPUGaussian3d> grad = gpu_gaussian->clone();
  params.grad_output.image = Image(shape, DataType::Float32, out_image_grad.data());
  params.gaussians_grad = grad;
  grad->memset(0);
  rast->backward(params);

  Gaussian3d gaussian_grad;
  grad->copy_to_host(gaussian_grad);
  const std::vector<vec3>& means_grad = gaussian_grad.means;
  const std::vector<vec3>& scales_grad = gaussian_grad.scales;
  const std::vector<vec4>& rotations_grad = gaussian_grad.rotations;
  const std::vector<float>& opacities_grad = gaussian_grad.opacities;
  const std::vector<vec3>& sh0_grad = gaussian_grad.sh0;

  // Print gradients for the two Gaussians
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "=== Gaussian Gradients ===" << std::endl;
  for (size_t i = 0; i < means_grad.size() && i < 2; ++i) {
    const auto& m = means_grad[i];
    const auto& s = scales_grad[i];
    const auto& r = rotations_grad[i];
    float o = opacities_grad[i];
    const auto& c0 = sh0_grad[i];

    std::cout << "Gaussian " << i << ":" << std::endl;
    std::cout << "  dMeans: [" << m.x << ", " << m.y << ", " << m.z << "]" << std::endl;
    std::cout << "  dScales: [" << s.x << ", " << s.y << ", " << s.z << "]" << std::endl;
    std::cout << "  dRotations: [" << r.x << ", " << r.y << ", " << r.z << ", " << r.w << "]" << std::endl;
    std::cout << "  dOpacities: " << o << std::endl;
    std::cout << "  dSH0: [" << c0.x << ", " << c0.y << ", " << c0.z << "]" << std::endl;
  }
  auto dinfo = params.densification_info->to_cpu();

  for (int i = 0; i < dinfo.size(); ++i) {
    std::cout << "  dDensificationInfo: " << dinfo[i].accum_absgrad_mean2d << std::endl;
    std::cout << "  dDensificationInfo: " << dinfo[i].accum_grad_mean2d << std::endl;
    std::cout << "  dDensificationInfo: " << dinfo[i].accum_counter << std::endl;
  }

  return 0;
}