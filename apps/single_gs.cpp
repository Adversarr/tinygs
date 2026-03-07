#include <algorithm>
#include <cxxopts.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <opencv2/opencv.hpp>
#include <tinygs/core/camera.hpp>

#include "tinygs/common.hpp"
#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/platform/buffer_utils.hpp"
#include "tinygs/platform/runtime_factory.hpp"
#include "tinygs/random/pcg32.hpp"
#include "tinygs/rasterizer/rasterizer.hpp"
#include "tinygs/utils/image_format.hpp"

int main(int argc, char** argv) {
  cxxopts::Options options("single_gs", "Single Gaussian Splatting");
  options.add_options()
    ("h,help", "Print help")
    ("r,rasterizer", "Rasterizer to use", cxxopts::value<std::string>()->default_value("fastgs"))
    ("fp16", "Use FP16 precision for output image", cxxopts::value<bool>()->default_value("false"))
    ("o1,opacity1", "Opacity of the Gaussian 1", cxxopts::value<float>()->default_value("0.6"))
    ("o2,opacity2", "Opacity of the Gaussian 2", cxxopts::value<float>()->default_value("0.6"))
    ("s1,scale1", "Scale of the Gaussian 1", cxxopts::value<float>()->default_value("0.1"))
    ("s2,scale2", "Scale of the Gaussian 2", cxxopts::value<float>()->default_value("0.1"))
    ("fd-check", "Run full finite-difference gradient validation", cxxopts::value<bool>()->default_value("true"))
    ("fd-eps", "Finite-difference epsilon", cxxopts::value<float>()->default_value("1e-3"));

  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  float opacity1 = result["opacity1"].as<float>();
  float opacity2 = result["opacity2"].as<float>();
  float scale1 = result["scale1"].as<float>();
  float scale2 = result["scale2"].as<float>();
  bool run_fd_check = result["fd-check"].as<bool>();
  float fd_eps = result["fd-eps"].as<float>();
  std::string rasterizer = result["rasterizer"].as<std::string>();
  bool use_fp16 = result["fp16"].as<bool>();

  std::cout<< "opacity1: " << opacity1 << std::endl;
  std::cout<< "opacity2: " << opacity2 << std::endl;
  std::cout<< "scale1: " << scale1 << std::endl;
  std::cout<< "scale2: " << scale2 << std::endl;
  std::cout<< "rasterizer: " << rasterizer << std::endl;
  std::cout<< "fp16: " << (use_fp16 ? "true" : "false") << std::endl;
  std::cout<< "fd_check: " << (run_fd_check ? "true" : "false") << std::endl;
  std::cout<< "fd_eps: " << fd_eps << std::endl;

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

  tinygs::BackendConfig backend_cfg;
  backend_cfg.type = tinygs::BackendType::Cuda;
  backend_cfg.device = 0;
  auto rt_result = tinygs::create_backend_runtime(backend_cfg);
  if (!rt_result.ok()) {
    std::cerr << "Failed to create runtime: " << tinygs::to_string(rt_result.error()) << std::endl;
    return 1;
  }
  auto runtime = rt_result.value();
  auto q_result = runtime->create_queue({});
  if (!q_result.ok()) {
    std::cerr << "Failed to create queue: " << tinygs::to_string(q_result.error()) << std::endl;
    return 1;
  }
  auto queue = q_result.value();
  auto sync_queue_or_throw = [&](const char* op_name) {
    const auto status = runtime->synchronize_queue(*queue);
    if (!status.ok()) {
      throw std::runtime_error(std::string(op_name) + " failed: " + tinygs::to_string(status));
    }
  };

  auto gpu_gaussian = std::make_shared<GPUGaussian3d>(*runtime);
  gpu_gaussian->set_sh_degree(3);
  gpu_gaussian->copy_from_host_async(gaussian, queue.get());

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
  CameraExtrinsics extrinsics{quat(1.0f, 0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 3.0f), 0, 0, 0};

  io.input.w2c = extrinsics.get_w2c();

  DataType out_data_type = use_fp16 ? DataType::Float16 : DataType::Float32;
  int total_pixels = width * height;

  tinygs::ImageShape shape;
  shape.width = static_cast<uint32_t>(width);
  shape.height = static_cast<uint32_t>(height);
  shape.channel = 3;
  int padded_size = shape.padded_size();
  int pad_width = shape.padded_width();
  int channel_stride = shape.padded_height() * pad_width;

  std::shared_ptr<tinygs::BackendBuffer> out_image_fp32;
  std::shared_ptr<tinygs::BackendBuffer> out_image_fp16;
  std::shared_ptr<tinygs::BackendBuffer> out_image_convert;

  if (use_fp16) {
    out_image_fp16 = tinygs::create_device_buffer_for<tinygs::float16_t>(*runtime, padded_size, "out_image_fp16");
    io.output.image.data = tinygs::buffer_data<tinygs::float16_t>(out_image_fp16);
    out_image_convert = tinygs::create_device_buffer_for<float>(*runtime, padded_size, "out_image_convert");
  } else {
    out_image_fp32 = tinygs::create_device_buffer_for<float>(*runtime, padded_size, "out_image_fp32");
    io.output.image.data = tinygs::buffer_data<float>(out_image_fp32);
  }
  io.output.image.shape = shape;
  io.output.image.data_type = out_data_type;

  auto tiled_to_linear_hwc = [&](const std::vector<float>& tiled) -> std::vector<float> {
    std::vector<float> linear_hwc(total_pixels * 3);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int pixel_offset = get_linear_index(y, x, pad_width);
        int r_idx = pixel_offset + 0 * channel_stride;
        int g_idx = pixel_offset + 1 * channel_stride;
        int b_idx = pixel_offset + 2 * channel_stride;
        int out_idx = (y * width + x) * 3;
        linear_hwc[out_idx + 0] = tiled[r_idx];
        linear_hwc[out_idx + 1] = tiled[g_idx];
        linear_hwc[out_idx + 2] = tiled[b_idx];
      }
    }
    return linear_hwc;
  };

  auto get_image_as_linear_hwc = [&]() -> std::vector<float> {
    std::vector<float> tiled_data(padded_size);
    if (use_fp16) {
      half_to_float_gpu(tinygs::buffer_data<float>(out_image_convert),
                        tinygs::buffer_data<tinygs::float16_t>(out_image_fp16), padded_size, queue.get());
      tinygs::copy_to_host_async(*runtime, *queue, out_image_convert, tiled_data);
    } else {
      tinygs::copy_to_host_async(*runtime, *queue, out_image_fp32, tiled_data);
    }
    sync_queue_or_throw("single_gs read image");
    std::vector<float> linear_hwc(total_pixels * 3);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int pixel_offset = get_linear_index(y, x, pad_width);
        int r_idx = pixel_offset + 0 * channel_stride;
        int g_idx = pixel_offset + 1 * channel_stride;
        int b_idx = pixel_offset + 2 * channel_stride;
        int out_idx = (y * width + x) * 3;
        linear_hwc[out_idx + 0] = tiled_data[r_idx];
        linear_hwc[out_idx + 1] = tiled_data[g_idx];
        linear_hwc[out_idx + 2] = tiled_data[b_idx];
      }
    }
    return linear_hwc;
  };

  tinygs::RasterizeContext params;
  params.inference = true;
  params.fwd_input = io.input;
  params.fwd_output = io.output;

  params.runtime = runtime.get();
  params.queue = queue.get();
  params.densification_info = tinygs::create_device_buffer_for<DensificationInfo>(
      *params.runtime, gpu_gaussian->size(), "densification_info");

  auto rast = create_rasterizer(rasterizer, *runtime);
  rast->set_gaussians(gpu_gaussian);
  params.fwd_input = io.input;
  params.fwd_output = io.output;

  std::shared_ptr<tinygs::BackendBuffer> out_image_grad_fp32;
  std::shared_ptr<tinygs::BackendBuffer> out_image_grad_fp16;
  std::shared_ptr<tinygs::BackendBuffer> out_image_grad_convert;
  std::vector<float> out_image_grad_host(total_pixels * 3);
  pcg32 rng(0, 1u);
  for (int i = 0; i < total_pixels * 3; ++i) {
    out_image_grad_host[i] = (static_cast<float>(rng.next_uint(256)) / 255.0f) / total_pixels;
  }

  std::vector<float> out_image_grad_tiled(padded_size);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int pixel_offset = get_linear_index(y, x, pad_width);
      int in_idx = (y * width + x) * 3;
      out_image_grad_tiled[pixel_offset + 0 * channel_stride] = out_image_grad_host[in_idx + 0];
      out_image_grad_tiled[pixel_offset + 1 * channel_stride] = out_image_grad_host[in_idx + 1];
      out_image_grad_tiled[pixel_offset + 2 * channel_stride] = out_image_grad_host[in_idx + 2];
    }
  }

  if (use_fp16) {
    out_image_grad_fp16 = tinygs::create_device_buffer_for<tinygs::float16_t>(*runtime, padded_size, "out_image_grad_fp16");
    out_image_grad_convert = tinygs::create_device_buffer_for<float>(*runtime, padded_size, "out_image_grad_convert");
    tinygs::copy_from_host_async(*runtime, *queue, out_image_grad_convert, out_image_grad_tiled);
    float_to_half_gpu(tinygs::buffer_data<tinygs::float16_t>(out_image_grad_fp16),
                      tinygs::buffer_data<float>(out_image_grad_convert), padded_size, queue.get());
  } else {
    out_image_grad_fp32 = tinygs::create_device_buffer_for<float>(*runtime, padded_size, "out_image_grad_fp32");
    tinygs::copy_from_host_async(*runtime, *queue, out_image_grad_fp32, out_image_grad_tiled);
  }

  std::vector<float> fd_loss_weights;
  if (use_fp16) {
    half_to_float_gpu(tinygs::buffer_data<float>(out_image_grad_convert),
                      tinygs::buffer_data<tinygs::float16_t>(out_image_grad_fp16), padded_size, queue.get());
    std::vector<float> out_image_grad_tiled_effective(padded_size);
    tinygs::copy_to_host_async(*runtime, *queue, out_image_grad_convert, out_image_grad_tiled_effective);
    sync_queue_or_throw("single_gs read fp16 loss weights");
    fd_loss_weights = tiled_to_linear_hwc(out_image_grad_tiled_effective);
  } else {
    fd_loss_weights = tiled_to_linear_hwc(out_image_grad_tiled);
  }

  auto eval_scalar_loss = [&](const Gaussian3d& g) -> double {
    gpu_gaussian->copy_from_host_async(g, queue.get());
    rast->forward(params);

    std::vector<float> pred = get_image_as_linear_hwc();

    double loss = 0.0;
    for (size_t i = 0; i < pred.size(); ++i) {
      loss += static_cast<double>(pred[i]) * static_cast<double>(fd_loss_weights[i]);
    }
    return loss;
  };

  rast->forward(params);

  cv::Mat image(height, width, CV_8UC3);
  std::vector<float> image_host = get_image_as_linear_hwc();
  std::vector<uchar> image_host_8uc3(total_pixels * 3);
  for (int i = 0; i < total_pixels * 3; ++i) {
    image_host_8uc3[i] = static_cast<uchar>(image_host[i] * 255.0f);
  }
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
  auto grad_owned = gpu_gaussian->clone_async(queue.get());
  std::shared_ptr<GPUGaussian3d> grad(std::move(grad_owned));
  void* grad_image_data = use_fp16 ? static_cast<void*>(tinygs::buffer_data<tinygs::float16_t>(out_image_grad_fp16))
                                   : static_cast<void*>(tinygs::buffer_data<float>(out_image_grad_fp32));
  params.grad_output.image = Image(shape, out_data_type, grad_image_data);
  params.gaussians_grad = grad;
  grad->memset_async(0, queue.get());
  rast->backward(params);

  Gaussian3d gaussian_grad;
  grad->copy_to_host_async(gaussian_grad, queue.get());
  size_t dinfo_count = params.densification_info->size_bytes() / sizeof(DensificationInfo);
  std::vector<DensificationInfo> dinfo(dinfo_count);
  tinygs::copy_to_host_async(*runtime, *queue, params.densification_info, dinfo);
  sync_queue_or_throw("single_gs read backward outputs");
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

  std::cout << "=== Densification Info ===" << std::endl;
  for (int i = 0; i < dinfo.size(); ++i) {
    std::cout << "Gaussian " << i << ":" << std::endl;
    std::cout << "  accum_counter: " << dinfo[i].accum_counter << std::endl;
    std::cout << "  accum_grad_mean2d: " << dinfo[i].accum_grad_mean2d << std::endl;
    std::cout << "  accum_absgrad_mean2d: " << dinfo[i].accum_absgrad_mean2d << std::endl;
    std::cout << "  max_radii_screen: " << dinfo[i].max_radii_screen << std::endl;
  }

  if (run_fd_check) {
    std::cout << "=== Finite Difference Gradient Check ===" << std::endl;

    Gaussian3d num_grad;
    num_grad.means.resize(gaussian.means.size(), vec3(0.0f));
    num_grad.scales.resize(gaussian.scales.size(), vec3(0.0f));
    num_grad.rotations.resize(gaussian.rotations.size(), vec4(0.0f));
    num_grad.opacities.resize(gaussian.opacities.size(), 0.0f);
    num_grad.sh0.resize(gaussian.sh0.size(), vec3(0.0f));
    num_grad.sh1.resize(gaussian.sh1.size(), vec3(0.0f));
    num_grad.sh2.resize(gaussian.sh2.size(), vec3(0.0f));
    num_grad.sh3.resize(gaussian.sh3.size(), vec3(0.0f));

    Gaussian3d gauss_work = gaussian;
    const Gaussian3d gauss_base = gaussian;

    auto central_diff = [&](float& x) -> float {
      float old = x;
      x = old + fd_eps;
      double loss_plus = eval_scalar_loss(gauss_work);
      x = old - fd_eps;
      double loss_minus = eval_scalar_loss(gauss_work);
      x = old;
      return static_cast<float>((loss_plus - loss_minus) / (2.0 * static_cast<double>(fd_eps)));
    };

    for (size_t i = 0; i < gauss_work.means.size(); ++i) {
      num_grad.means[i].x = central_diff(gauss_work.means[i].x);
      num_grad.means[i].y = central_diff(gauss_work.means[i].y);
      num_grad.means[i].z = central_diff(gauss_work.means[i].z);
    }
    for (size_t i = 0; i < gauss_work.scales.size(); ++i) {
      num_grad.scales[i].x = central_diff(gauss_work.scales[i].x);
      num_grad.scales[i].y = central_diff(gauss_work.scales[i].y);
      num_grad.scales[i].z = central_diff(gauss_work.scales[i].z);
    }
    for (size_t i = 0; i < gauss_work.rotations.size(); ++i) {
      num_grad.rotations[i].x = central_diff(gauss_work.rotations[i].x);
      num_grad.rotations[i].y = central_diff(gauss_work.rotations[i].y);
      num_grad.rotations[i].z = central_diff(gauss_work.rotations[i].z);
      num_grad.rotations[i].w = central_diff(gauss_work.rotations[i].w);
    }
    for (size_t i = 0; i < gauss_work.opacities.size(); ++i) {
      num_grad.opacities[i] = central_diff(gauss_work.opacities[i]);
    }
    for (size_t i = 0; i < gauss_work.sh0.size(); ++i) {
      num_grad.sh0[i].x = central_diff(gauss_work.sh0[i].x);
      num_grad.sh0[i].y = central_diff(gauss_work.sh0[i].y);
      num_grad.sh0[i].z = central_diff(gauss_work.sh0[i].z);
    }
    for (size_t i = 0; i < gauss_work.sh1.size(); ++i) {
      num_grad.sh1[i].x = central_diff(gauss_work.sh1[i].x);
      num_grad.sh1[i].y = central_diff(gauss_work.sh1[i].y);
      num_grad.sh1[i].z = central_diff(gauss_work.sh1[i].z);
    }
    for (size_t i = 0; i < gauss_work.sh2.size(); ++i) {
      num_grad.sh2[i].x = central_diff(gauss_work.sh2[i].x);
      num_grad.sh2[i].y = central_diff(gauss_work.sh2[i].y);
      num_grad.sh2[i].z = central_diff(gauss_work.sh2[i].z);
    }
    for (size_t i = 0; i < gauss_work.sh3.size(); ++i) {
      num_grad.sh3[i].x = central_diff(gauss_work.sh3[i].x);
      num_grad.sh3[i].y = central_diff(gauss_work.sh3[i].y);
      num_grad.sh3[i].z = central_diff(gauss_work.sh3[i].z);
    }

    // Restore original parameters on GPU for consistency after checking.
    gpu_gaussian->copy_from_host_async(gauss_base, queue.get());

    size_t count = 0;
    double sum_abs = 0.0;
    double sum_rel = 0.0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    std::string max_abs_name;
    std::string max_rel_name;

    auto report = [&](const std::string& name, float ana, float num) {
      float abs_err = std::abs(ana - num);
      float denom = std::max(std::abs(num), 1e-6f);
      float rel_err = abs_err / denom;

      std::cout << std::setw(24) << name
                << " | ana=" << std::setw(12) << ana
                << " num=" << std::setw(12) << num
                << " abs=" << std::setw(12) << abs_err
                << " rel=" << std::setw(12) << rel_err << std::endl;

      sum_abs += abs_err;
      sum_rel += rel_err;
      count++;
      if (abs_err > max_abs) {
        max_abs = abs_err;
        max_abs_name = name;
      }
      if (rel_err > max_rel) {
        max_rel = rel_err;
        max_rel_name = name;
      }
    };

    for (size_t i = 0; i < gaussian_grad.means.size(); ++i) {
      report("means[" + std::to_string(i) + "].x", gaussian_grad.means[i].x, num_grad.means[i].x);
      report("means[" + std::to_string(i) + "].y", gaussian_grad.means[i].y, num_grad.means[i].y);
      report("means[" + std::to_string(i) + "].z", gaussian_grad.means[i].z, num_grad.means[i].z);
    }
    for (size_t i = 0; i < gaussian_grad.scales.size(); ++i) {
      report("scales[" + std::to_string(i) + "].x", gaussian_grad.scales[i].x, num_grad.scales[i].x);
      report("scales[" + std::to_string(i) + "].y", gaussian_grad.scales[i].y, num_grad.scales[i].y);
      report("scales[" + std::to_string(i) + "].z", gaussian_grad.scales[i].z, num_grad.scales[i].z);
    }
    for (size_t i = 0; i < gaussian_grad.rotations.size(); ++i) {
      report("rotations[" + std::to_string(i) + "].w", gaussian_grad.rotations[i].x, num_grad.rotations[i].x);
      report("rotations[" + std::to_string(i) + "].x", gaussian_grad.rotations[i].y, num_grad.rotations[i].y);
      report("rotations[" + std::to_string(i) + "].y", gaussian_grad.rotations[i].z, num_grad.rotations[i].z);
      report("rotations[" + std::to_string(i) + "].z", gaussian_grad.rotations[i].w, num_grad.rotations[i].w);
    }
    for (size_t i = 0; i < gaussian_grad.opacities.size(); ++i) {
      report("opacities[" + std::to_string(i) + "]", gaussian_grad.opacities[i], num_grad.opacities[i]);
    }
    for (size_t i = 0; i < gaussian_grad.sh0.size(); ++i) {
      report("sh0[" + std::to_string(i) + "].r", gaussian_grad.sh0[i].x, num_grad.sh0[i].x);
      report("sh0[" + std::to_string(i) + "].g", gaussian_grad.sh0[i].y, num_grad.sh0[i].y);
      report("sh0[" + std::to_string(i) + "].b", gaussian_grad.sh0[i].z, num_grad.sh0[i].z);
    }
    for (size_t i = 0; i < gaussian_grad.sh1.size(); ++i) {
      report("sh1[" + std::to_string(i) + "].r", gaussian_grad.sh1[i].x, num_grad.sh1[i].x);
      report("sh1[" + std::to_string(i) + "].g", gaussian_grad.sh1[i].y, num_grad.sh1[i].y);
      report("sh1[" + std::to_string(i) + "].b", gaussian_grad.sh1[i].z, num_grad.sh1[i].z);
    }
    for (size_t i = 0; i < gaussian_grad.sh2.size(); ++i) {
      report("sh2[" + std::to_string(i) + "].r", gaussian_grad.sh2[i].x, num_grad.sh2[i].x);
      report("sh2[" + std::to_string(i) + "].g", gaussian_grad.sh2[i].y, num_grad.sh2[i].y);
      report("sh2[" + std::to_string(i) + "].b", gaussian_grad.sh2[i].z, num_grad.sh2[i].z);
    }
    for (size_t i = 0; i < gaussian_grad.sh3.size(); ++i) {
      report("sh3[" + std::to_string(i) + "].r", gaussian_grad.sh3[i].x, num_grad.sh3[i].x);
      report("sh3[" + std::to_string(i) + "].g", gaussian_grad.sh3[i].y, num_grad.sh3[i].y);
      report("sh3[" + std::to_string(i) + "].b", gaussian_grad.sh3[i].z, num_grad.sh3[i].z);
    }

    double mean_abs = count > 0 ? (sum_abs / static_cast<double>(count)) : 0.0;
    double mean_rel = count > 0 ? (sum_rel / static_cast<double>(count)) : 0.0;
    std::cout << "=== FD Summary ===" << std::endl;
    std::cout << "Checked components: " << count << std::endl;
    std::cout << "Mean abs error: " << mean_abs << std::endl;
    std::cout << "Mean rel error: " << mean_rel << std::endl;
    std::cout << "Max abs error: " << max_abs << " at " << max_abs_name << std::endl;
    std::cout << "Max rel error: " << max_rel << " at " << max_rel_name << std::endl;
  }

  return 0;
}
