#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <cxxopts.hpp>

#include "tinygs/common.hpp"
#include "tinygs/core/image.hpp"
#include "tinygs/cuda/common_host.hpp"
#include "tinygs/loss/fused_ssim.hpp"
#include "tinygs/loss/l1.hpp"
#include "tinygs/loss/psnr.hpp"
#include "tinygs/platform/buffer_utils.hpp"
#include "tinygs/platform/runtime_factory.hpp"
#include "tinygs/utils/image_format.hpp"

namespace {

using tinygs::BackendBuffer;
using tinygs::BackendQueue;
using tinygs::BackendRuntime;
using tinygs::DataType;
using tinygs::Image;
using tinygs::ImageShape;
using tinygs::LossContext;

constexpr int k_kernel_radius = 5;
constexpr int k_kernel_size = 2 * k_kernel_radius + 1;
constexpr std::array<double, k_kernel_size> k_gauss = {
  0.001028380123898387,
  0.0075987582094967365,
  0.036000773310661316,
  0.10936068743467331,
  0.21300552785396576,
  0.26601171493530273,
  0.21300552785396576,
  0.10936068743467331,
  0.036000773310661316,
  0.0075987582094967365,
  0.001028380123898387,
};
constexpr double k_ssim_c1 = 0.01 * 0.01;
constexpr double k_ssim_c2 = 0.03 * 0.03;

struct PixelCoord {
  uint32_t idx = 0;
  uint32_t c = 0;
  uint32_t y = 0;
  uint32_t x = 0;
};

struct LossGradResult {
  std::vector<float> loss;
  std::vector<float> grad;
  double total = 0.0;
};

struct ScalarDiff {
  double ref = 0.0;
  double test = 0.0;
  double abs = 0.0;
  double rel = 0.0;
};

struct VectorDiffStats {
  size_t count = 0;
  size_t finite_count = 0;
  size_t nonfinite_ref = 0;
  size_t nonfinite_test = 0;
  size_t nonfinite_err = 0;
  double mean_abs = 0.0;
  double rmse = 0.0;
  double max_abs = 0.0;
  double max_rel = 0.0;
  double p50 = 0.0;
  double p90 = 0.0;
  double p99 = 0.0;
  double cosine = 0.0;
  double ref_at_max = 0.0;
  double test_at_max = 0.0;
  PixelCoord max_coord{};
};

struct TestCase {
  std::string name;
  std::vector<float> pred;
  std::vector<float> target;
};

struct SsimPointStats {
  double mu1 = 0.0;
  double mu2 = 0.0;
  double sigma1_sq = 0.0;
  double sigma2_sq = 0.0;
  double sigma12 = 0.0;
  double ssim = 0.0;
};

inline uint32_t channel_stride(const ImageShape& shape) {
  return shape.padded_width() * shape.padded_height();
}

inline uint32_t tiled_index(const ImageShape& shape, uint32_t c, uint32_t y, uint32_t x) {
  return c * channel_stride(shape) + tinygs::get_linear_index_tiled(y, x, shape.tiled_width());
}

inline void synchronize_queue_or_throw(
    BackendRuntime& runtime,
    BackendQueue* queue,
    const char* op_name) {
  const auto status = runtime.synchronize_queue(*queue);
  if (!status.ok()) {
    throw std::runtime_error(std::string(op_name) + " failed: " + tinygs::to_string(status));
  }
}

std::vector<PixelCoord> make_active_pixels(const ImageShape& shape) {
  std::vector<PixelCoord> active;
  active.reserve(shape.size());
  for (uint32_t c = 0; c < shape.channel; ++c) {
    for (uint32_t y = 0; y < shape.height; ++y) {
      for (uint32_t x = 0; x < shape.width; ++x) {
        active.push_back({tiled_index(shape, c, y, x), c, y, x});
      }
    }
  }
  return active;
}

template <typename T>
double sample_or_zero(const std::vector<T>& image, const ImageShape& shape, uint32_t c, int y, int x) {
  if (x < 0 || y < 0 || x >= static_cast<int>(shape.width) || y >= static_cast<int>(shape.height)) {
    return 0.0;
  }
  return static_cast<double>(image[tiled_index(shape, c, static_cast<uint32_t>(y), static_cast<uint32_t>(x))]);
}

template <typename T>
double sum_active(const std::vector<T>& values, const std::vector<PixelCoord>& active) {
  double s = 0.0;
  for (const auto& p : active) {
    s += static_cast<double>(values[p.idx]);
  }
  return s;
}

double percentile_sorted(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) {
    return 0.0;
  }
  const double clamped_q = std::clamp(q, 0.0, 1.0);
  const double pos = clamped_q * static_cast<double>(sorted.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  if (lo == hi) {
    return sorted[lo];
  }
  const double w = pos - static_cast<double>(lo);
  return sorted[lo] * (1.0 - w) + sorted[hi] * w;
}

ScalarDiff scalar_diff(double ref, double test) {
  ScalarDiff diff;
  diff.ref = ref;
  diff.test = test;
  diff.abs = std::abs(test - ref);
  diff.rel = diff.abs / std::max(std::abs(ref), 1e-12);
  return diff;
}

VectorDiffStats vector_diff_stats(const std::vector<float>& ref,
                                  const std::vector<float>& test,
                                  const std::vector<PixelCoord>& active) {
  CHECK_THROW(ref.size() == test.size());

  VectorDiffStats stats;
  stats.count = active.size();
  if (active.empty()) {
    return stats;
  }

  double sum_abs = 0.0;
  double sum_sq = 0.0;
  double dot = 0.0;
  double ref_norm_sq = 0.0;
  double test_norm_sq = 0.0;
  std::vector<double> abs_errors;
  abs_errors.reserve(active.size());

  bool has_max = false;
  for (const auto& p : active) {
    const double r = static_cast<double>(ref[p.idx]);
    const double t = static_cast<double>(test[p.idx]);
    const double e = t - r;

    if (!std::isfinite(r)) {
      ++stats.nonfinite_ref;
    }
    if (!std::isfinite(t)) {
      ++stats.nonfinite_test;
    }
    if (!std::isfinite(e)) {
      ++stats.nonfinite_err;
      continue;
    }

    ++stats.finite_count;
    const double abs_e = std::abs(e);
    const double rel_e = abs_e / std::max(std::abs(r), 1e-12);
    sum_abs += abs_e;
    sum_sq += abs_e * abs_e;
    dot += r * t;
    ref_norm_sq += r * r;
    test_norm_sq += t * t;
    abs_errors.push_back(abs_e);

    if (!has_max || abs_e > stats.max_abs) {
      has_max = true;
      stats.max_abs = abs_e;
      stats.max_rel = rel_e;
      stats.ref_at_max = r;
      stats.test_at_max = t;
      stats.max_coord = p;
    }
  }

  if (stats.finite_count == 0) {
    return stats;
  }

  std::sort(abs_errors.begin(), abs_errors.end());
  stats.mean_abs = sum_abs / static_cast<double>(stats.finite_count);
  stats.rmse = std::sqrt(sum_sq / static_cast<double>(stats.finite_count));
  stats.p50 = percentile_sorted(abs_errors, 0.50);
  stats.p90 = percentile_sorted(abs_errors, 0.90);
  stats.p99 = percentile_sorted(abs_errors, 0.99);
  stats.cosine = dot / (std::sqrt(ref_norm_sq) * std::sqrt(test_norm_sq) + 1e-30);
  return stats;
}

void print_scalar_report(const std::string& title, const ScalarDiff& d) {
  std::cout << "  [" << title << "]"
            << " cpu=" << d.ref
            << " cuda=" << d.test
            << " abs=" << d.abs
            << " rel=" << d.rel
            << "\n";
}

void print_vector_report(const std::string& title, const VectorDiffStats& s) {
  std::cout << "  [" << title << "]"
            << " n=" << s.count
            << " finite=" << s.finite_count
            << " max_abs=" << s.max_abs
            << " mean_abs=" << s.mean_abs
            << " rmse=" << s.rmse
            << " p50=" << s.p50
            << " p90=" << s.p90
            << " p99=" << s.p99
            << " max_rel=" << s.max_rel
            << " cosine=" << s.cosine
            << " worst(c,y,x)=(" << s.max_coord.c << "," << s.max_coord.y << "," << s.max_coord.x << ")"
            << " ref=" << s.ref_at_max
            << " cuda=" << s.test_at_max
            << " nonfinite(ref,test,err)=("
            << s.nonfinite_ref << "," << s.nonfinite_test << "," << s.nonfinite_err << ")"
            << "\n";
}

void fill_random_valid(std::vector<float>& image, const ImageShape& shape,
                       std::mt19937& rng, float lo, float hi) {
  std::uniform_real_distribution<float> dist(lo, hi);
  for (uint32_t c = 0; c < shape.channel; ++c) {
    for (uint32_t y = 0; y < shape.height; ++y) {
      for (uint32_t x = 0; x < shape.width; ++x) {
        image[tiled_index(shape, c, y, x)] = dist(rng);
      }
    }
  }
}

void add_noise_valid(std::vector<float>& image, const ImageShape& shape,
                     std::mt19937& rng, float lo, float hi) {
  std::uniform_real_distribution<float> dist(lo, hi);
  for (uint32_t c = 0; c < shape.channel; ++c) {
    for (uint32_t y = 0; y < shape.height; ++y) {
      for (uint32_t x = 0; x < shape.width; ++x) {
        image[tiled_index(shape, c, y, x)] += dist(rng);
      }
    }
  }
}

std::vector<TestCase> build_test_cases(const ImageShape& shape, uint32_t seed, int random_cases) {
  std::mt19937 rng(seed);
  const size_t padded_size = shape.padded_size();

  std::vector<TestCase> cases;
  cases.reserve(static_cast<size_t>(std::max(random_cases, 0) + 4));

  {
    TestCase t;
    t.name = "random_uniform";
    t.pred.assign(padded_size, 0.0f);
    t.target.assign(padded_size, 0.0f);
    fill_random_valid(t.pred, shape, rng, 0.0f, 1.0f);
    fill_random_valid(t.target, shape, rng, 0.0f, 1.0f);
    cases.push_back(std::move(t));
  }
  {
    TestCase t;
    t.name = "clamp_stress";
    t.pred.assign(padded_size, 0.0f);
    t.target.assign(padded_size, 0.0f);
    fill_random_valid(t.pred, shape, rng, -0.5f, 1.5f);
    fill_random_valid(t.target, shape, rng, 0.0f, 1.0f);
    cases.push_back(std::move(t));
  }
  {
    TestCase t;
    t.name = "near_match";
    t.pred.assign(padded_size, 0.0f);
    t.target.assign(padded_size, 0.0f);
    fill_random_valid(t.target, shape, rng, 0.0f, 1.0f);
    t.pred = t.target;
    add_noise_valid(t.pred, shape, rng, -2e-3f, 2e-3f);
    cases.push_back(std::move(t));
  }
  {
    TestCase t;
    t.name = "identical";
    t.pred.assign(padded_size, 0.0f);
    t.target.assign(padded_size, 0.0f);
    fill_random_valid(t.target, shape, rng, 0.0f, 1.0f);
    t.pred = t.target;
    cases.push_back(std::move(t));
  }

  for (int i = 0; i < random_cases; ++i) {
    TestCase t;
    t.name = "random_extra_" + std::to_string(i);
    t.pred.assign(padded_size, 0.0f);
    t.target.assign(padded_size, 0.0f);
    fill_random_valid(t.pred, shape, rng, -0.25f, 1.25f);
    fill_random_valid(t.target, shape, rng, 0.0f, 1.0f);
    cases.push_back(std::move(t));
  }

  return cases;
}

std::vector<float> quantize_to_fp16(
    BackendRuntime& runtime,
    BackendQueue* queue,
    const std::vector<float>& values) {
  const size_t n = values.size();

  auto input = tinygs::create_device_buffer_for<float>(runtime, n, "quantize_input");
  tinygs::copy_from_host_async(runtime, *queue, input, values);

  auto fp16 = tinygs::create_device_buffer_for<tinygs::float16_t>(runtime, n, "quantize_fp16");
  tinygs::float_to_half_gpu(tinygs::buffer_data<tinygs::float16_t>(fp16),
                             tinygs::buffer_data<float>(input), static_cast<int>(n), queue);

  auto back = tinygs::create_device_buffer_for<float>(runtime, n, "quantize_back");
  tinygs::half_to_float_gpu(tinygs::buffer_data<float>(back),
                            tinygs::buffer_data<tinygs::float16_t>(fp16), static_cast<int>(n), queue);

  std::vector<float> out(n, 0.0f);
  tinygs::copy_to_host_async(runtime, *queue, back, out);
  synchronize_queue_or_throw(runtime, queue, "quantize_to_fp16");
  return out;
}

SsimPointStats compute_ssim_point(const std::vector<float>& pred,
                                  const std::vector<float>& target,
                                  const ImageShape& shape,
                                  uint32_t c, int y, int x) {
  SsimPointStats stats;
  double ex2 = 0.0;
  double ey2 = 0.0;
  double exy = 0.0;

  for (int ky = -k_kernel_radius; ky <= k_kernel_radius; ++ky) {
    const double wy = k_gauss[ky + k_kernel_radius];
    for (int kx = -k_kernel_radius; kx <= k_kernel_radius; ++kx) {
      const double wx = k_gauss[kx + k_kernel_radius];
      const double w = wy * wx;

      const double p = sample_or_zero(pred, shape, c, y + ky, x + kx);
      const double t = sample_or_zero(target, shape, c, y + ky, x + kx);
      stats.mu1 += p * w;
      stats.mu2 += t * w;
      ex2 += p * p * w;
      ey2 += t * t * w;
      exy += p * t * w;
    }
  }

  const double mu1_sq = stats.mu1 * stats.mu1;
  const double mu2_sq = stats.mu2 * stats.mu2;
  stats.sigma1_sq = ex2 - mu1_sq;
  stats.sigma2_sq = ey2 - mu2_sq;
  stats.sigma12 = exy - stats.mu1 * stats.mu2;

  const double a = mu1_sq + mu2_sq + k_ssim_c1;
  const double b = stats.sigma1_sq + stats.sigma2_sq + k_ssim_c2;
  const double c_term = 2.0 * stats.mu1 * stats.mu2 + k_ssim_c1;
  const double d_term = 2.0 * stats.sigma12 + k_ssim_c2;
  stats.ssim = (c_term * d_term) / (a * b);
  return stats;
}

double compute_ssim_total_forward_cpu(const std::vector<float>& pred,
                                      const std::vector<float>& target,
                                      const ImageShape& shape,
                                      float scale) {
  const double actual_scale = static_cast<double>(scale) / static_cast<double>(shape.size());
  double total = 0.0;
  for (uint32_t c = 0; c < shape.channel; ++c) {
    for (uint32_t y = 0; y < shape.height; ++y) {
      for (uint32_t x = 0; x < shape.width; ++x) {
        const SsimPointStats point = compute_ssim_point(pred, target, shape, c, static_cast<int>(y), static_cast<int>(x));
        total += (1.0 - point.ssim) * actual_scale;
      }
    }
  }
  return total;
}

LossGradResult compute_l1_cpu(const std::vector<float>& pred,
                              const std::vector<float>& target,
                              const ImageShape& shape,
                              float scale) {
  CHECK_THROW(pred.size() == target.size());

  LossGradResult out;
  out.loss.assign(pred.size(), 0.0f);
  out.grad.assign(pred.size(), 0.0f);

  const double actual_scale = static_cast<double>(scale) / static_cast<double>(shape.size());
  for (size_t i = 0; i < pred.size(); ++i) {
    const double p_raw = static_cast<double>(pred[i]);
    const double p = std::clamp(p_raw, 0.0, 1.0);
    const double t = static_cast<double>(target[i]);
    const double diff = p - t;
    const double sat_deriv = (p_raw > 0.0 && p_raw < 1.0) ? 1.0 : 0.0;
    const double g = std::copysign(1.0, diff) * sat_deriv;

    out.loss[i] = static_cast<float>(std::abs(diff) * actual_scale);
    out.grad[i] = static_cast<float>(g * actual_scale);
  }
  return out;
}

double compute_psnr_cpu(const std::vector<float>& pred,
                        const std::vector<float>& target,
                        const ImageShape& shape) {
  CHECK_THROW(shape.channel > 0);

  double psnr_sum = 0.0;
  const double denom = static_cast<double>(shape.width) * static_cast<double>(shape.height);

  for (uint32_t c = 0; c < shape.channel; ++c) {
    double sq_err_sum = 0.0;
    for (uint32_t y = 0; y < shape.height; ++y) {
      for (uint32_t x = 0; x < shape.width; ++x) {
        const uint32_t idx = tiled_index(shape, c, y, x);
        const double p = std::clamp(static_cast<double>(pred[idx]), 0.0, 1.0);
        const double t = static_cast<double>(target[idx]);
        const double d = p - t;
        sq_err_sum += d * d;
      }
    }
    const double mse = sq_err_sum / denom;
    psnr_sum += (mse == 0.0) ? 100.0 : -10.0 * std::log10(mse);
  }
  return psnr_sum / static_cast<double>(shape.channel);
}

LossGradResult compute_ssim_cpu(const std::vector<float>& pred,
                                const std::vector<float>& target,
                                const ImageShape& shape,
                                float scale) {
  CHECK_THROW(pred.size() == target.size());

  LossGradResult out;
  out.loss.assign(pred.size(), 0.0f);
  out.grad.assign(pred.size(), 0.0f);

  std::vector<double> dm_dmu1(pred.size(), 0.0);
  std::vector<double> dm_dsigma1_sq(pred.size(), 0.0);
  std::vector<double> dm_dsigma12(pred.size(), 0.0);

  const double actual_scale = static_cast<double>(scale) / static_cast<double>(shape.size());

  for (uint32_t c = 0; c < shape.channel; ++c) {
    for (uint32_t y = 0; y < shape.height; ++y) {
      for (uint32_t x = 0; x < shape.width; ++x) {
        const uint32_t idx = tiled_index(shape, c, y, x);
        const SsimPointStats s = compute_ssim_point(pred, target, shape, c, static_cast<int>(y), static_cast<int>(x));
        const double loss = (1.0 - s.ssim) * actual_scale;
        out.loss[idx] = static_cast<float>(loss);
        out.total += loss;

        const double mu1_sq = s.mu1 * s.mu1;
        const double mu2_sq = s.mu2 * s.mu2;
        const double a = mu1_sq + mu2_sq + k_ssim_c1;
        const double b = s.sigma1_sq + s.sigma2_sq + k_ssim_c2;
        const double c_term = 2.0 * s.mu1 * s.mu2 + k_ssim_c1;
        const double d_term = 2.0 * s.sigma12 + k_ssim_c2;

        dm_dmu1[idx] = (
          (s.mu2 * 2.0 * d_term) / (a * b)
          - (s.mu2 * 2.0 * c_term) / (a * b)
          - (s.mu1 * 2.0 * c_term * d_term) / (a * a * b)
          + (s.mu1 * 2.0 * c_term * d_term) / (a * b * b)
        );
        dm_dsigma1_sq[idx] = (-c_term * d_term) / (a * b * b);
        dm_dsigma12[idx] = (2.0 * c_term) / (a * b);
      }
    }
  }

  for (uint32_t c = 0; c < shape.channel; ++c) {
    for (uint32_t y = 0; y < shape.height; ++y) {
      for (uint32_t x = 0; x < shape.width; ++x) {
        double conv_mu = 0.0;
        double conv_sigma1 = 0.0;
        double conv_sigma12 = 0.0;
        for (int ky = -k_kernel_radius; ky <= k_kernel_radius; ++ky) {
          const double wy = k_gauss[ky + k_kernel_radius];
          for (int kx = -k_kernel_radius; kx <= k_kernel_radius; ++kx) {
            const double wx = k_gauss[kx + k_kernel_radius];
            const double w = wy * wx;
            conv_mu += sample_or_zero(dm_dmu1, shape, c, static_cast<int>(y) + ky, static_cast<int>(x) + kx) * w;
            conv_sigma1 += sample_or_zero(dm_dsigma1_sq, shape, c, static_cast<int>(y) + ky, static_cast<int>(x) + kx) * w;
            conv_sigma12 += sample_or_zero(dm_dsigma12, shape, c, static_cast<int>(y) + ky, static_cast<int>(x) + kx) * w;
          }
        }

        const uint32_t idx = tiled_index(shape, c, y, x);
        const double p = static_cast<double>(pred[idx]);
        const double t = static_cast<double>(target[idx]);
        const double grad = -(conv_mu + 2.0 * p * conv_sigma1 + t * conv_sigma12) * actual_scale;
        out.grad[idx] = static_cast<float>(grad);
      }
    }
  }

  return out;
}

template <typename LossOp>
LossGradResult run_cuda_loss_fp32(
    BackendRuntime& runtime,
  BackendQueue* queue,
    const std::vector<float>& pred,
    const std::vector<float>& target,
    const ImageShape& shape,
    float scale) {
  const size_t n = shape.padded_size();

  auto pred_d = tinygs::create_device_buffer_for<float>(runtime, n, "pred");
  auto target_d = tinygs::create_device_buffer_for<float>(runtime, n, "target");
  auto loss_d = tinygs::create_device_buffer_for<float>(runtime, n, "loss");
  auto grad_d = tinygs::create_device_buffer_for<float>(runtime, n, "grad");

  tinygs::copy_from_host_async(runtime, *queue, pred_d, pred);
  tinygs::copy_from_host_async(runtime, *queue, target_d, target);
  tinygs::fill_buffer_zero_async(runtime, *queue, loss_d);
  tinygs::fill_buffer_zero_async(runtime, *queue, grad_d);

  LossContext ctx;
  ctx.pred = Image(shape, DataType::Float32, tinygs::buffer_data<float>(pred_d));
  ctx.target = Image(shape, DataType::Float32, tinygs::buffer_data<float>(target_d));
  ctx.loss = Image(shape, DataType::Float32, tinygs::buffer_data<float>(loss_d));
  ctx.grad = Image(shape, DataType::Float32, tinygs::buffer_data<float>(grad_d));
  ctx.queue = queue;

  LossOp op(runtime);
  op.evaluate(ctx, scale);

  LossGradResult out;
  out.loss.assign(n, 0.0f);
  out.grad.assign(n, 0.0f);
  tinygs::copy_to_host_async(runtime, *queue, loss_d, out.loss);
  tinygs::copy_to_host_async(runtime, *queue, grad_d, out.grad);
  synchronize_queue_or_throw(runtime, queue, "run_cuda_loss_fp32");
  return out;
}

template <typename LossOp>
LossGradResult run_cuda_loss_fp16(
    BackendRuntime& runtime,
  BackendQueue* queue,
    const std::vector<float>& pred,
    const std::vector<float>& target,
    const ImageShape& shape,
    float scale) {
  const int n = static_cast<int>(shape.padded_size());

  auto pred_f = tinygs::create_device_buffer_for<float>(runtime, static_cast<size_t>(n), "pred_f");
  auto target_f = tinygs::create_device_buffer_for<float>(runtime, static_cast<size_t>(n), "target_f");
  tinygs::copy_from_host_async(runtime, *queue, pred_f, pred);
  tinygs::copy_from_host_async(runtime, *queue, target_f, target);

  auto pred_h = tinygs::create_device_buffer_for<tinygs::float16_t>(runtime, static_cast<size_t>(n), "pred_h");
  auto target_h = tinygs::create_device_buffer_for<tinygs::float16_t>(runtime, static_cast<size_t>(n), "target_h");
  auto loss_h = tinygs::create_device_buffer_for<tinygs::float16_t>(runtime, static_cast<size_t>(n), "loss_h");
  auto grad_h = tinygs::create_device_buffer_for<tinygs::float16_t>(runtime, static_cast<size_t>(n), "grad_h");

  tinygs::float_to_half_gpu(tinygs::buffer_data<tinygs::float16_t>(pred_h),
                            tinygs::buffer_data<float>(pred_f), n, queue);
  tinygs::float_to_half_gpu(tinygs::buffer_data<tinygs::float16_t>(target_h),
                            tinygs::buffer_data<float>(target_f), n, queue);
  tinygs::fill_buffer_zero_async(runtime, *queue, loss_h);
  tinygs::fill_buffer_zero_async(runtime, *queue, grad_h);

  LossContext ctx;
  ctx.pred = Image(shape, DataType::Float16, tinygs::buffer_data<tinygs::float16_t>(pred_h));
  ctx.target = Image(shape, DataType::Float16, tinygs::buffer_data<tinygs::float16_t>(target_h));
  ctx.loss = Image(shape, DataType::Float16, tinygs::buffer_data<tinygs::float16_t>(loss_h));
  ctx.grad = Image(shape, DataType::Float16, tinygs::buffer_data<tinygs::float16_t>(grad_h));
  ctx.queue = queue;

  LossOp op(runtime);
  op.evaluate(ctx, scale);

  auto loss_f = tinygs::create_device_buffer_for<float>(runtime, static_cast<size_t>(n), "loss_f");
  auto grad_f = tinygs::create_device_buffer_for<float>(runtime, static_cast<size_t>(n), "grad_f");
  tinygs::half_to_float_gpu(tinygs::buffer_data<float>(loss_f),
                            tinygs::buffer_data<tinygs::float16_t>(loss_h), n, queue);
  tinygs::half_to_float_gpu(tinygs::buffer_data<float>(grad_f),
                            tinygs::buffer_data<tinygs::float16_t>(grad_h), n, queue);

  LossGradResult out;
  out.loss.assign(static_cast<size_t>(n), 0.0f);
  out.grad.assign(static_cast<size_t>(n), 0.0f);
  tinygs::copy_to_host_async(runtime, *queue, loss_f, out.loss);
  tinygs::copy_to_host_async(runtime, *queue, grad_f, out.grad);
  synchronize_queue_or_throw(runtime, queue, "run_cuda_loss_fp16");
  return out;
}

double run_cuda_psnr_fp32(
    BackendRuntime& runtime,
  BackendQueue* queue,
    const std::vector<float>& pred,
    const std::vector<float>& target,
    const ImageShape& shape) {
  const size_t n = shape.padded_size();
  auto pred_d = tinygs::create_device_buffer_for<float>(runtime, n, "pred");
  auto target_d = tinygs::create_device_buffer_for<float>(runtime, n, "target");
  tinygs::copy_from_host_async(runtime, *queue, pred_d, pred);
  tinygs::copy_from_host_async(runtime, *queue, target_d, target);
  synchronize_queue_or_throw(runtime, queue, "run_cuda_psnr_fp32 uploads");

  tinygs::PsnrMetric metric(runtime);
  return static_cast<double>(metric.evaluate(
    Image(shape, DataType::Float32, tinygs::buffer_data<float>(pred_d)),
    Image(shape, DataType::Float32, tinygs::buffer_data<float>(target_d))
  ));
}

double run_cuda_psnr_fp16(
    BackendRuntime& runtime,
  BackendQueue* queue,
    const std::vector<float>& pred,
    const std::vector<float>& target,
    const ImageShape& shape) {
  const int n = static_cast<int>(shape.padded_size());

  auto pred_f = tinygs::create_device_buffer_for<float>(runtime, static_cast<size_t>(n), "pred_f");
  auto target_f = tinygs::create_device_buffer_for<float>(runtime, static_cast<size_t>(n), "target_f");
  tinygs::copy_from_host_async(runtime, *queue, pred_f, pred);
  tinygs::copy_from_host_async(runtime, *queue, target_f, target);

  auto pred_h = tinygs::create_device_buffer_for<tinygs::float16_t>(runtime, static_cast<size_t>(n), "pred_h");
  auto target_h = tinygs::create_device_buffer_for<tinygs::float16_t>(runtime, static_cast<size_t>(n), "target_h");
  tinygs::float_to_half_gpu(tinygs::buffer_data<tinygs::float16_t>(pred_h),
                            tinygs::buffer_data<float>(pred_f), n, queue);
  tinygs::float_to_half_gpu(tinygs::buffer_data<tinygs::float16_t>(target_h),
                            tinygs::buffer_data<float>(target_f), n, queue);
  synchronize_queue_or_throw(runtime, queue, "run_cuda_psnr_fp16 uploads");

  tinygs::PsnrMetric metric(runtime);
  return static_cast<double>(metric.evaluate(
    Image(shape, DataType::Float16, tinygs::buffer_data<tinygs::float16_t>(pred_h)),
    Image(shape, DataType::Float16, tinygs::buffer_data<tinygs::float16_t>(target_h))
  ));
}

VectorDiffStats ssim_fd_check(const std::vector<float>& pred,
                              const std::vector<float>& target,
                              const ImageShape& shape,
                              float scale,
                              const std::vector<float>& analytic_grad,
                              const std::vector<PixelCoord>& active,
                              uint32_t seed,
                              int samples,
                              double eps) {
  const size_t n = pred.size();
  std::vector<float> pred_work = pred;
  std::vector<float> fd_grad(n, 0.0f);
  std::vector<float> analytic_sampled(n, 0.0f);

  std::vector<PixelCoord> selected = active;
  std::mt19937 rng(seed);
  std::shuffle(selected.begin(), selected.end(), rng);
  if (samples >= 0 && static_cast<size_t>(samples) < selected.size()) {
    selected.resize(static_cast<size_t>(samples));
  }

  for (const auto& p : selected) {
    const float v = pred_work[p.idx];
    pred_work[p.idx] = v + static_cast<float>(eps);
    const double plus = compute_ssim_total_forward_cpu(pred_work, target, shape, scale);
    pred_work[p.idx] = v - static_cast<float>(eps);
    const double minus = compute_ssim_total_forward_cpu(pred_work, target, shape, scale);
    pred_work[p.idx] = v;

    fd_grad[p.idx] = static_cast<float>((plus - minus) / (2.0 * eps));
    analytic_sampled[p.idx] = analytic_grad[p.idx];
  }

  return vector_diff_stats(analytic_sampled, fd_grad, selected);
}

} // namespace

int main(int argc, char** argv) {
  cxxopts::Options options("loss_psnr_check", "Numerical checker for L1 / SSIM / PSNR (CPU reference vs CUDA)");
  options.add_options()
    ("h,help", "Print help")
    ("width", "Image width", cxxopts::value<int>()->default_value("37"))
    ("height", "Image height", cxxopts::value<int>()->default_value("29"))
    ("seed", "Random seed", cxxopts::value<uint32_t>()->default_value("42"))
    ("scale", "Loss scale passed into CUDA loss evaluate()", cxxopts::value<float>()->default_value("1.0"))
    ("random-cases", "Additional random cases", cxxopts::value<int>()->default_value("2"))
    ("fp32", "Run fp32 path", cxxopts::value<bool>()->default_value("true"))
    ("fp16", "Run fp16 path", cxxopts::value<bool>()->default_value("true"))
    ("fd-check", "Run CPU SSIM backward finite-difference spot-check", cxxopts::value<bool>()->default_value("true"))
    ("fd-samples", "Number of pixels sampled for SSIM finite-difference check", cxxopts::value<int>()->default_value("32"))
    ("fd-eps", "Finite-difference epsilon for SSIM CPU check", cxxopts::value<double>()->default_value("1e-3"));

  const auto parsed = options.parse(argc, argv);
  if (parsed.count("help")) {
    std::cout << options.help() << "\n";
    return 0;
  }

  const int width = parsed["width"].as<int>();
  const int height = parsed["height"].as<int>();
  const uint32_t seed = parsed["seed"].as<uint32_t>();
  const float scale = parsed["scale"].as<float>();
  const int random_cases = parsed["random-cases"].as<int>();
  const bool run_fp32 = parsed["fp32"].as<bool>();
  const bool run_fp16 = parsed["fp16"].as<bool>();
  const bool run_fd_check = parsed["fd-check"].as<bool>();
  const int fd_samples = parsed["fd-samples"].as<int>();
  const double fd_eps = parsed["fd-eps"].as<double>();

  if (!run_fp32 && !run_fp16) {
    throw std::runtime_error("At least one of --fp32 / --fp16 must be enabled.");
  }
  if (width <= 0 || height <= 0) {
    throw std::runtime_error("width and height must be positive.");
  }

  tinygs::BackendConfig backend_cfg;
  backend_cfg.type = tinygs::BackendType::Cuda;
  backend_cfg.device = 0;
  auto rt_result = tinygs::create_backend_runtime(backend_cfg);
  if (!rt_result.ok()) {
    std::cerr << "Failed to create runtime: " << tinygs::to_string(rt_result.error()) << "\n";
    return 1;
  }
  auto runtime = rt_result.value();
  auto q_result = runtime->create_queue({});
  if (!q_result.ok()) {
    std::cerr << "Failed to create queue: " << tinygs::to_string(q_result.error()) << "\n";
    return 1;
  }
  auto queue = q_result.value();

  ImageShape shape;
  shape.width = static_cast<uint32_t>(width);
  shape.height = static_cast<uint32_t>(height);
  shape.channel = 3;

  const std::vector<PixelCoord> active = make_active_pixels(shape);
  const auto test_cases = build_test_cases(shape, seed, random_cases);

  std::cout << std::scientific << std::setprecision(9);
  std::cout << "=== loss_psnr_check ===\n";
  std::cout << "shape: " << tinygs::to_string(shape)
            << ", padded_size=" << shape.padded_size()
            << ", active_size=" << active.size()
            << ", seed=" << seed
            << ", scale=" << scale
            << ", random_cases=" << random_cases
            << ", fd_check=" << (run_fd_check ? "true" : "false")
            << ", fd_samples=" << fd_samples
            << ", fd_eps=" << fd_eps
            << "\n";

  std::vector<std::pair<std::string, bool>> modes;
  if (run_fp32) {
    modes.emplace_back("fp32", false);
  }
  if (run_fp16) {
    modes.emplace_back("fp16", true);
  }

  for (const auto& mode : modes) {
    const std::string mode_name = mode.first;
    const bool use_fp16 = mode.second;
    std::cout << "\n=== Mode: " << mode_name << " ===\n";

    for (size_t case_idx = 0; case_idx < test_cases.size(); ++case_idx) {
      const TestCase& t = test_cases[case_idx];
      std::vector<float> pred = t.pred;
      std::vector<float> target = t.target;
      if (use_fp16) {
          pred = quantize_to_fp16(*runtime, queue.get(), pred);
          target = quantize_to_fp16(*runtime, queue.get(), target);
      }

      std::cout << "\n--- Case: " << t.name << " ---\n";

      LossGradResult cpu_l1 = compute_l1_cpu(pred, target, shape, scale);
      cpu_l1.total = sum_active(cpu_l1.loss, active);
      LossGradResult cuda_l1 = use_fp16
        ? run_cuda_loss_fp16<tinygs::L1Loss>(*runtime, queue.get(), pred, target, shape, scale)
        : run_cuda_loss_fp32<tinygs::L1Loss>(*runtime, queue.get(), pred, target, shape, scale);
      cuda_l1.total = sum_active(cuda_l1.loss, active);

      const ScalarDiff l1_total_diff = scalar_diff(cpu_l1.total, cuda_l1.total);
      const VectorDiffStats l1_loss_stats = vector_diff_stats(cpu_l1.loss, cuda_l1.loss, active);
      const VectorDiffStats l1_grad_stats = vector_diff_stats(cpu_l1.grad, cuda_l1.grad, active);

      print_scalar_report("l1.total", l1_total_diff);
      print_vector_report("l1.loss_map", l1_loss_stats);
      print_vector_report("l1.grad_map", l1_grad_stats);

      LossGradResult cpu_ssim = compute_ssim_cpu(pred, target, shape, scale);
      cpu_ssim.total = sum_active(cpu_ssim.loss, active);
      LossGradResult cuda_ssim = use_fp16
        ? run_cuda_loss_fp16<tinygs::FusedSSIMLoss>(*runtime, queue.get(), pred, target, shape, scale)
        : run_cuda_loss_fp32<tinygs::FusedSSIMLoss>(*runtime, queue.get(), pred, target, shape, scale);
      cuda_ssim.total = sum_active(cuda_ssim.loss, active);

      const ScalarDiff ssim_total_diff = scalar_diff(cpu_ssim.total, cuda_ssim.total);
      const VectorDiffStats ssim_loss_stats = vector_diff_stats(cpu_ssim.loss, cuda_ssim.loss, active);
      const VectorDiffStats ssim_grad_stats = vector_diff_stats(cpu_ssim.grad, cuda_ssim.grad, active);

      print_scalar_report("ssim.total", ssim_total_diff);
      print_vector_report("ssim.loss_map", ssim_loss_stats);
      print_vector_report("ssim.grad_map", ssim_grad_stats);

      if (run_fd_check && case_idx == 0) {
        const VectorDiffStats fd_stats = ssim_fd_check(
          pred,
          target,
          shape,
          scale,
          cpu_ssim.grad,
          active,
          seed + static_cast<uint32_t>(use_fp16 ? 1001 : 17),
          fd_samples,
          fd_eps
        );
        print_vector_report("ssim.cpu_grad_vs_fd(analytic_vs_fd)", fd_stats);
      }

      const double cpu_psnr = compute_psnr_cpu(pred, target, shape);
      const double cuda_psnr = use_fp16
        ? run_cuda_psnr_fp16(*runtime, queue.get(), pred, target, shape)
        : run_cuda_psnr_fp32(*runtime, queue.get(), pred, target, shape);
      const ScalarDiff psnr_diff = scalar_diff(cpu_psnr, cuda_psnr);
      print_scalar_report("psnr", psnr_diff);
    }
  }

  std::cout << "\nDone.\n";
  return 0;
}
