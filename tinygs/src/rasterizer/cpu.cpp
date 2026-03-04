/// @file cpu.cpp
/// @brief CPU reference rasterizer — naive, correct, deterministic.
///
/// Follows docs/KHR_gaussian_splatting.md and docs/MATH.md exactly.
/// All GPU data is copied to host, processed, and results copied back.
/// Output image uses the project-standard CHW-tiled layout (8×8 tiles).

#include "tinygs/rasterizer/cpu.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "tinygs/core/gaussian.hpp"
#include "tinygs/cuda/common_host.hpp"

namespace tinygs {

// ────────────────────────── SH Constants (Condon-Shortley phase) ──────────────────────────
// Matches docs/KHR_gaussian_splatting.md Appendix A.
static constexpr float kSH_C0 = 0.28209479177387814f;
static constexpr float kSH_C1 = 0.4886025119029199f;
static constexpr float kSH_C2[] = {
  1.0925484305920792f,
  -1.0925484305920792f,
  0.31539156525252005f,
  -1.0925484305920792f,
  0.5462742152960396f,
};
static constexpr float kSH_C3[] = {
  -0.5900435899266435f,
  2.890611442640554f,
  -0.4570457994644658f,
  0.3731763325901154f,
  -0.4570457994644658f,
  1.445305721320277f,
  -0.5900435899266435f,
};

// ────────────────────────── Helper: read SH from CPU Gaussian3d ──────────────────────────
// Gaussian3d stores SH in AoS: e.g. sh1 = [G0_c0, G0_c1, G0_c2, G1_c0, ...], each vec3(R,G,B).
// For degree d, each gaussian has kSHDegreeNumCoeffs[d] coefficients.

static vec3 read_sh_cpu(const std::vector<vec3>& sh_buf, int n_coeffs, int gaussian_idx, int coeff_idx) {
  return sh_buf[gaussian_idx * n_coeffs + coeff_idx];
}

// ────────────────────────── SH evaluation ──────────────────────────
/// Evaluate spherical harmonics for a given direction.  Returns the final RGB color (clamped ≥ 0).
/// @param dir Normalized world-space direction from camera to gaussian center.
/// @param sh_degree Current active SH degree (0–3).
static vec3 evaluate_sh(
    int gaussian_idx,
    int sh_degree,
    const vec3& dir,
    const Gaussian3d& gs) {

  // Degree 0: DC term
  vec3 dc = gs.sh0[gaussian_idx];
  vec3 result = kSH_C0 * dc;

  if (sh_degree >= 1) {
    float x = dir.x, y = dir.y, z = dir.z;
    int nc1 = kSHDegreeNumCoeffs[1]; // 3
    vec3 s0 = read_sh_cpu(gs.sh1, nc1, gaussian_idx, 0);
    vec3 s1 = read_sh_cpu(gs.sh1, nc1, gaussian_idx, 1);
    vec3 s2 = read_sh_cpu(gs.sh1, nc1, gaussian_idx, 2);
    // Y_{1,-1} = -C1 * y,  Y_{1,0} = C1 * z,  Y_{1,1} = -C1 * x
    result += -kSH_C1 * y * s0
            +  kSH_C1 * z * s1
            + -kSH_C1 * x * s2;

    if (sh_degree >= 2) {
      float xx = x * x, yy = y * y, zz = z * z;
      float xy = x * y, yz = y * z, xz = x * z;
      int nc2 = kSHDegreeNumCoeffs[2]; // 5
      vec3 s3 = read_sh_cpu(gs.sh2, nc2, gaussian_idx, 0);
      vec3 s4 = read_sh_cpu(gs.sh2, nc2, gaussian_idx, 1);
      vec3 s5 = read_sh_cpu(gs.sh2, nc2, gaussian_idx, 2);
      vec3 s6 = read_sh_cpu(gs.sh2, nc2, gaussian_idx, 3);
      vec3 s7 = read_sh_cpu(gs.sh2, nc2, gaussian_idx, 4);
      result += kSH_C2[0] * xy * s3
              + kSH_C2[1] * yz * s4
              + kSH_C2[2] * (2.0f * zz - xx - yy) * s5
              + kSH_C2[3] * xz * s6
              + kSH_C2[4] * (xx - yy) * s7;

      if (sh_degree >= 3) {
        int nc3 = kSHDegreeNumCoeffs[3]; // 7
        vec3 s8  = read_sh_cpu(gs.sh3, nc3, gaussian_idx, 0);
        vec3 s9  = read_sh_cpu(gs.sh3, nc3, gaussian_idx, 1);
        vec3 s10 = read_sh_cpu(gs.sh3, nc3, gaussian_idx, 2);
        vec3 s11 = read_sh_cpu(gs.sh3, nc3, gaussian_idx, 3);
        vec3 s12 = read_sh_cpu(gs.sh3, nc3, gaussian_idx, 4);
        vec3 s13 = read_sh_cpu(gs.sh3, nc3, gaussian_idx, 5);
        vec3 s14 = read_sh_cpu(gs.sh3, nc3, gaussian_idx, 6);
        result += kSH_C3[0] * y * (3.0f * xx - yy)            * s8
                + kSH_C3[1] * xy * z                           * s9
                + kSH_C3[2] * y * (4.0f * zz - xx - yy)       * s10
                + kSH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * s11
                + kSH_C3[4] * x * (4.0f * zz - xx - yy)       * s12
                + kSH_C3[5] * z * (xx - yy)                    * s13
                + kSH_C3[6] * x * (xx - 3.0f * yy)            * s14;
      }
    }
  }

  // Add 0.5 bias (training convention) and clamp
  result += 0.5f;
  result = glm::max(result, vec3(0.0f));
  return result;
}

// ────────────────────────── Build 3D covariance ──────────────────────────
/// Σ = R S S^T R^T  where R = quat_to_mat3(normalize(q)), S = diag(exp(scale))
/// Internal tinygs rotation storage uses vec4(w, x, y, z).
static quat normalize_rotation_wxyz(const vec4& raw_rot_wxyz) {
  return glm::normalize(quat(raw_rot_wxyz.x, raw_rot_wxyz.y, raw_rot_wxyz.z, raw_rot_wxyz.w));
}

static mat3x3 build_cov3d(const vec4& raw_quat, const vec3& raw_scale) {
  quat q = normalize_rotation_wxyz(raw_quat);
  mat3x3 R = glm::mat3_cast(q);
  vec3 s = activate_scale(raw_scale); // exp(raw_scale)
  // RS
  mat3x3 RS;
  RS[0] = R[0] * s.x;
  RS[1] = R[1] * s.y;
  RS[2] = R[2] * s.z;
  // Σ = RS * (RS)^T
  mat3x3 cov = RS * glm::transpose(RS);
  return cov;
}

// ────────────────────────── Project to 2D (EWA splatting) ──────────────────────────
/// Returns the 2×2 projected covariance matrix Σ' = J W Σ W^T J^T
/// where W = upper-left 3×3 of w2c, J = Jacobian of perspective projection.
/// Also returns the 2D mean in pixel coordinates.
struct Projected2D {
  vec2 mean2d;    // pixel coords
  float cov2d_a, cov2d_b, cov2d_c; // symmetric 2×2: [[a,b],[b,c]]
  float depth;    // z in camera space
};

static Projected2D project_gaussian(
    const vec3& mean3d,
    const mat3x3& cov3d,
    const mat4x4& w2c,
    float fx, float fy, float cx, float cy) {

  // Transform mean to camera space
  vec4 mean_h = vec4(mean3d, 1.0f);
  vec4 mean_cam4 = w2c * mean_h;
  float x_cam = mean_cam4.x;
  float y_cam = mean_cam4.y;
  float z_cam = mean_cam4.z;

  // 2D pixel coordinates via pinhole projection
  float u = fx * x_cam / z_cam + cx;
  float v = fy * y_cam / z_cam + cy;

  // W = upper-left 3×3 of w2c (the rotation part)
  mat3x3 W;
  W[0] = vec3(w2c[0]);
  W[1] = vec3(w2c[1]);
  W[2] = vec3(w2c[2]);

  // Jacobian of perspective projection
  // J = [[fx/z, 0, -fx*x/z^2], [0, fy/z, -fy*y/z^2]]
  // We store J as 2 rows × 3 cols.
  float inv_z = 1.0f / z_cam;
  float inv_z2 = inv_z * inv_z;

  // Compute T = J * W (2×3 matrix)
  // But first compute W * Σ * W^T (3×3 in camera space), then J * ... * J^T
  mat3x3 cov_cam = W * cov3d * glm::transpose(W);

  // Now compute J * cov_cam * J^T
  // J row 0: [fx/z, 0, -fx*x/z^2]
  // J row 1: [0, fy/z, -fy*y/z^2]
  // Build the 2×2 result directly:
  float j00 = fx * inv_z;
  float j02 = -fx * x_cam * inv_z2;
  float j11 = fy * inv_z;
  float j12 = -fy * y_cam * inv_z2;

  // cov2d = J * cov_cam * J^T
  // Element [0][0] = j00*cov_cam[0][0]*j00 + j00*cov_cam[2][0]*j02 + j02*cov_cam[0][2]*j00 + j02*cov_cam[2][2]*j02
  // Simplify using matrix multiplication:
  // row0 of J * cov_cam:  r0 = j00 * cov_cam.col(0) + j02 * cov_cam.col(2) (only col indices 0,1,2)
  // Wait - cov_cam is 3×3 and in GLM, cov_cam[col][row].
  // Let me just do it carefully.
  // cov_cam[i][j] = element at column i, row j (GLM convention)
  // So cov_cam element (row, col) = cov_cam[col][row].
  
  // J * cov_cam (2×3 result):
  // result(i, j) = sum_k J(i, k) * cov_cam(k, j)
  // = sum_k J(i, k) * cov_cam[j][k]
  
  // J(0, k) = {j00, 0, j02}
  // J(1, k) = {0, j11, j12}
  
  // (J * cov_cam)(0, j) = j00 * cov_cam[j][0] + j02 * cov_cam[j][2]
  // (J * cov_cam)(1, j) = j11 * cov_cam[j][1] + j12 * cov_cam[j][2]
  
  // Then (J * cov_cam * J^T)(i, j) = sum_k (J * cov_cam)(i, k) * J(j, k)
  
  float jc00 = j00 * cov_cam[0][0] + j02 * cov_cam[0][2];
  float jc01 = j00 * cov_cam[1][0] + j02 * cov_cam[1][2];
  float jc02 = j00 * cov_cam[2][0] + j02 * cov_cam[2][2];
  
  float jc10 = j11 * cov_cam[0][1] + j12 * cov_cam[0][2];
  float jc11 = j11 * cov_cam[1][1] + j12 * cov_cam[1][2];
  float jc12 = j11 * cov_cam[2][1] + j12 * cov_cam[2][2];
  
  float a = jc00 * j00 + jc01 * 0.0f + jc02 * j02;
  float b = jc00 * 0.0f + jc01 * j11 + jc02 * j12;
  float c = jc10 * 0.0f + jc11 * j11 + jc12 * j12;

  // Add small epsilon to diagonal for numerical stability (low-pass filter)
  a += 0.3f;
  c += 0.3f;

  Projected2D out;
  out.mean2d = vec2(u, v);
  out.cov2d_a = a;
  out.cov2d_b = b;
  out.cov2d_c = c;
  out.depth = z_cam;
  return out;
}

// ────────────────────────── Compute bounding rect at 3σ ──────────────────────────
/// Returns axis-aligned bounding rectangle in pixel coordinates.
static void compute_aabb_3sigma(
    float cov_a, float cov_b, float cov_c,
    const vec2& mean2d,
    int width, int height,
    int& x_min, int& x_max, int& y_min, int& y_max) {

  // Eigenvalues of 2×2 symmetric matrix [[a,b],[b,c]]
  float trace = cov_a + cov_c;
  float det = cov_a * cov_c - cov_b * cov_b;
  float disc = std::sqrt(std::max(0.0f, trace * trace * 0.25f - det));
  float lambda1 = trace * 0.5f + disc;
  float lambda2 = trace * 0.5f - disc;

  // 3-sigma radius
  float radius = 3.0f * std::sqrt(std::max(lambda1, lambda2));

  x_min = std::max(0, (int)std::floor(mean2d.x - radius));
  x_max = std::min(width - 1, (int)std::ceil(mean2d.x + radius));
  y_min = std::max(0, (int)std::floor(mean2d.y - radius));
  y_max = std::min(height - 1, (int)std::ceil(mean2d.y + radius));
}

// ──────────────────────────── Forward pass ────────────────────────────

void CPUReferenceRasterizer::forward(const RasterizeContext& ctx) {
  CHECK_THROW(m_gaussians != nullptr);
  const int N = static_cast<int>(m_gaussians->size());
  const int width = ctx.fwd_input.width;
  const int height = ctx.fwd_input.height;
  const float near = ctx.fwd_input.near;
  const float far = ctx.fwd_input.far;

  // Extract camera intrinsics (GLM column-major: K[col][row])
  const mat3x3& K = ctx.fwd_input.K;
  const float fx = K[0][0];
  const float fy = K[1][1];
  const float cx = K[2][0];
  const float cy = K[2][1];
  const mat4x4& w2c = ctx.fwd_input.w2c;

  // Camera position in world space
  mat4x4 c2w = glm::inverse(w2c);
  vec3 cam_pos = vec3(c2w[3]);

  // Copy gaussian data from GPU to host
  Gaussian3d gs;
  m_gaussians->copy_to_host(gs);
  int sh_degree = m_gaussians->get_sh_degree();

  // ──── Per-Gaussian preprocessing ────
  struct GaussInfo {
    int idx;
    Projected2D proj;
    vec3 color;
    float opacity;
    float inv_det;   // 1 / det(Σ2D)
  };

  std::vector<GaussInfo> visible;
  visible.reserve(N);
  
  // Densification info
  std::vector<DensificationInfo> dinfo(N);

  for (int i = 0; i < N; ++i) {
    vec3 mean = gs.means[i];
    vec4 raw_rot = gs.rotations[i];
    vec3 raw_scale = gs.scales[i];
    float raw_opacity = gs.opacities[i];

    // Build 3D covariance
    mat3x3 cov3d = build_cov3d(raw_rot, raw_scale);

    // Project
    Projected2D proj = project_gaussian(mean, cov3d, w2c, fx, fy, cx, cy);

    // Depth cull
    if (proj.depth < near || proj.depth > far) continue;

    // Compute determinant of 2D covariance
    float det = proj.cov2d_a * proj.cov2d_c - proj.cov2d_b * proj.cov2d_b;
    if (det <= 0.0f) continue;

    // Compute bounding rect
    int x_min, x_max, y_min, y_max;
    compute_aabb_3sigma(proj.cov2d_a, proj.cov2d_b, proj.cov2d_c,
                        proj.mean2d, width, height,
                        x_min, x_max, y_min, y_max);
    if (x_min > x_max || y_min > y_max) continue;

    // Activated opacity
    float opacity = activate_opacity(raw_opacity);

    // Compute color from SH
    vec3 dir = mean - cam_pos;
    float len = glm::length(dir);
    if (len > 0.0f) dir /= len;
    vec3 color = evaluate_sh(i, sh_degree, dir, gs);

    GaussInfo gi;
    gi.idx = i;
    gi.proj = proj;
    gi.color = color;
    gi.opacity = opacity;
    gi.inv_det = 1.0f / det;
    visible.push_back(gi);

    // Densification: max radius in screen space
    float trace = proj.cov2d_a + proj.cov2d_c;
    float disc = std::sqrt(std::max(0.0f, trace * trace * 0.25f - det));
    float lambda_max = trace * 0.5f + disc;
    float radius = 3.0f * std::sqrt(std::max(0.0f, lambda_max));
    dinfo[i].max_radii_screen = std::max(dinfo[i].max_radii_screen, radius);
  }

  // Sort by depth (front-to-back for alpha compositing)
  std::sort(visible.begin(), visible.end(),
            [](const GaussInfo& a, const GaussInfo& b) {
              return a.proj.depth < b.proj.depth;
            });

  // ──── Alpha compositing ────
  // Allocate host image in CHW-tiled layout
  ImageShape shape{static_cast<uint32_t>(width), static_cast<uint32_t>(height), 3};
  uint32_t padded_w = shape.padded_width();
  uint32_t padded_h = shape.padded_height();
  uint32_t tiled_w = shape.tiled_width();
  uint32_t channel_stride = padded_w * padded_h;
  uint32_t total_size = shape.padded_size(); // 3 * channel_stride
  std::vector<float> image(total_size, 0.0f);

  // Per-pixel transmittance (1.0 = fully transparent)
  std::vector<float> transmittance(width * height, 1.0f);

  // For each Gaussian in depth order, splat onto pixels
  for (const auto& gi : visible) {
    float a = gi.proj.cov2d_a;
    float b = gi.proj.cov2d_b;
    float c = gi.proj.cov2d_c;

    // Inverse of 2×2 covariance: [[c, -b], [-b, a]] / det
    float inv_a =  c * gi.inv_det;
    float inv_b = -b * gi.inv_det;
    float inv_c =  a * gi.inv_det;

    float mu_x = gi.proj.mean2d.x;
    float mu_y = gi.proj.mean2d.y;

    // Bounding rect
    int x_min, x_max, y_min, y_max;
    compute_aabb_3sigma(a, b, c,
                        gi.proj.mean2d, width, height,
                        x_min, x_max, y_min, y_max);

    for (int py = y_min; py <= y_max; ++py) {
      for (int px = x_min; px <= x_max; ++px) {
        // Pixel center offset from gaussian center
        // Note: pixel coordinate = (px + 0.5, py + 0.5) for center of pixel
        float dx = static_cast<float>(px) + 0.5f - mu_x;
        float dy = static_cast<float>(py) + 0.5f - mu_y;

        // Gaussian exponent: -0.5 * (dx, dy)^T Σ^{-1} (dx, dy)
        float power = -0.5f * (inv_a * dx * dx + 2.0f * inv_b * dx * dy + inv_c * dy * dy);

        // Skip if negligible
        if (power > 0.0f) power = 0.0f;  // numerical guard
        if (power < -4.5f) continue;      // exp(-4.5) ≈ 0.011, matches 3σ

        float gaussian_val = std::exp(power);
        float alpha = std::min(0.99f, gi.opacity * gaussian_val);
        if (alpha < 1.0f / 255.0f) continue;

        int pix_linear = py * width + px;
        float T = transmittance[pix_linear];
        if (T < 1e-4f) continue; // fully opaque

        float weight = alpha * T;

        // Write to CHW-tiled image
        uint32_t tiled_idx = get_linear_index_tiled(
            static_cast<uint32_t>(py),
            static_cast<uint32_t>(px),
            tiled_w);

        image[0 * channel_stride + tiled_idx] += weight * gi.color.x; // R
        image[1 * channel_stride + tiled_idx] += weight * gi.color.y; // G
        image[2 * channel_stride + tiled_idx] += weight * gi.color.z; // B

        transmittance[pix_linear] = T * (1.0f - alpha);
      }
    }
  }

  // Copy image to GPU
  CHECK_THROW(ctx.fwd_output.image.data != nullptr);
  CUDA_CHECK_THROW(cudaMemcpy(
      ctx.fwd_output.image.data, image.data(),
      total_size * sizeof(float),
      cudaMemcpyHostToDevice));

  // Copy densification info to GPU if present
  if (ctx.densification_info) {
    CUDA_CHECK_THROW(cudaMemcpy(
        ctx.densification_info->data(), dinfo.data(),
        N * sizeof(DensificationInfo),
        cudaMemcpyHostToDevice));
  }

  // Sync if a stream was specified
  if (ctx.stream) {
    CUDA_CHECK_THROW(cudaStreamSynchronize(ctx.stream));
  }
}

// ──────────────────────────── Backward pass ────────────────────────────
/// Computes gradients w.r.t. all Gaussian parameters given dL/d(image).
/// Uses the same algorithm as forward but with reverse-mode AD applied
/// to the alpha-compositing and all projection steps.

void CPUReferenceRasterizer::backward(RasterizeContext& ctx) {
  CHECK_THROW(m_gaussians != nullptr);
  CHECK_THROW(ctx.gaussians_grad != nullptr);
  const int N = static_cast<int>(m_gaussians->size());
  const int width = ctx.fwd_input.width;
  const int height = ctx.fwd_input.height;
  const float near = ctx.fwd_input.near;
  const float far = ctx.fwd_input.far;

  const mat3x3& K = ctx.fwd_input.K;
  const float fx = K[0][0];
  const float fy = K[1][1];
  const float cx = K[2][0];
  const float cy = K[2][1];
  const mat4x4& w2c = ctx.fwd_input.w2c;
  mat4x4 c2w = glm::inverse(w2c);
  vec3 cam_pos = vec3(c2w[3]);

  float grad_scaler = ctx.grad_scaler;

  // Copy gaussian data from GPU
  Gaussian3d gs;
  m_gaussians->copy_to_host(gs);
  int sh_degree = m_gaussians->get_sh_degree();

  // Copy grad_output image from GPU (dL/d_image) — CHW-tiled
  ImageShape shape{static_cast<uint32_t>(width), static_cast<uint32_t>(height), 3};
  uint32_t padded_w = shape.padded_width();
  uint32_t padded_h = shape.padded_height();
  uint32_t tiled_w = shape.tiled_width();
  uint32_t channel_stride = padded_w * padded_h;
  uint32_t total_size = shape.padded_size();

  std::vector<float> grad_image(total_size, 0.0f);
  CHECK_THROW(ctx.grad_output.image.data != nullptr);
  CUDA_CHECK_THROW(cudaMemcpy(
      grad_image.data(), ctx.grad_output.image.data,
      total_size * sizeof(float),
      cudaMemcpyDeviceToHost));

  // ──── Preprocessing: identical to forward ────
  struct GaussInfo {
    int idx;
    Projected2D proj;
    vec3 color;
    float opacity;            // activated
    float raw_opacity;
    float inv_det;
    mat3x3 cov3d;
    // For SH backward
    vec3 dir;
  };

  std::vector<GaussInfo> visible;
  visible.reserve(N);

  for (int i = 0; i < N; ++i) {
    vec3 mean = gs.means[i];
    vec4 raw_rot = gs.rotations[i];
    vec3 raw_scale = gs.scales[i];
    float raw_opacity = gs.opacities[i];

    mat3x3 cov3d = build_cov3d(raw_rot, raw_scale);
    Projected2D proj = project_gaussian(mean, cov3d, w2c, fx, fy, cx, cy);
    if (proj.depth < near || proj.depth > far) continue;

    float det = proj.cov2d_a * proj.cov2d_c - proj.cov2d_b * proj.cov2d_b;
    if (det <= 0.0f) continue;

    int x_min, x_max, y_min, y_max;
    compute_aabb_3sigma(proj.cov2d_a, proj.cov2d_b, proj.cov2d_c,
                        proj.mean2d, width, height,
                        x_min, x_max, y_min, y_max);
    if (x_min > x_max || y_min > y_max) continue;

    float opacity = activate_opacity(raw_opacity);
    vec3 dir = mean - cam_pos;
    float len = glm::length(dir);
    if (len > 0.0f) dir /= len;
    vec3 color = evaluate_sh(i, sh_degree, dir, gs);

    GaussInfo gi;
    gi.idx = i;
    gi.proj = proj;
    gi.color = color;
    gi.opacity = opacity;
    gi.raw_opacity = raw_opacity;
    gi.inv_det = 1.0f / det;
    gi.cov3d = cov3d;
    gi.dir = dir;
    visible.push_back(gi);
  }

  // Sort by depth (same order as forward)
  std::sort(visible.begin(), visible.end(),
            [](const GaussInfo& a, const GaussInfo& b) {
              return a.proj.depth < b.proj.depth;
            });

  // ──── Forward re-computation for per-pixel state ────
  // We need per-pixel accumulated color C_prefix and transmittance T_prefix
  // for each Gaussian in order.
  // Strategy: two-pass approach.
  //   Pass 1: Forward — record per-pixel transmittance before each Gaussian contributes.
  //   Pass 2: Backward — accumulate gradients.

  // Per-pixel accumulators (simple arrays indexed by [py * width + px])
  std::vector<float> T_pixel(width * height, 1.0f); // transmittance before current gaussian
  
  // For backward, we need to accumulate dL/d(color_i), dL/d(alpha_i), dL/d(mean2d_i)
  // for each Gaussian across all pixels it touches.
  
  // Gradient accumulators for each Gaussian (indexed by visible list position)
  int n_vis = static_cast<int>(visible.size());
  
  // Output gradient structure
  Gaussian3d grad_gs;
  grad_gs.means.resize(N, vec3(0.0f));
  grad_gs.opacities.resize(N, 0.0f);
  grad_gs.rotations.resize(N, vec4(0.0f));
  grad_gs.scales.resize(N, vec3(0.0f));
  grad_gs.sh0.resize(N, vec3(0.0f));
  grad_gs.sh1.resize(N * kSHDegreeNumCoeffs[1], vec3(0.0f));
  grad_gs.sh2.resize(N * kSHDegreeNumCoeffs[2], vec3(0.0f));
  grad_gs.sh3.resize(N * kSHDegreeNumCoeffs[3], vec3(0.0f));

  // ──── Combined forward-backward pass ────
  // For each Gaussian in depth order, compute the contribution and gradient.
  
  // We also need suffix sums for the backward pass:
  // dL/dC_pixel = sum_{j>=i} (dL/d_out_c * alpha_j * T_before_j * color_j)
  // But it's easier to use the recursion:
  //   dL/d(alpha_i) = T_i * [sum_c color_ic * dL/dC_pixel_c - (suffix_color · dL/dC_pixel)]
  // where suffix_color is the contribution from Gaussians after i.
  //
  // Actually, the standard backward for alpha compositing is:
  //   out_c = sum_i weight_i * color_ic    where weight_i = alpha_i * T_i
  //   T_i = prod_{j<i} (1 - alpha_j)
  //
  //   dL/d(color_ic) = weight_i * dL/d(out_c)
  //   dL/d(alpha_i) = T_i * sum_c [color_ic * dL/d(out_c)] - (1/(1-alpha_i)) * sum_c [S_ic * dL/d(out_c)]
  //   where S_ic = sum_{j>i} weight_j * color_jc  (suffix contribution after i)
  //
  // For simplicity and correctness, we compute this per-pixel using pre-computed arrays.
  
  // Pre-compute per-pixel accumulated color for the suffix sum.
  // accumulated_suffix[pix] = sum_{j>current} weight_j * color_j
  // We'll compute this as: total_color[pix] - prefix_color[pix] (including current)
  
  // First, compute the total forward pass to get per-pixel total color
  // (matching the forward pass output).
  struct PerPixelPerGaussian {
    float alpha;      // opacity * G(pixel)
    float T_before;   // transmittance before this Gaussian
  };
  
  // We'll process each pixel independently for correctness.
  // For each pixel, build the ordered list of Gaussians that affect it.
  
  // Per-pixel: list of (visible_idx, alpha)
  // This is O(P*N) in the worst case, but for reference correctness that's fine.
  
  // Reset transmittance
  std::fill(T_pixel.begin(), T_pixel.end(), 1.0f);

  // Densification info: copy existing from GPU (preserves max_radii_screen from forward)
  std::vector<DensificationInfo> dinfo(N);
  if (ctx.densification_info) {
    CUDA_CHECK_THROW(cudaMemcpy(
        dinfo.data(), ctx.densification_info->data(),
        N * sizeof(DensificationInfo),
        cudaMemcpyDeviceToHost));
  }
  
  // For each visible Gaussian, iterate over its covered pixels
  // First pass: record per-pixel per-gaussian alpha and T_before
  struct PixelContrib {
    float alpha;
    float T_before;
  };

  // Indexed as [vis_idx] -> map of pixel_linear -> PixelContrib
  // (We store per visible gaussian which pixels it affects)
  std::vector<std::vector<std::pair<int, PixelContrib>>> vis_pixel_contribs(n_vis);

  for (int vi = 0; vi < n_vis; ++vi) {
    const auto& gi = visible[vi];
    float a = gi.proj.cov2d_a;
    float b = gi.proj.cov2d_b;
    float c = gi.proj.cov2d_c;
    float inv_a =  c * gi.inv_det;
    float inv_b = -b * gi.inv_det;
    float inv_c =  a * gi.inv_det;
    float mu_x = gi.proj.mean2d.x;
    float mu_y = gi.proj.mean2d.y;

    int x_min, x_max, y_min, y_max;
    compute_aabb_3sigma(a, b, c, gi.proj.mean2d, width, height,
                        x_min, x_max, y_min, y_max);

    for (int py = y_min; py <= y_max; ++py) {
      for (int px = x_min; px <= x_max; ++px) {
        float dx = static_cast<float>(px) + 0.5f - mu_x;
        float dy = static_cast<float>(py) + 0.5f - mu_y;
        float power = -0.5f * (inv_a * dx * dx + 2.0f * inv_b * dx * dy + inv_c * dy * dy);
        if (power > 0.0f) power = 0.0f;
        if (power < -4.5f) continue;

        float gaussian_val = std::exp(power);
        float alpha = std::min(0.99f, gi.opacity * gaussian_val);
        if (alpha < 1.0f / 255.0f) continue;

        int pix = py * width + px;
        float T = T_pixel[pix];
        if (T < 1e-4f) continue;

        PixelContrib pc;
        pc.alpha = alpha;
        pc.T_before = T;
        vis_pixel_contribs[vi].push_back({pix, pc});

        T_pixel[pix] = T * (1.0f - alpha);
      }
    }
  }

  // ──── Backward: accumulate gradients ────
  // Per-pixel suffix accumulator: S(pix) = sum_{j > current vis_idx} weight_j * color_j
  // We iterate visible list in reverse; suffix_rgb[pix] is correct before we update it.
  std::vector<vec3> suffix_rgb(width * height, vec3(0.0f));

  // Iterate in reverse depth order
  for (int vi = n_vis - 1; vi >= 0; --vi) {
    const auto& gi = visible[vi];
    int gauss_idx = gi.idx;

    // Per-Gaussian gradient accumulators
    vec3 dL_dcolor(0.0f);
    vec2 dL_dmean2d(0.0f);
    vec2 dL_dmean2d_abs(0.0f);  // Component-wise absolute gradient for densification
    float dL_dopacity = 0.0f;
    float dL_dcov2d_a = 0.0f, dL_dcov2d_b = 0.0f, dL_dcov2d_c = 0.0f;

    float a = gi.proj.cov2d_a;
    float b_cov = gi.proj.cov2d_b;
    float c = gi.proj.cov2d_c;
    float inv_a =  c * gi.inv_det;
    float inv_b = -b_cov * gi.inv_det;
    float inv_c =  a * gi.inv_det;
    float mu_x = gi.proj.mean2d.x;
    float mu_y = gi.proj.mean2d.y;

    float det = a * c - b_cov * b_cov;
    float inv_det2 = 1.0f / (det * det);

    // Single per-pixel loop: compute all gradients before updating suffix_rgb
    for (const auto& [pix, pc] : vis_pixel_contribs[vi]) {
      int px = pix % width;
      int py = pix / width;

      uint32_t tiled_idx = get_linear_index_tiled(
          static_cast<uint32_t>(py),
          static_cast<uint32_t>(px),
          tiled_w);

      // dL/d(out) for this pixel
      vec3 dL_dout;
      dL_dout.x = grad_image[0 * channel_stride + tiled_idx] * grad_scaler;
      dL_dout.y = grad_image[1 * channel_stride + tiled_idx] * grad_scaler;
      dL_dout.z = grad_image[2 * channel_stride + tiled_idx] * grad_scaler;

      float alpha_i = pc.alpha;
      float T_i = pc.T_before;
      float weight_i = alpha_i * T_i;

      // dL/d(color_i) += weight_i * dL/d(out)
      dL_dcolor += weight_i * dL_dout;

      // dL/d(alpha_i):
      //   d(out)/d(alpha_i) = T_i * color_i - suffix / (1 - alpha_i)
      //   suffix_rgb[pix] correctly holds sum_{j>vi} weight_j * color_j
      vec3 suffix = suffix_rgb[pix];
      float dalpha = T_i * glm::dot(gi.color, dL_dout);
      if (std::abs(1.0f - alpha_i) > 1e-6f) {
        dalpha -= (1.0f / (1.0f - alpha_i)) * glm::dot(suffix, dL_dout);
      }

      // Gaussian evaluation recomputation
      float dx = static_cast<float>(px) + 0.5f - mu_x;
      float dy = static_cast<float>(py) + 0.5f - mu_y;
      float power = -0.5f * (inv_a * dx * dx + 2.0f * inv_b * dx * dy + inv_c * dy * dy);
      if (power > 0.0f) power = 0.0f;
      float gaussian_val = std::exp(power);

      // Chain through alpha = min(0.99, opacity * gauss_val)
      float dL_dgauss = 0.0f;
      if (gi.opacity * gaussian_val < 0.99f) {
        dL_dgauss = dalpha * gi.opacity;
        dL_dopacity += dalpha * gaussian_val;
      }
      float dL_dpower = dL_dgauss * gaussian_val;

      // d(power)/d(mean2d)
      // power = -0.5 * (inv_a*dx^2 + 2*inv_b*dx*dy + inv_c*dy^2)
      // d(power)/d(mu_x) = (inv_a*dx + inv_b*dy)  [since dx = px-mu_x, d(dx)/d(mu_x) = -1]
      // d(power)/d(mu_y) = (inv_b*dx + inv_c*dy)
      float dmean2d_x = dL_dpower * (inv_a * dx + inv_b * dy);
      float dmean2d_y = dL_dpower * (inv_b * dx + inv_c * dy);
      dL_dmean2d.x += dmean2d_x;
      dL_dmean2d.y += dmean2d_y;
      dL_dmean2d_abs.x += std::abs(dmean2d_x);
      dL_dmean2d_abs.y += std::abs(dmean2d_y);

      // d(power)/d(Σ^{-1}) then chain to d(Σ)
      float dL_dinv_a = dL_dpower * (-0.5f * dx * dx);
      float dL_dinv_b = dL_dpower * (-1.0f * dx * dy);
      float dL_dinv_c = dL_dpower * (-0.5f * dy * dy);

      // Explicit partial derivatives of Σ^{-1} entries w.r.t. Σ entries:
      // inv_a = c/det,  inv_b = -b/det,  inv_c = a/det,  det = a*c - b²
      //
      // d(inv_a)/da = -c²/det²
      // d(inv_a)/db = 2bc/det²
      // d(inv_a)/dc = -b²/det²
      //
      // d(inv_b)/da = bc/det²
      // d(inv_b)/db = -(ac+b²)/det²
      // d(inv_b)/dc = ab/det²
      //
      // d(inv_c)/da = -b²/det²
      // d(inv_c)/db = 2ab/det²
      // d(inv_c)/dc = -a²/det²
      dL_dcov2d_a += dL_dinv_a * (-c * c * inv_det2)
                   + dL_dinv_b * (b_cov * c * inv_det2)
                   + dL_dinv_c * (-b_cov * b_cov * inv_det2);

      dL_dcov2d_b += dL_dinv_a * (2.0f * b_cov * c * inv_det2)
                   + dL_dinv_b * (-(a * c + b_cov * b_cov) * inv_det2)
                   + dL_dinv_c * (2.0f * a * b_cov * inv_det2);

      dL_dcov2d_c += dL_dinv_a * (-b_cov * b_cov * inv_det2)
                   + dL_dinv_b * (a * b_cov * inv_det2)
                   + dL_dinv_c * (-a * a * inv_det2);
    }

    // NOW update suffix_rgb for the next (earlier) Gaussian's backward pass
    for (const auto& [pix, pc] : vis_pixel_contribs[vi]) {
      suffix_rgb[pix] += pc.alpha * pc.T_before * gi.color;
    }

    // ──── Backward through projection chain ────

    // 1. Apply opacity activation chain: raw -> sigmoid(raw) = opacity
    //    dL_dopacity was accumulated in the clean per-pixel loop above.
    float dL_draw_opacity = dL_dopacity * activate_opacity_deriv(gi.raw_opacity);
    grad_gs.opacities[gauss_idx] += dL_draw_opacity;

    // 2. Gradients w.r.t. SH0 (and higher degrees w.r.t. color only affect SH)
    //    color = evaluate_sh(...) which is SH_C0 * sh0[i] + ... + 0.5, clamped to >= 0.
    //    dL/d(sh0) = dL/d(color) * d(color)/d(sh0) = dL/dcolor * SH_C0  (if color > 0)
    //    For higher degrees, similar.
    
    // Clamp mask: gradient flows only through unclamped channels.
    // evaluate_sh returns max(SH_result + 0.5, 0). Check if > 0.
    vec3 clamp_mask;
    clamp_mask.x = (gi.color.x > 0.0f) ? 1.0f : 0.0f;
    clamp_mask.y = (gi.color.y > 0.0f) ? 1.0f : 0.0f;
    clamp_mask.z = (gi.color.z > 0.0f) ? 1.0f : 0.0f;
    
    vec3 dL_dcolor_clamped = dL_dcolor * clamp_mask;

    // SH0 gradient
    grad_gs.sh0[gauss_idx] += kSH_C0 * dL_dcolor_clamped;

    if (sh_degree >= 1) {
      float x = gi.dir.x, y = gi.dir.y, z = gi.dir.z;
      int nc1 = kSHDegreeNumCoeffs[1];
      grad_gs.sh1[gauss_idx * nc1 + 0] += -kSH_C1 * y * dL_dcolor_clamped;
      grad_gs.sh1[gauss_idx * nc1 + 1] +=  kSH_C1 * z * dL_dcolor_clamped;
      grad_gs.sh1[gauss_idx * nc1 + 2] += -kSH_C1 * x * dL_dcolor_clamped;

      if (sh_degree >= 2) {
        float xx = x * x, yy = y * y, zz = z * z;
        float xy = x * y, yz = y * z, xz = x * z;
        int nc2 = kSHDegreeNumCoeffs[2];
        grad_gs.sh2[gauss_idx * nc2 + 0] += kSH_C2[0] * xy * dL_dcolor_clamped;
        grad_gs.sh2[gauss_idx * nc2 + 1] += kSH_C2[1] * yz * dL_dcolor_clamped;
        grad_gs.sh2[gauss_idx * nc2 + 2] += kSH_C2[2] * (2.0f * zz - xx - yy) * dL_dcolor_clamped;
        grad_gs.sh2[gauss_idx * nc2 + 3] += kSH_C2[3] * xz * dL_dcolor_clamped;
        grad_gs.sh2[gauss_idx * nc2 + 4] += kSH_C2[4] * (xx - yy) * dL_dcolor_clamped;

        if (sh_degree >= 3) {
          int nc3 = kSHDegreeNumCoeffs[3];
          grad_gs.sh3[gauss_idx * nc3 + 0] += kSH_C3[0] * y * (3.0f * xx - yy) * dL_dcolor_clamped;
          grad_gs.sh3[gauss_idx * nc3 + 1] += kSH_C3[1] * xy * z * dL_dcolor_clamped;
          grad_gs.sh3[gauss_idx * nc3 + 2] += kSH_C3[2] * y * (4.0f * zz - xx - yy) * dL_dcolor_clamped;
          grad_gs.sh3[gauss_idx * nc3 + 3] += kSH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * dL_dcolor_clamped;
          grad_gs.sh3[gauss_idx * nc3 + 4] += kSH_C3[4] * x * (4.0f * zz - xx - yy) * dL_dcolor_clamped;
          grad_gs.sh3[gauss_idx * nc3 + 5] += kSH_C3[5] * z * (xx - yy) * dL_dcolor_clamped;
          grad_gs.sh3[gauss_idx * nc3 + 6] += kSH_C3[6] * x * (xx - 3.0f * yy) * dL_dcolor_clamped;
        }
      }
    }

    // 3. Gradients w.r.t. mean2d → mean3d
    //    mean2d = (fx * x_cam / z_cam + cx, fy * y_cam / z_cam + cy)
    //    d(mean2d)/d(mean_cam) = J (the Jacobian we already computed)
    //    d(mean_cam)/d(mean3d) = w2c[:3,:3] = W
    
    // We accumulated dL_dmean2d from all pixels.
    // Now chain through projection:
    // dL/d(x_cam) = dL/du * fx/z + dL/dv * 0 = dL_dmean2d.x * fx / z_cam
    // dL/d(y_cam) = dL/du * 0 + dL/dv * fy/z = dL_dmean2d.y * fy / z_cam
    // dL/d(z_cam) = dL/du * (-fx*x_cam/z^2) + dL/dv * (-fy*y_cam/z^2)
    float z_cam = gi.proj.depth;
    float inv_z = 1.0f / z_cam;
    float inv_z2 = inv_z * inv_z;
    
    vec4 mean_cam4 = w2c * vec4(gs.means[gauss_idx], 1.0f);
    float x_cam = mean_cam4.x;
    float y_cam = mean_cam4.y;
    
    vec3 dL_dmean_cam;
    dL_dmean_cam.x = dL_dmean2d.x * fx * inv_z;
    dL_dmean_cam.y = dL_dmean2d.y * fy * inv_z;
    dL_dmean_cam.z = dL_dmean2d.x * (-fx * x_cam * inv_z2) + dL_dmean2d.y * (-fy * y_cam * inv_z2);
    
    // dL/d(mean3d) = W^T * dL/d(mean_cam)
    mat3x3 W;
    W[0] = vec3(w2c[0]);
    W[1] = vec3(w2c[1]);
    W[2] = vec3(w2c[2]);
    grad_gs.means[gauss_idx] += glm::transpose(W) * dL_dmean_cam;

    // 4. Gradients w.r.t. cov2d → cov3d → rotation, scale
    // cov2d = J * W * cov3d * W^T * J^T
    // dL/d(cov2d) was correctly accumulated in the clean per-pixel loop above.

    // Now chain cov2d gradients through EWA projection to cov3d, then to rotation and scale.
    // Let M = J * W (2×3 matrix)
    // cov2d = M * cov3d * M^T
    // dL/d(cov3d) = M^T * dL/d(cov2d) * M
    
    // Build M = J * W
    float j00 = fx * inv_z;
    float j02 = -fx * x_cam * inv_z2;
    float j11 = fy * inv_z;
    float j12 = -fy * y_cam * inv_z2;
    
    // M = J (2×3) * W (3×3)
    // M(i,j) = sum_k J(i,k) * W(k,j)
    // But W is the upper-left 3×3 of w2c, stored column-major.
    // W[col][row], so W(row, col) = W[col][row]
    // J(0,k) = {j00, 0, j02}, J(1,k) = {0, j11, j12}
    float M[2][3];
    for (int col = 0; col < 3; ++col) {
      M[0][col] = j00 * W[col][0] + j02 * W[col][2];
      M[1][col] = j11 * W[col][1] + j12 * W[col][2];
    }

    // dL/d(cov2d) as symmetric 2×2: [[dL_da, dL_db], [dL_db, dL_dc]]
    // dL/d(cov3d) = M^T * dL_dcov2d * M  (3×3 result)
    // First compute dL_dcov2d * M (2×3)
    float dM[2][3];
    for (int j = 0; j < 3; ++j) {
      dM[0][j] = dL_dcov2d_a * M[0][j] + dL_dcov2d_b * M[1][j];
      dM[1][j] = dL_dcov2d_b * M[0][j] + dL_dcov2d_c * M[1][j];
    }

    // dL_dcov3d(i,j) = M^T(i,:) * dM(:,j) = M(0,i)*dM(0,j) + M(1,i)*dM(1,j)
    mat3x3 dL_dcov3d;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        dL_dcov3d[j][i] = M[0][i] * dM[0][j] + M[1][i] * dM[1][j];
      }
    }

    // Chain rule: cov3d = R * S * S^T * R^T
    // dL/d(R), dL/d(S)
    quat q = normalize_rotation_wxyz(gs.rotations[gauss_idx]);
    mat3x3 R = glm::mat3_cast(q);
    vec3 s = activate_scale(gs.scales[gauss_idx]);
    
    // RS matrix
    mat3x3 RS;
    RS[0] = R[0] * s.x;
    RS[1] = R[1] * s.y;
    RS[2] = R[2] * s.z;

    // cov3d = RS * RS^T
    // dL/d(RS) = (dL/d(cov3d) + dL/d(cov3d)^T) * RS
    mat3x3 dL_dcov3d_sym = dL_dcov3d + glm::transpose(dL_dcov3d);
    mat3x3 dL_dRS = dL_dcov3d_sym * RS;

    // dL/d(s_k) = sum_i dL_dRS[k][i] * R[k][i]  for each column k
    // dL/d(R[j][i]) = dL_dRS[j][i] * s_j
    vec3 dL_ds;
    dL_ds.x = glm::dot(vec3(dL_dRS[0]), vec3(R[0]));
    dL_ds.y = glm::dot(vec3(dL_dRS[1]), vec3(R[1]));
    dL_ds.z = glm::dot(vec3(dL_dRS[2]), vec3(R[2]));

    // Chain through scale activation: scale = exp(raw_scale)
    // dL/d(raw_scale) = dL/d(scale) * exp(raw_scale) = dL_ds * s
    grad_gs.scales[gauss_idx] += dL_ds * s;

    // dL/d(R)
    mat3x3 dL_dR;
    dL_dR[0] = vec3(dL_dRS[0]) * s.x;
    dL_dR[1] = vec3(dL_dRS[1]) * s.y;
    dL_dR[2] = vec3(dL_dRS[2]) * s.z;

    // Chain R → quaternion
    // R = mat3_cast(normalize(q_raw))
    // We compute dL/d(q_normalized) first, then chain through normalization.
    
    // For dL/d(quat) from dL/d(R), we use the relationship:
    // R is computed from normalized quaternion (w,x,y,z).
    // The Jacobian dR/dq is complex but well-known.
    float w = q.w, qx = q.x, qy = q.y, qz = q.z;

    // dL/dw from dL/dR:
    // R = [[1-2(y²+z²), 2(xy-wz), 2(xz+wy)],
    //      [2(xy+wz), 1-2(x²+z²), 2(yz-wx)],
    //      [2(xz-wy), 2(yz+wx), 1-2(x²+y²)]]
    // dR/dw = [[0, -2z, 2y],
    //          [2z, 0, -2x],
    //          [-2y, 2x, 0]]

    // dL/dw = sum_{i,j} dL_dR[j][i] * dR[j][i]/dw
    // GLM column-major: dL_dR[col][row], so dL_dR element (row, col) = dL_dR[col][row]
    float dL_dw = 0.0f;
    dL_dw += dL_dR[1][0] * (-2.0f * qz); // dR(0,1)/dw
    dL_dw += dL_dR[2][0] * ( 2.0f * qy); // dR(0,2)/dw
    dL_dw += dL_dR[0][1] * ( 2.0f * qz); // dR(1,0)/dw
    dL_dw += dL_dR[2][1] * (-2.0f * qx); // dR(1,2)/dw
    dL_dw += dL_dR[0][2] * (-2.0f * qy); // dR(2,0)/dw
    dL_dw += dL_dR[1][2] * ( 2.0f * qx); // dR(2,1)/dw

    float dL_dqx = 0.0f;
    // dR/dx:
    // dR(0,0)/dx = 0,          dR(0,1)/dx = 2y,   dR(0,2)/dx = 2z
    // dR(1,0)/dx = 2y,         dR(1,1)/dx = -4x,  dR(1,2)/dx = -2w
    // dR(2,0)/dx = 2z,         dR(2,1)/dx = 2w,   dR(2,2)/dx = -4x
    dL_dqx += dL_dR[1][0] * ( 2.0f * qy);
    dL_dqx += dL_dR[2][0] * ( 2.0f * qz);
    dL_dqx += dL_dR[0][1] * ( 2.0f * qy);
    dL_dqx += dL_dR[1][1] * (-4.0f * qx);
    dL_dqx += dL_dR[2][1] * (-2.0f * w);
    dL_dqx += dL_dR[0][2] * ( 2.0f * qz);
    dL_dqx += dL_dR[1][2] * ( 2.0f * w);
    dL_dqx += dL_dR[2][2] * (-4.0f * qx);

    float dL_dqy = 0.0f;
    // dR/dy:
    // dR(0,0)/dy = -4y,  dR(0,1)/dy = 2x,   dR(0,2)/dy = 2w
    // dR(1,0)/dy = 2x,   dR(1,1)/dy = 0,    dR(1,2)/dy = 2z
    // dR(2,0)/dy = -2w,  dR(2,1)/dy = 2z,   dR(2,2)/dy = -4y
    dL_dqy += dL_dR[0][0] * (-4.0f * qy);
    dL_dqy += dL_dR[1][0] * ( 2.0f * qx);
    dL_dqy += dL_dR[2][0] * ( 2.0f * w);
    dL_dqy += dL_dR[0][1] * ( 2.0f * qx);
    dL_dqy += dL_dR[2][1] * ( 2.0f * qz);
    dL_dqy += dL_dR[0][2] * (-2.0f * w);
    dL_dqy += dL_dR[1][2] * ( 2.0f * qz);
    dL_dqy += dL_dR[2][2] * (-4.0f * qy);

    float dL_dqz = 0.0f;
    // dR/dz:
    // dR(0,0)/dz = -4z,  dR(0,1)/dz = -2w,  dR(0,2)/dz = 2x
    // dR(1,0)/dz = 2w,   dR(1,1)/dz = -4z,  dR(1,2)/dz = 2y
    // dR(2,0)/dz = 2x,   dR(2,1)/dz = 2y,   dR(2,2)/dz = 0
    dL_dqz += dL_dR[0][0] * (-4.0f * qz);
    dL_dqz += dL_dR[1][0] * (-2.0f * w);
    dL_dqz += dL_dR[2][0] * ( 2.0f * qx);
    dL_dqz += dL_dR[0][1] * ( 2.0f * w);
    dL_dqz += dL_dR[1][1] * (-4.0f * qz);
    dL_dqz += dL_dR[2][1] * ( 2.0f * qy);
    dL_dqz += dL_dR[0][2] * ( 2.0f * qx);
    dL_dqz += dL_dR[1][2] * ( 2.0f * qy);

    // dL/d(normalized_q) = (dL_dw, dL_dqx, dL_dqy, dL_dqz)
    // Chain through quaternion normalization: q_norm = q_raw / ||q_raw||
    vec4 raw_q = gs.rotations[gauss_idx];
    float norm_sq = glm::dot(raw_q, raw_q);
    float norm = std::sqrt(norm_sq);
    float inv_norm = 1.0f / (norm + 1e-8f);
    
    // d(q_norm_i)/d(q_raw_j) = (delta_ij - q_norm_i * q_norm_j) / ||q_raw||
    // Or equivalently: dL/d(q_raw) = (dL/d(q_norm) - q_norm * dot(q_norm, dL/d(q_norm))) / ||q_raw||
    vec4 dL_dqnorm(dL_dw, dL_dqx, dL_dqy, dL_dqz);
    vec4 q_norm_v(q.w, q.x, q.y, q.z);
    vec4 dL_draw_q = (dL_dqnorm - q_norm_v * glm::dot(q_norm_v, dL_dqnorm)) * inv_norm;
    // Internal tinygs rotation storage is vec4(w, x, y, z), matching the GLM
    // constructor argument order used above.
    
    grad_gs.rotations[gauss_idx] += dL_draw_q;

    // 5. Through-Jacobian correction to mean gradient.
    //    cov2d = J * T * J^T where T = W * cov3d * W^T.
    //    J depends on (x_cam, y_cam, z_cam), so changes in mean3d affect
    //    cov2d through J. dL/dJ = 2 * G * J * T  (2×3).
    {
      // T_cam = W * cov3d * W^T = (W*RS) * (W*RS)^T
      mat3x3 WRS;
      WRS[0] = W * RS[0];
      WRS[1] = W * RS[1];
      WRS[2] = W * RS[2];
      mat3x3 T_cam = WRS * glm::transpose(WRS);

      // G * J  (2×3),  G = [[dL_da, dL_db], [dL_db, dL_dc]]
      // J = [[j00, 0, j02], [0, j11, j12]]
      float GJ[2][3];
      GJ[0][0] = dL_dcov2d_a * j00;
      GJ[0][1] = dL_dcov2d_b * j11;
      GJ[0][2] = dL_dcov2d_a * j02 + dL_dcov2d_b * j12;
      GJ[1][0] = dL_dcov2d_b * j00;
      GJ[1][1] = dL_dcov2d_c * j11;
      GJ[1][2] = dL_dcov2d_b * j02 + dL_dcov2d_c * j12;

      // dL_dJ = 2 * GJ * T_cam  (2×3)
      // (GJ * T_cam)_{i,j} = sum_k GJ[i][k] * T_cam(k,j)
      // GLM column-major: T_cam(row, col) = T_cam[col][row]
      float dL_dJ[2][3];
      for (int i = 0; i < 2; ++i) {
        for (int j_idx = 0; j_idx < 3; ++j_idx) {
          dL_dJ[i][j_idx] = 2.0f * (GJ[i][0] * T_cam[j_idx][0]
                                   + GJ[i][1] * T_cam[j_idx][1]
                                   + GJ[i][2] * T_cam[j_idx][2]);
        }
      }

      // Chain dL_dJ through dJ/d(mean_cam):
      //   dJ_{02}/d(x_cam) = -fx/z²
      //   dJ_{12}/d(y_cam) = -fy/z²
      //   dJ_{00}/d(z_cam) = -fx/z²   dJ_{02}/d(z_cam) = 2fx*x/z³
      //   dJ_{11}/d(z_cam) = -fy/z²   dJ_{12}/d(z_cam) = 2fy*y/z³
      float inv_z3 = inv_z2 * inv_z;
      vec3 dL_dmean_cam_J;
      dL_dmean_cam_J.x = dL_dJ[0][2] * (-fx * inv_z2);
      dL_dmean_cam_J.y = dL_dJ[1][2] * (-fy * inv_z2);
      dL_dmean_cam_J.z = dL_dJ[0][0] * (-fx * inv_z2)
                        + dL_dJ[0][2] * (2.0f * fx * x_cam * inv_z3)
                        + dL_dJ[1][1] * (-fy * inv_z2)
                        + dL_dJ[1][2] * (2.0f * fy * y_cam * inv_z3);

      grad_gs.means[gauss_idx] += glm::transpose(W) * dL_dmean_cam_J;
    }

    // 6. Densification: accumulate signed and absolute gradients
    // Scale to NDC space: multiply by (0.5 * w, 0.5 * h) to convert screen-space
    // gradients to NDC-space gradients (matching FastGS behavior).
    // Signed: components cancel across pixels -> smaller norm (directional info)
    // Absolute: components always add -> larger norm (magnitude for under-reconstruction)
    float scale_x = 0.5f * static_cast<float>(width);
    float scale_y = 0.5f * static_cast<float>(height);
    dinfo[gauss_idx].accum_grad_mean2d += std::sqrt(
        (dL_dmean2d.x * scale_x) * (dL_dmean2d.x * scale_x) +
        (dL_dmean2d.y * scale_y) * (dL_dmean2d.y * scale_y));
    dinfo[gauss_idx].accum_absgrad_mean2d += std::sqrt(
        (dL_dmean2d_abs.x * scale_x) * (dL_dmean2d_abs.x * scale_x) +
        (dL_dmean2d_abs.y * scale_y) * (dL_dmean2d_abs.y * scale_y));
    dinfo[gauss_idx].accum_counter += 1.0f;
  }

  // Copy gradients to GPU
  ctx.gaussians_grad->copy_from_host(grad_gs);

  // Copy densification info to GPU if present
  if (ctx.densification_info) {
    CUDA_CHECK_THROW(cudaMemcpy(
        ctx.densification_info->data(), dinfo.data(),
        N * sizeof(DensificationInfo),
        cudaMemcpyHostToDevice));
  }

  if (ctx.stream) {
    CUDA_CHECK_THROW(cudaStreamSynchronize(ctx.stream));
  }
}

// ──────────────────────────── Params ────────────────────────────

json CPUReferenceRasterizer::get_params() const {
  return m_params.to_json();
}

void CPUReferenceRasterizer::set_params(const json& j) {
  m_params.from_json(j);
}

}  // namespace tinygs
