#include <gtest/gtest.h>
#include <tinygs/core/gaussian.hpp>
#include <cmath>

namespace {

using namespace tinygs;

TEST(GaussianActivationTest, ActivateScale) {
  EXPECT_FLOAT_EQ(activate_scale(0.0f), 1.0f);
  EXPECT_FLOAT_EQ(activate_scale(1.0f), std::exp(1.0f));
  EXPECT_FLOAT_EQ(activate_scale(-1.0f), std::exp(-1.0f));
  EXPECT_NEAR(activate_scale(2.0f), 7.389f, 0.001f);
}

TEST(GaussianActivationTest, DeactivateScale) {
  EXPECT_FLOAT_EQ(deactivate_scale(1.0f), 0.0f);
  EXPECT_FLOAT_EQ(deactivate_scale(std::exp(1.0f)), 1.0f);
  EXPECT_FLOAT_EQ(deactivate_scale(std::exp(-1.0f)), -1.0f);
}

TEST(GaussianActivationTest, ActivateDeactivateScaleInverse) {
  float x = 5.0f;
  float activated = activate_scale(x);
  float deactivated = deactivate_scale(activated);
  EXPECT_NEAR(deactivated, x, 1e-5f);
  
  x = -3.0f;
  activated = activate_scale(x);
  deactivated = deactivate_scale(activated);
  EXPECT_NEAR(deactivated, x, 1e-5f);
}

TEST(GaussianActivationTest, ActivateScaleDeriv) {
  EXPECT_FLOAT_EQ(activate_scale_deriv(0.0f), 1.0f);
  EXPECT_FLOAT_EQ(activate_scale_deriv(1.0f), std::exp(1.0f));
  EXPECT_FLOAT_EQ(activate_scale_deriv(2.0f), std::exp(2.0f));
}

TEST(GaussianActivationTest, ActivateScaleVec3) {
  vec3 x(0.0f, 1.0f, -1.0f);
  vec3 activated = activate_scale(x);
  
  EXPECT_FLOAT_EQ(activated.x, 1.0f);
  EXPECT_FLOAT_EQ(activated.y, std::exp(1.0f));
  EXPECT_FLOAT_EQ(activated.z, std::exp(-1.0f));
}

TEST(GaussianActivationTest, DeactivateScaleVec3) {
  vec3 x(1.0f, std::exp(1.0f), std::exp(-1.0f));
  vec3 deactivated = deactivate_scale(x);
  
  EXPECT_FLOAT_EQ(deactivated.x, 0.0f);
  EXPECT_NEAR(deactivated.y, 1.0f, 1e-5f);
  EXPECT_NEAR(deactivated.z, -1.0f, 1e-5f);
}

TEST(GaussianActivationTest, ActivateOpacity) {
  EXPECT_FLOAT_EQ(activate_opacity(0.0f), 0.5f);
  EXPECT_NEAR(activate_opacity(10.0f), 1.0f, 1e-4f);
  EXPECT_NEAR(activate_opacity(-10.0f), 0.0f, 1e-4f);
  EXPECT_NEAR(activate_opacity(2.0f), 0.8808f, 0.001f);
}

TEST(GaussianActivationTest, DeactivateOpacity) {
  EXPECT_FLOAT_EQ(deactivate_opacity(0.5f), 0.0f);
  EXPECT_GT(deactivate_opacity(0.9f), 0.0f);
  EXPECT_LT(deactivate_opacity(0.1f), 0.0f);
}

TEST(GaussianActivationTest, ActivateDeactivateOpacityInverse) {
  float x = 0.3f;
  float deactivated = deactivate_opacity(x);
  float activated = activate_opacity(deactivated);
  EXPECT_NEAR(activated, x, 1e-5f);
  
  x = 0.7f;
  deactivated = deactivate_opacity(x);
  activated = activate_opacity(deactivated);
  EXPECT_NEAR(activated, x, 1e-5f);
}

TEST(GaussianActivationTest, ActivateOpacityDeriv) {
  EXPECT_FLOAT_EQ(activate_opacity_deriv(0.0f), 0.25f);
  
  float deriv_at_1 = activate_opacity_deriv(1.0f);
  float act_1 = activate_opacity(1.0f);
  EXPECT_FLOAT_EQ(deriv_at_1, act_1 * (1.0f - act_1));
}

TEST(GaussianActivationTest, ActivateOpacityDerivSymmetric) {
  float deriv_pos = activate_opacity_deriv(2.0f);
  float deriv_neg = activate_opacity_deriv(-2.0f);
  EXPECT_NEAR(deriv_pos, deriv_neg, 1e-5f);
}

TEST(GaussianActivationTest, OpacityRange) {
  for (float x = -5.0f; x <= 5.0f; x += 0.5f) {
    float opacity = activate_opacity(x);
    EXPECT_GE(opacity, 0.0f);
    EXPECT_LE(opacity, 1.0f);
  }
}

TEST(GaussianActivationTest, ScaleAlwaysPositive) {
  for (float x = -10.0f; x <= 10.0f; x += 1.0f) {
    float scale = activate_scale(x);
    EXPECT_GT(scale, 0.0f);
  }
}

TEST(SHCoeffsTest, kSHDegreeNumCoeffsCorrect) {
  EXPECT_EQ(kSHDegreeNumCoeffs[0], 1);
  EXPECT_EQ(kSHDegreeNumCoeffs[1], 3);
  EXPECT_EQ(kSHDegreeNumCoeffs[2], 5);
  EXPECT_EQ(kSHDegreeNumCoeffs[3], 7);
}

TEST(DensificationInfoTest, DefaultValues) {
  DensificationInfo info;
  EXPECT_FLOAT_EQ(info.accum_counter, 0.0f);
  EXPECT_FLOAT_EQ(info.accum_grad_mean2d, 0.0f);
  EXPECT_FLOAT_EQ(info.accum_absgrad_mean2d, 0.0f);
  EXPECT_FLOAT_EQ(info.max_radii_screen, 0.0f);
  EXPECT_FLOAT_EQ(info.metric_importance_score, 0.0f);
  EXPECT_FLOAT_EQ(info.metric_pruning_score, 0.0f);
}

TEST(Gaussian3dTest, DefaultVectorsEmpty) {
  Gaussian3d gaussians;
  EXPECT_TRUE(gaussians.means.empty());
  EXPECT_TRUE(gaussians.opacities.empty());
  EXPECT_TRUE(gaussians.rotations.empty());
  EXPECT_TRUE(gaussians.scales.empty());
  EXPECT_TRUE(gaussians.sh0.empty());
  EXPECT_TRUE(gaussians.sh1.empty());
  EXPECT_TRUE(gaussians.sh2.empty());
  EXPECT_TRUE(gaussians.sh3.empty());
}

}
