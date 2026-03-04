#include <gtest/gtest.h>
#include <tinygs/initialization/knn.hpp>
#include <tinygs/core/pointcloud.hpp>
#include <tinygs/core/gaussian.hpp>
#include <cmath>
#include <vector>

namespace {

using namespace tinygs;

class KnnAlgorithmTest : public ::testing::Test {
protected:
  void SetUp() override {
    KnnParameters params;
    params.enable_radius_outlier_removal = false;
    knn_init = std::make_unique<KnnInitialization>(params);
  }

  std::unique_ptr<KnnInitialization> knn_init;
};

TEST_F(KnnAlgorithmTest, InitializeRegularGrid) {
  PointCloud pc;
  for (int x = 0; x < 3; ++x) {
    for (int y = 0; y < 3; ++y) {
      for (int z = 0; z < 3; ++z) {
        pc.points.push_back(vec3(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)));
        pc.colors.push_back(vec3(0.5f, 0.5f, 0.5f));
      }
    }
  }

  knn_init->initialize(pc);

  const auto& gaussians = knn_init->gaussians();
  EXPECT_EQ(gaussians.means.size(), 27);

  for (size_t i = 0; i < gaussians.scales.size(); ++i) {
    EXPECT_TRUE(std::isfinite(gaussians.scales[i].x));
    EXPECT_TRUE(std::isfinite(gaussians.scales[i].y));
    EXPECT_TRUE(std::isfinite(gaussians.scales[i].z));
  }
}

TEST_F(KnnAlgorithmTest, ScaleValuesArePositive) {
  PointCloud pc;
  for (int i = 0; i < 10; ++i) {
    pc.points.push_back(vec3(static_cast<float>(i) * 0.1f, 0.0f, 0.0f));
    pc.colors.push_back(vec3(0.5f, 0.5f, 0.5f));
  }

  knn_init->initialize(pc);

  const auto& gaussians = knn_init->gaussians();
  for (const auto& scale : gaussians.scales) {
    float sx = activate_scale(scale.x);
    float sy = activate_scale(scale.y);
    float sz = activate_scale(scale.z);

    EXPECT_GT(sx, 0.0f);
    EXPECT_GT(sy, 0.0f);
    EXPECT_GT(sz, 0.0f);
  }
}

TEST_F(KnnAlgorithmTest, RotationQuaternionsNormalized) {
  PointCloud pc;
  for (int i = 0; i < 10; ++i) {
    pc.points.push_back(vec3(static_cast<float>(i) * 0.1f, 0.0f, 0.0f));
    pc.colors.push_back(vec3(0.5f, 0.5f, 0.5f));
  }

  knn_init->initialize(pc);

  const auto& gaussians = knn_init->gaussians();
  for (const auto& rot : gaussians.rotations) {
    float norm = std::sqrt(rot.w * rot.w + rot.x * rot.x + rot.y * rot.y + rot.z * rot.z);
    EXPECT_NEAR(norm, 1.0f, 1e-5f);
  }
}

TEST_F(KnnAlgorithmTest, OpacityValuesInRange) {
  PointCloud pc;
  for (int i = 0; i < 10; ++i) {
    pc.points.push_back(vec3(static_cast<float>(i) * 0.1f, 0.0f, 0.0f));
    pc.colors.push_back(vec3(0.5f, 0.5f, 0.5f));
  }

  knn_init->initialize(pc);

  const auto& gaussians = knn_init->gaussians();
  for (const auto& opacity : gaussians.opacities) {
    float activated = activate_opacity(opacity);
    EXPECT_GE(activated, 0.0f);
    EXPECT_LE(activated, 1.0f);
  }
}

TEST_F(KnnAlgorithmTest, IsotropicModeProducesUniformScales) {
  KnnParameters params;
  params.use_anisotropic = false;
  params.enable_radius_outlier_removal = false;
  auto custom_knn = std::make_unique<KnnInitialization>(params);

  PointCloud pc;
  for (int i = 0; i < 10; ++i) {
    pc.points.push_back(vec3(static_cast<float>(i) * 0.1f, 0.0f, 0.0f));
    pc.colors.push_back(vec3(0.5f, 0.5f, 0.5f));
  }

  custom_knn->initialize(pc);

  const auto& gaussians = custom_knn->gaussians();
  for (const auto& scale : gaussians.scales) {
    EXPECT_FLOAT_EQ(scale.x, scale.y);
    EXPECT_FLOAT_EQ(scale.y, scale.z);
  }
}

TEST_F(KnnAlgorithmTest, EmptyPointCloud) {
  PointCloud pc;

  knn_init->initialize(pc);

  const auto& gaussians = knn_init->gaussians();
  EXPECT_EQ(gaussians.means.size(), 0);
}

TEST_F(KnnAlgorithmTest, MeanPositionsMatchPointCloud) {
  PointCloud pc;
  pc.points.push_back(vec3(1.0f, 2.0f, 3.0f));
  pc.points.push_back(vec3(4.0f, 5.0f, 6.0f));
  pc.points.push_back(vec3(7.0f, 8.0f, 9.0f));
  pc.colors.push_back(vec3(0.5f, 0.5f, 0.5f));
  pc.colors.push_back(vec3(0.5f, 0.5f, 0.5f));
  pc.colors.push_back(vec3(0.5f, 0.5f, 0.5f));

  knn_init->initialize(pc);

  const auto& gaussians = knn_init->gaussians();
  EXPECT_EQ(gaussians.means.size(), 3);

  EXPECT_FLOAT_EQ(gaussians.means[0].x, 1.0f);
  EXPECT_FLOAT_EQ(gaussians.means[0].y, 2.0f);
  EXPECT_FLOAT_EQ(gaussians.means[0].z, 3.0f);
}

TEST_F(KnnAlgorithmTest, ShCoefficientsInitialized) {
  PointCloud pc;
  for (int i = 0; i < 10; ++i) {
    pc.points.push_back(vec3(static_cast<float>(i) * 0.1f, 0.0f, 0.0f));
    pc.colors.push_back(vec3(0.5f, 0.5f, 0.5f));
  }

  knn_init->initialize(pc);

  const auto& gaussians = knn_init->gaussians();
  EXPECT_EQ(gaussians.sh0.size(), 10);

  for (const auto& sh : gaussians.sh0) {
    EXPECT_TRUE(std::isfinite(sh.x));
    EXPECT_TRUE(std::isfinite(sh.y));
    EXPECT_TRUE(std::isfinite(sh.z));
  }
}

TEST_F(KnnAlgorithmTest, CustomInitOpacity) {
  KnnParameters params;
  params.init_opacity = 0.8f;
  params.enable_radius_outlier_removal = false;
  auto custom_knn = std::make_unique<KnnInitialization>(params);

  PointCloud pc;
  for (int i = 0; i < 10; ++i) {
    pc.points.push_back(vec3(static_cast<float>(i) * 0.1f, 0.0f, 0.0f));
    pc.colors.push_back(vec3(0.5f, 0.5f, 0.5f));
  }

  custom_knn->initialize(pc);

  const auto& gaussians = custom_knn->gaussians();
  for (const auto& opacity : gaussians.opacities) {
    float activated = activate_opacity(opacity);
    EXPECT_NEAR(activated, 0.8f, 0.01f);
  }
}

TEST_F(KnnAlgorithmTest, CustomInitScaling) {
  KnnParameters params;
  params.init_scaling = 2.0f;
  params.enable_radius_outlier_removal = false;
  auto custom_knn = std::make_unique<KnnInitialization>(params);

  PointCloud pc;
  for (int i = 0; i < 10; ++i) {
    pc.points.push_back(vec3(static_cast<float>(i) * 0.1f, 0.0f, 0.0f));
    pc.colors.push_back(vec3(0.5f, 0.5f, 0.5f));
  }

  custom_knn->initialize(pc);

  const auto& gaussians = custom_knn->gaussians();
  for (const auto& scale : gaussians.scales) {
    EXPECT_TRUE(std::isfinite(scale.x));
    EXPECT_TRUE(std::isfinite(scale.y));
    EXPECT_TRUE(std::isfinite(scale.z));
  }
}

TEST_F(KnnAlgorithmTest, ColorAffectsShCoefficients) {
  PointCloud pc;
  pc.points.push_back(vec3(0.0f, 0.0f, 0.0f));
  pc.points.push_back(vec3(1.0f, 0.0f, 0.0f));
  pc.colors.push_back(vec3(1.0f, 0.0f, 0.0f));
  pc.colors.push_back(vec3(0.0f, 1.0f, 0.0f));

  knn_init->initialize(pc);

  const auto& gaussians = knn_init->gaussians();
  EXPECT_EQ(gaussians.sh0.size(), 2);

  EXPECT_GT(gaussians.sh0[0].x, gaussians.sh0[0].y);
  EXPECT_GT(gaussians.sh0[1].y, gaussians.sh0[1].x);
}

}
