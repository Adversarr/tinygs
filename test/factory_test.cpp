#include <gtest/gtest.h>
#include <tinygs/dataset/dataset.hpp>
#include <tinygs/rasterizer/rasterizer.hpp>
#include <tinygs/loss/loss.hpp>
#include <tinygs/optim/optim.hpp>
#include <tinygs/platform/backend_build.hpp>
#include <tinygs/platform/runtime_factory.hpp>
#include <tinygs/cuda/common_host.hpp>

namespace {

using namespace tinygs;

bool has_cuda_device() {
  try {
    return cuda_device_count() > 0;
  } catch (...) {
    return false;
  }
}

BackendConfig make_cuda_config() {
  BackendConfig config;
  config.type = compiled_backend_type();
  config.device = 0;
  return config;
}

std::shared_ptr<BackendRuntime> make_runtime() {
  const auto runtime_result = create_backend_runtime(make_cuda_config());
  if (!runtime_result.ok()) {
    return nullptr;
  }
  return runtime_result.value();
}

TEST(CreateDatasetTest, CreatesImageDataset) {
    auto runtime = make_runtime();
    if (!runtime) {
      GTEST_SKIP() << "Failed to create backend runtime.";
    }
    auto dataset = create_dataset("image", *runtime);
    EXPECT_NE(dataset, nullptr);
}

TEST(CreateDatasetTest, CreatesCaseInsensitive) {
    auto runtime = make_runtime();
    if (!runtime) {
      GTEST_SKIP() << "Failed to create backend runtime.";
    }
    auto d1 = create_dataset("IMAGE", *runtime);
    auto d2 = create_dataset("Image", *runtime);
    auto d3 = create_dataset("ImAgE", *runtime);
    
    EXPECT_NE(d1, nullptr);
    EXPECT_NE(d2, nullptr);
    EXPECT_NE(d3, nullptr);
}

TEST(CreateDatasetTest, ThrowsOnUnknownType) {
    auto runtime = make_runtime();
    if (!runtime) {
      GTEST_SKIP() << "Failed to create backend runtime.";
    }
    EXPECT_THROW(create_dataset("unknown", *runtime), std::runtime_error);
    EXPECT_THROW(create_dataset("invalid_type", *runtime), std::runtime_error);
    EXPECT_THROW(create_dataset("colmap", *runtime), std::runtime_error);
}

TEST(CreateRasterizerTest, CreatesFastGSRasterizer) {
    auto runtime = make_runtime();
    if (!runtime) {
      GTEST_SKIP() << "Failed to create backend runtime.";
    }
    auto rasterizer = create_rasterizer("fastgs", *runtime);
    EXPECT_NE(rasterizer, nullptr);
}

TEST(CreateRasterizerTest, CreatesCaseInsensitive) {
    auto runtime = make_runtime();
    if (!runtime) {
      GTEST_SKIP() << "Failed to create backend runtime.";
    }
    auto r1 = create_rasterizer("FASTGS", *runtime);
    auto r2 = create_rasterizer("FastGS", *runtime);
    
    EXPECT_NE(r1, nullptr);
    EXPECT_NE(r2, nullptr);
}

TEST(CreateRasterizerTest, ThrowsOnUnknownType) {
    auto runtime = make_runtime();
    if (!runtime) {
      GTEST_SKIP() << "Failed to create backend runtime.";
    }
    EXPECT_THROW(create_rasterizer("unknown", *runtime), std::runtime_error);
    EXPECT_THROW(create_rasterizer("invalid", *runtime), std::runtime_error);
    EXPECT_THROW(create_rasterizer("gsplat", *runtime), std::runtime_error);
}

class CreateLossTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!has_cuda_device()) {
      GTEST_SKIP() << "No CUDA device available.";
    }
    runtime = make_runtime();
    if (!runtime) {
      GTEST_SKIP() << "Failed to create backend runtime.";
    }
  }
  std::shared_ptr<BackendRuntime> runtime;
};

TEST_F(CreateLossTest, CreatesL1Loss) {
    auto loss = create_loss(*runtime, "l1");
    EXPECT_NE(loss, nullptr);
    EXPECT_EQ(loss->name(), "l1");
}

TEST_F(CreateLossTest, CreatesL2Loss) {
    auto loss = create_loss(*runtime, "l2");
    EXPECT_NE(loss, nullptr);
    EXPECT_EQ(loss->name(), "l2");
}

TEST_F(CreateLossTest, CreatesHuberLoss) {
    auto loss = create_loss(*runtime, "huber");
    EXPECT_NE(loss, nullptr);
    EXPECT_EQ(loss->name(), "huber");
}

TEST_F(CreateLossTest, CreatesFusedSSIMLoss) {
    auto loss = create_loss(*runtime, "fused_ssim");
    EXPECT_NE(loss, nullptr);
    EXPECT_EQ(loss->name(), "fused_ssim");
}

TEST_F(CreateLossTest, CreatesCaseInsensitive) {
    auto l1 = create_loss(*runtime, "L1");
    auto l2 = create_loss(*runtime, "L2");
    auto huber = create_loss(*runtime, "HUBER");
    auto ssim = create_loss(*runtime, "FUSED_SSIM");
    
    EXPECT_NE(l1, nullptr);
    EXPECT_NE(l2, nullptr);
    EXPECT_NE(huber, nullptr);
    EXPECT_NE(ssim, nullptr);
}

TEST_F(CreateLossTest, ThrowsOnUnknownType) {
    EXPECT_THROW(create_loss(*runtime, "unknown"), std::runtime_error);
    EXPECT_THROW(create_loss(*runtime, "mse"), std::runtime_error);
    EXPECT_THROW(create_loss(*runtime, "cross_entropy"), std::runtime_error);
}

class CreateMetricTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!has_cuda_device()) {
      GTEST_SKIP() << "No CUDA device available.";
    }
    runtime = make_runtime();
    if (!runtime) {
      GTEST_SKIP() << "Failed to create backend runtime.";
    }
  }
  std::shared_ptr<BackendRuntime> runtime;
};

TEST_F(CreateMetricTest, CreatesPSNRMetric) {
    auto metric = create_metric(*runtime, "psnr");
    EXPECT_NE(metric, nullptr);
    EXPECT_EQ(metric->name(), "psnr");
}

TEST_F(CreateMetricTest, CreatesCaseInsensitive) {
    auto m1 = create_metric(*runtime, "PSNR");
    auto m2 = create_metric(*runtime, "Psnr");
    
    EXPECT_NE(m1, nullptr);
    EXPECT_NE(m2, nullptr);
}

TEST_F(CreateMetricTest, ThrowsOnUnknownType) {
    EXPECT_THROW(create_metric(*runtime, "unknown"), std::runtime_error);
    EXPECT_THROW(create_metric(*runtime, "ssim"), std::runtime_error);
    EXPECT_THROW(create_metric(*runtime, "mse"), std::runtime_error);
}

}
