#include <gtest/gtest.h>
#include <tinygs/dataset/dataset.hpp>
#include <tinygs/rasterizer/rasterizer.hpp>
#include <tinygs/loss/loss.hpp>
#include <tinygs/optim/optim.hpp>

namespace {

using namespace tinygs;

TEST(CreateDatasetTest, CreatesImageDataset) {
    auto dataset = create_dataset("image");
    EXPECT_NE(dataset, nullptr);
}

TEST(CreateDatasetTest, CreatesCaseInsensitive) {
    auto d1 = create_dataset("IMAGE");
    auto d2 = create_dataset("Image");
    auto d3 = create_dataset("ImAgE");
    
    EXPECT_NE(d1, nullptr);
    EXPECT_NE(d2, nullptr);
    EXPECT_NE(d3, nullptr);
}

TEST(CreateDatasetTest, ThrowsOnUnknownType) {
    EXPECT_THROW(create_dataset("unknown"), std::runtime_error);
    EXPECT_THROW(create_dataset("invalid_type"), std::runtime_error);
    EXPECT_THROW(create_dataset("colmap"), std::runtime_error);
}

TEST(CreateRasterizerTest, CreatesFastGSRasterizer) {
    auto rasterizer = create_rasterizer("fastgs");
    EXPECT_NE(rasterizer, nullptr);
}

TEST(CreateRasterizerTest, CreatesCaseInsensitive) {
    auto r1 = create_rasterizer("FASTGS");
    auto r2 = create_rasterizer("FastGS");
    
    EXPECT_NE(r1, nullptr);
    EXPECT_NE(r2, nullptr);
}

TEST(CreateRasterizerTest, ThrowsOnUnknownType) {
    EXPECT_THROW(create_rasterizer("unknown"), std::runtime_error);
    EXPECT_THROW(create_rasterizer("invalid"), std::runtime_error);
    EXPECT_THROW(create_rasterizer("gsplat"), std::runtime_error);
}

TEST(CreateLossTest, CreatesL1Loss) {
    auto loss = create_loss("l1");
    EXPECT_NE(loss, nullptr);
    EXPECT_EQ(loss->name(), "l1");
}

TEST(CreateLossTest, CreatesL2Loss) {
    auto loss = create_loss("l2");
    EXPECT_NE(loss, nullptr);
    EXPECT_EQ(loss->name(), "l2");
}

TEST(CreateLossTest, CreatesHuberLoss) {
    auto loss = create_loss("huber");
    EXPECT_NE(loss, nullptr);
    EXPECT_EQ(loss->name(), "huber");
}

TEST(CreateLossTest, CreatesFusedSSIMLoss) {
    auto loss = create_loss("fused_ssim");
    EXPECT_NE(loss, nullptr);
    EXPECT_EQ(loss->name(), "fused_ssim");
}

TEST(CreateLossTest, CreatesCaseInsensitive) {
    auto l1 = create_loss("L1");
    auto l2 = create_loss("L2");
    auto huber = create_loss("HUBER");
    auto ssim = create_loss("FUSED_SSIM");
    
    EXPECT_NE(l1, nullptr);
    EXPECT_NE(l2, nullptr);
    EXPECT_NE(huber, nullptr);
    EXPECT_NE(ssim, nullptr);
}

TEST(CreateLossTest, ThrowsOnUnknownType) {
    EXPECT_THROW(create_loss("unknown"), std::runtime_error);
    EXPECT_THROW(create_loss("mse"), std::runtime_error);
    EXPECT_THROW(create_loss("cross_entropy"), std::runtime_error);
}

TEST(CreateMetricTest, CreatesPSNRMetric) {
    auto metric = create_metric("psnr");
    EXPECT_NE(metric, nullptr);
    EXPECT_EQ(metric->name(), "psnr");
}

TEST(CreateMetricTest, CreatesCaseInsensitive) {
    auto m1 = create_metric("PSNR");
    auto m2 = create_metric("Psnr");
    
    EXPECT_NE(m1, nullptr);
    EXPECT_NE(m2, nullptr);
}

TEST(CreateMetricTest, ThrowsOnUnknownType) {
    EXPECT_THROW(create_metric("unknown"), std::runtime_error);
    EXPECT_THROW(create_metric("ssim"), std::runtime_error);
    EXPECT_THROW(create_metric("mse"), std::runtime_error);
}

}
