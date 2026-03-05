#include <gtest/gtest.h>

#include "tinygs/platform/backend_factory.hpp"

using namespace tinygs;

TEST(BackendTypesTest, ToString) {
  EXPECT_EQ(to_string(BackendType::Cuda), "cuda");
  EXPECT_EQ(to_string(BackendType::Hip), "hip");
  EXPECT_EQ(to_string(BackendType::Metal), "metal");
}

TEST(BackendTypesTest, FromStringCaseInsensitive) {
  EXPECT_EQ(backend_type_from_string("cuda"), BackendType::Cuda);
  EXPECT_EQ(backend_type_from_string("CUDA"), BackendType::Cuda);
  EXPECT_EQ(backend_type_from_string("Hip"), BackendType::Hip);
  EXPECT_EQ(backend_type_from_string("METAL"), BackendType::Metal);
}

TEST(BackendTypesTest, FromStringUnknownThrows) {
  EXPECT_THROW(backend_type_from_string("vulkan"), std::runtime_error);
}

TEST(BackendTypesTest, BackendConfigRoundTrip) {
  BackendConfig cfg;
  cfg.from_json(json{{"type", "cuda"}, {"device", 3}});

  EXPECT_EQ(cfg.type, BackendType::Cuda);
  EXPECT_EQ(cfg.device, 3);

  const json out = cfg.to_json();
  EXPECT_EQ(out.at("type").get<std::string>(), "cuda");
  EXPECT_EQ(out.at("device").get<int>(), 3);
}

TEST(BackendTypesTest, BackendConfigRequiresType) {
  BackendConfig cfg;
  EXPECT_THROW(cfg.from_json(json{{"device", 0}}), std::runtime_error);
}

TEST(BackendFactoryTest, CreatesCudaContext) {
  BackendConfig cfg;
  cfg.type = BackendType::Cuda;
  cfg.device = 2;

  auto ctx = create_backend_context(cfg);
  ASSERT_NE(ctx, nullptr);
  EXPECT_EQ(ctx->type(), BackendType::Cuda);
  EXPECT_EQ(ctx->device(), 2);
}

TEST(BackendFactoryTest, HipAndMetalNotImplementedYet) {
  BackendConfig hip_cfg;
  hip_cfg.type = BackendType::Hip;
  EXPECT_THROW(create_backend_context(hip_cfg), std::runtime_error);

  BackendConfig metal_cfg;
  metal_cfg.type = BackendType::Metal;
  EXPECT_THROW(create_backend_context(metal_cfg), std::runtime_error);
}
