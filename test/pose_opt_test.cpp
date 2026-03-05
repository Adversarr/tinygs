#include <gtest/gtest.h>
#include "tinygs/pose_opt/pose_opt.hpp"
#include "tinygs/pose_opt/none.hpp"
#include "tinygs/pose_opt/adamw.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>

using namespace tinygs;

class PoseOptNoneTest : public ::testing::Test {
protected:
  void SetUp() override {
    ident_mat = mat4x4(1.0f);
    test_mat = mat4x4(
      1.0f, 0.0f, 0.0f, 0.1f,
      0.0f, 1.0f, 0.0f, 0.2f,
      0.0f, 0.0f, 1.0f, 0.3f,
      0.0f, 0.0f, 0.0f, 1.0f
    );
  }

  mat4x4 ident_mat;
  mat4x4 test_mat;
};

TEST_F(PoseOptNoneTest, QueryReturnsInputMatrix) {
  PoseOptNone opt;
  mat4x4 result = opt.query(123, test_mat);
  
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(result[i][j], test_mat[i][j]);
    }
  }
}

TEST_F(PoseOptNoneTest, UpdateDoesNothing) {
  PoseOptNone opt;
  mat4x4 grad = mat4x4(1.0f);
  
  EXPECT_NO_THROW(opt.update(123, grad, 0.01f));
  
  mat4x4 result = opt.query(123, test_mat);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(result[i][j], test_mat[i][j]);
    }
  }
}

TEST_F(PoseOptNoneTest, GetParamsReturnsTypeNone) {
  PoseOptNone opt;
  json params = opt.get_params();
  
  EXPECT_TRUE(params.is_object());
  EXPECT_EQ(params["type"], "none");
}

TEST_F(PoseOptNoneTest, SetParamsDoesNothing) {
  PoseOptNone opt;
  json params = {{"lr", 0.5f}};
  
  EXPECT_NO_THROW(opt.set_params(params));
}

class PoseOptAdamWTest : public ::testing::Test {
protected:
  void SetUp() override {
    ident_mat = mat4x4(1.0f);
    test_w2c = mat4x4(
      1.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 1.0f
    );
  }

  mat4x4 ident_mat;
  mat4x4 test_w2c;
};

TEST_F(PoseOptAdamWTest, QueryWithIdentityMatrix) {
  PoseOptAdamW opt;
  mat4x4 result = opt.query(1, ident_mat);
  
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(result[i][j], ident_mat[i][j]);
    }
  }
}

TEST_F(PoseOptAdamWTest, DefaultParams) {
  PoseOptAdamW opt;
  json params = opt.get_params();
  
  EXPECT_EQ(params["type"], "adamw");
  EXPECT_FLOAT_EQ(params["lr"].get<float>(), 0.01f);
  EXPECT_FLOAT_EQ(params["beta1"].get<float>(), 0.9f);
  EXPECT_FLOAT_EQ(params["beta2"].get<float>(), 0.999f);
  EXPECT_FLOAT_EQ(params["epsilon"].get<float>(), 1e-8f);
  EXPECT_FLOAT_EQ(params["weight_decay"].get<float>(), 0.01f);
}

TEST_F(PoseOptAdamWTest, SetParamsFromJson) {
  PoseOptAdamW opt;
  json new_params = {
    {"lr", 0.02f},
    {"beta1", 0.85f},
    {"beta2", 0.99f},
    {"epsilon", 1e-7f},
    {"weight_decay", 0.005f}
  };
  
  opt.set_params(new_params);
  json params = opt.get_params();
  
  EXPECT_FLOAT_EQ(params["lr"].get<float>(), 0.02f);
  EXPECT_FLOAT_EQ(params["beta1"].get<float>(), 0.85f);
  EXPECT_FLOAT_EQ(params["beta2"].get<float>(), 0.99f);
  EXPECT_FLOAT_EQ(params["epsilon"].get<float>(), 1e-7f);
  EXPECT_FLOAT_EQ(params["weight_decay"].get<float>(), 0.005f);
}

TEST_F(PoseOptAdamWTest, SetParamsPartialUpdate) {
  PoseOptAdamW opt;
  json new_params = {{"lr", 0.2f}};
  
  opt.set_params(new_params);
  json params = opt.get_params();
  
  EXPECT_FLOAT_EQ(params["lr"].get<float>(), 0.2f);
  EXPECT_FLOAT_EQ(params["beta1"].get<float>(), 0.9f);
  EXPECT_FLOAT_EQ(params["beta2"].get<float>(), 0.999f);
}

TEST_F(PoseOptAdamWTest, UpdateWithZeroGradient) {
  PoseOptAdamW opt;
  mat4x4 zero_grad = mat4x4(0.0f);
  
  opt.query(1, test_w2c);
  opt.update(1, zero_grad, 1.0f);
  
  mat4x4 result = opt.query(1, test_w2c);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(result[i][j], test_w2c[i][j]);
    }
  }
}

TEST_F(PoseOptAdamWTest, MultipleUpdatesConverge) {
  PoseOptAdamW opt;
  mat4x4 grad = mat4x4(0.0f);
  grad[3] = vec4(0.01f, 0.0f, 0.0f, 0.0f);
  
  opt.query(1, test_w2c);
  for (int i = 0; i < 5; ++i) {
    opt.update(1, grad, 1.0f);
  }
  
  mat4x4 result = opt.query(1, test_w2c);
  EXPECT_NE(result[3][0], test_w2c[3][0]);
}

class PoseOptFactoryTest : public ::testing::Test {
};

TEST_F(PoseOptFactoryTest, CreateNone) {
  auto opt = create_pose_opt("none");
  EXPECT_NE(opt, nullptr);
  
  json params = opt->get_params();
  EXPECT_EQ(params["type"], "none");
}

TEST_F(PoseOptFactoryTest, CreateAdamW) {
  auto opt = create_pose_opt("adamw");
  EXPECT_NE(opt, nullptr);
  
  json params = opt->get_params();
  EXPECT_EQ(params["type"], "adamw");
}

TEST_F(PoseOptFactoryTest, CreateCaseInsensitive) {
  auto opt1 = create_pose_opt("NONE");
  EXPECT_NE(opt1, nullptr);
  EXPECT_EQ(opt1->get_params()["type"], "none");
  
  auto opt2 = create_pose_opt("AdamW");
  EXPECT_NE(opt2, nullptr);
  EXPECT_EQ(opt2->get_params()["type"], "adamw");
}

TEST_F(PoseOptFactoryTest, CreateUnknownThrows) {
  EXPECT_THROW(create_pose_opt("unknown"), std::runtime_error);
  EXPECT_THROW(create_pose_opt("invalid"), std::runtime_error);
  EXPECT_THROW(create_pose_opt(""), std::runtime_error);
}

class PoseOptMultiTimestampTest : public ::testing::Test {
protected:
  void SetUp() override {
    w2c_1 = mat4x4(1.0f);
    w2c_2 = glm::translate(mat4x4(1.0f), vec3(1.0f, 0.0f, 0.0f));
  }

  mat4x4 w2c_1;
  mat4x4 w2c_2;
};

TEST_F(PoseOptMultiTimestampTest, AdamWMultipleTimestamps) {
  PoseOptAdamW opt;
  
  opt.query(1, w2c_1);
  opt.query(2, w2c_2);
  
  mat4x4 grad = mat4x4(0.0f);
  grad[3] = vec4(0.05f, 0.0f, 0.0f, 0.0f);
  
  opt.update(1, grad, 1.0f);
  
  mat4x4 r1 = opt.query(1, w2c_1);
  mat4x4 r2 = opt.query(2, w2c_2);
  
  EXPECT_NE(r1[3][0], w2c_1[3][0]);
  EXPECT_FLOAT_EQ(r2[3][0], w2c_2[3][0]);
}
