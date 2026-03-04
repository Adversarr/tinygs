#include <gtest/gtest.h>
#include <tinygs/core/camera.hpp>
#include <cmath>

namespace {

using namespace tinygs;

TEST(CameraIntrinsicsTest, ParseMinimalLine) {
  std::string line = "1 PINHOLE 1920 1080 1000.0 1000.0 960.0 540.0";
  auto intr = CameraIntrinsics::parse(line);
  
  EXPECT_EQ(intr.uid, 1ULL);
  EXPECT_EQ(intr.model, CameraModel::Pinhole);
  EXPECT_EQ(intr.width, 1920);
  EXPECT_EQ(intr.height, 1080);
  EXPECT_FLOAT_EQ(intr.fx, 1000.0f);
  EXPECT_FLOAT_EQ(intr.fy, 1000.0f);
  EXPECT_FLOAT_EQ(intr.cx, 960.0f);
  EXPECT_FLOAT_EQ(intr.cy, 540.0f);
  EXPECT_FLOAT_EQ(intr.k1, 0.0f);
  EXPECT_FLOAT_EQ(intr.k2, 0.0f);
  EXPECT_FLOAT_EQ(intr.k3, 0.0f);
  EXPECT_FLOAT_EQ(intr.p1, 0.0f);
  EXPECT_FLOAT_EQ(intr.p2, 0.0f);
}

TEST(CameraIntrinsicsTest, ParseWithDistortion) {
  std::string line = "2 PINHOLE 800 600 500.0 500.0 400.0 300.0 0.1 0.2 0.3 0.01 0.02";
  auto intr = CameraIntrinsics::parse(line);
  
  EXPECT_EQ(intr.uid, 2ULL);
  EXPECT_EQ(intr.model, CameraModel::Pinhole);
  EXPECT_EQ(intr.width, 800);
  EXPECT_EQ(intr.height, 600);
  EXPECT_FLOAT_EQ(intr.fx, 500.0f);
  EXPECT_FLOAT_EQ(intr.fy, 500.0f);
  EXPECT_FLOAT_EQ(intr.cx, 400.0f);
  EXPECT_FLOAT_EQ(intr.cy, 300.0f);
  EXPECT_FLOAT_EQ(intr.k1, 0.1f);
  EXPECT_FLOAT_EQ(intr.k2, 0.2f);
  EXPECT_FLOAT_EQ(intr.k3, 0.3f);
  EXPECT_FLOAT_EQ(intr.p1, 0.01f);
  EXPECT_FLOAT_EQ(intr.p2, 0.02f);
}

TEST(CameraIntrinsicsTest, ParseInvalidThrows) {
  EXPECT_THROW(CameraIntrinsics::parse("1 PINHOLE 1920 1080"), std::runtime_error);
  EXPECT_THROW(CameraIntrinsics::parse("1 UNKNOWN 1920 1080 1000 1000 960 540"), std::runtime_error);
  EXPECT_THROW(CameraIntrinsics::parse("1 PINHOLE 1920 1080 1000 1000 960 540 0.1 0.2"), std::runtime_error);
}

TEST(CameraIntrinsicsTest, ToMat3) {
  CameraIntrinsics intr;
  intr.fx = 1000.0f;
  intr.fy = 1100.0f;
  intr.cx = 960.0f;
  intr.cy = 540.0f;
  
  mat3x3 K = intr.to_mat3();
  
  EXPECT_FLOAT_EQ(K[0][0], 1000.0f);
  EXPECT_FLOAT_EQ(K[1][1], 1100.0f);
  EXPECT_FLOAT_EQ(K[2][0], 960.0f);
  EXPECT_FLOAT_EQ(K[2][1], 540.0f);
  EXPECT_FLOAT_EQ(K[2][2], 1.0f);
  EXPECT_FLOAT_EQ(K[0][1], 0.0f);
  EXPECT_FLOAT_EQ(K[0][2], 0.0f);
  EXPECT_FLOAT_EQ(K[1][0], 0.0f);
  EXPECT_FLOAT_EQ(K[1][2], 0.0f);
}

TEST(CameraIntrinsicsTest, ToString) {
  CameraIntrinsics intr;
  intr.uid = 1;
  intr.model = CameraModel::Pinhole;
  intr.width = 1920;
  intr.height = 1080;
  intr.fx = 1000.0f;
  intr.fy = 1000.0f;
  intr.cx = 960.0f;
  intr.cy = 540.0f;
  intr.k1 = 0.0f;
  intr.k2 = 0.0f;
  intr.k3 = 0.0f;
  intr.p1 = 0.0f;
  intr.p2 = 0.0f;
  
  std::string str = intr.to_string();
  EXPECT_TRUE(str.find("PINHOLE") != std::string::npos);
  EXPECT_TRUE(str.find("1920") != std::string::npos);
  EXPECT_TRUE(str.find("1080") != std::string::npos);
}

TEST(CameraExtrinsicsTest, ParseBasicLine) {
  std::string line = "1 1.0 0.0 0.0 0.0 0.1 0.2 0.3 0 12345";
  auto extr = CameraExtrinsics::parse(line);
  
  EXPECT_EQ(extr.frame_idx, 1ULL);
  EXPECT_EQ(extr.cam_uid, 0ULL);
  EXPECT_EQ(extr.timestamp, 12345ULL);
  EXPECT_FLOAT_EQ(extr.m_q.w, 1.0f);
  EXPECT_FLOAT_EQ(extr.m_q.x, 0.0f);
  EXPECT_FLOAT_EQ(extr.m_q.y, 0.0f);
  EXPECT_FLOAT_EQ(extr.m_q.z, 0.0f);
  EXPECT_FLOAT_EQ(extr.m_t.x, 0.1f);
  EXPECT_FLOAT_EQ(extr.m_t.y, 0.2f);
  EXPECT_FLOAT_EQ(extr.m_t.z, 0.3f);
}

TEST(CameraExtrinsicsTest, ParseWithTimestampFromFilename) {
  std::string line = "10 0.707 0.0 0.707 0.0 1.0 2.0 3.0 1 00123456.jpg";
  auto extr = CameraExtrinsics::parse(line);
  
  EXPECT_EQ(extr.frame_idx, 10ULL);
  EXPECT_EQ(extr.timestamp, 123456ULL);
}

TEST(CameraExtrinsicsTest, ParseInvalidThrows) {
  EXPECT_THROW(CameraExtrinsics::parse("1 1.0 0.0 0.0"), std::runtime_error);
  EXPECT_THROW(CameraExtrinsics::parse("short"), std::runtime_error);
}

TEST(CameraExtrinsicsTest, QuaternionNormalization) {
  std::string line = "1 2.0 0.0 0.0 0.0 0.0 0.0 0.0 0 0";
  auto extr = CameraExtrinsics::parse(line);
  
  float norm = std::sqrt(extr.m_q.w * extr.m_q.w + 
                         extr.m_q.x * extr.m_q.x + 
                         extr.m_q.y * extr.m_q.y + 
                         extr.m_q.z * extr.m_q.z);
  EXPECT_NEAR(norm, 1.0f, 1e-6f);
}

TEST(CameraExtrinsicsTest, GetW2CAndC2W) {
  quat q(1.0f, 0.0f, 0.0f, 0.0f);
  vec3 t(0.1f, 0.2f, 0.3f);
  CameraExtrinsics extr(q, t, 1, 0, 0);
  
  mat4x4 w2c = extr.get_w2c();
  mat4x4 c2w = extr.get_c2w();
  
  EXPECT_FLOAT_EQ(w2c[3][0], 0.1f);
  EXPECT_FLOAT_EQ(w2c[3][1], 0.2f);
  EXPECT_FLOAT_EQ(w2c[3][2], 0.3f);
  
  EXPECT_FLOAT_EQ(c2w[3][0], -0.1f);
  EXPECT_FLOAT_EQ(c2w[3][1], -0.2f);
  EXPECT_FLOAT_EQ(c2w[3][2], -0.3f);
}

TEST(MakeW2CTest, FromMatrixAndQuaternion) {
  mat3x3 rot = mat3x3(1.0f);
  vec3 t(1.0f, 2.0f, 3.0f);
  
  mat4x4 w2c = make_w2c(rot, t);
  EXPECT_FLOAT_EQ(w2c[3][0], 1.0f);
  EXPECT_FLOAT_EQ(w2c[3][1], 2.0f);
  EXPECT_FLOAT_EQ(w2c[3][2], 3.0f);
  
  quat q(1.0f, 0.0f, 0.0f, 0.0f);
  mat4x4 w2c2 = make_w2c(q, t);
  EXPECT_FLOAT_EQ(w2c2[3][0], 1.0f);
  EXPECT_FLOAT_EQ(w2c2[3][1], 2.0f);
  EXPECT_FLOAT_EQ(w2c2[3][2], 3.0f);
}

TEST(CameraTest, GetPosition) {
  CameraIntrinsics intr;
  intr.uid = 0;
  intr.model = CameraModel::Pinhole;
  intr.width = 800;
  intr.height = 600;
  intr.fx = intr.fy = 500.0f;
  intr.cx = 400.0f;
  intr.cy = 300.0f;
  
  quat q(1.0f, 0.0f, 0.0f, 0.0f);
  vec3 t(0.0f, 0.0f, 0.0f);
  CameraExtrinsics extr(q, t, 0, 0, 0);
  
  Camera cam(intr, extr);
  vec3 pos = cam.get_position();
  EXPECT_FLOAT_EQ(pos.x, 0.0f);
  EXPECT_FLOAT_EQ(pos.y, 0.0f);
  EXPECT_FLOAT_EQ(pos.z, 0.0f);
}

TEST(CameraTest, GetDirections) {
  CameraIntrinsics intr;
  intr.uid = 0;
  intr.model = CameraModel::Pinhole;
  intr.width = 800;
  intr.height = 600;
  
  quat q(1.0f, 0.0f, 0.0f, 0.0f);
  vec3 t(0.0f, 0.0f, 0.0f);
  CameraExtrinsics extr(q, t, 0, 0, 0);
  
  Camera cam(intr, extr);
  
  vec3 forward = cam.get_forward();
  vec3 up = cam.get_up();
  vec3 right = cam.get_right();
  
  EXPECT_FLOAT_EQ(forward.x, 0.0f);
  EXPECT_FLOAT_EQ(forward.y, 0.0f);
  EXPECT_FLOAT_EQ(forward.z, -1.0f);
  
  EXPECT_FLOAT_EQ(up.x, 0.0f);
  EXPECT_FLOAT_EQ(up.y, 1.0f);
  EXPECT_FLOAT_EQ(up.z, 0.0f);
  
  EXPECT_FLOAT_EQ(right.x, 1.0f);
  EXPECT_FLOAT_EQ(right.y, 0.0f);
  EXPECT_FLOAT_EQ(right.z, 0.0f);
}

}
