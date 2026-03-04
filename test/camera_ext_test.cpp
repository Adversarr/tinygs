#include <gtest/gtest.h>
#include <tinygs/core/camera_ext.hpp>
#include <cmath>

namespace {

using namespace tinygs;

class CameraExtrinsicsInterpolationTest : public ::testing::Test {
protected:
    void SetUp() override {
        quat q1(1.0f, 0.0f, 0.0f, 0.0f);
        vec3 t1(0.0f, 0.0f, 0.0f);
        ext_a = CameraExtrinsics(q1, t1, 1, 100, 0);
        
        quat q2(0.707f, 0.0f, 0.707f, 0.0f);
        vec3 t2(1.0f, 2.0f, 3.0f);
        ext_b = CameraExtrinsics(q2, t2, 2, 200, 0);
    }

    CameraExtrinsics ext_a;
    CameraExtrinsics ext_b;
};

TEST_F(CameraExtrinsicsInterpolationTest, InterpolateAtStart) {
    CameraExtrinsics result = interpolate(ext_a, ext_b, 0.0f);
    
    EXPECT_NEAR(result.m_q.x, ext_a.m_q.x, 1e-6f);
    EXPECT_NEAR(result.m_q.y, ext_a.m_q.y, 1e-6f);
    EXPECT_NEAR(result.m_q.z, ext_a.m_q.z, 1e-6f);
    EXPECT_NEAR(result.m_q.w, ext_a.m_q.w, 1e-6f);
    
    EXPECT_NEAR(result.m_t.x, ext_a.m_t.x, 1e-6f);
    EXPECT_NEAR(result.m_t.y, ext_a.m_t.y, 1e-6f);
    EXPECT_NEAR(result.m_t.z, ext_a.m_t.z, 1e-6f);
}

TEST_F(CameraExtrinsicsInterpolationTest, InterpolateAtEnd) {
    CameraExtrinsics result = interpolate(ext_a, ext_b, 1.0f);
    
    EXPECT_NEAR(result.m_t.x, ext_b.m_t.x, 1e-6f);
    EXPECT_NEAR(result.m_t.y, ext_b.m_t.y, 1e-6f);
    EXPECT_NEAR(result.m_t.z, ext_b.m_t.z, 1e-6f);
}

TEST_F(CameraExtrinsicsInterpolationTest, InterpolateAtMidpoint) {
    CameraExtrinsics result = interpolate(ext_a, ext_b, 0.5f);
    
    EXPECT_NEAR(result.m_t.x, 0.5f, 1e-6f);
    EXPECT_NEAR(result.m_t.y, 1.0f, 1e-6f);
    EXPECT_NEAR(result.m_t.z, 1.5f, 1e-6f);
}

TEST_F(CameraExtrinsicsInterpolationTest, InterpolateTranslationOnly) {
    quat q(1.0f, 0.0f, 0.0f, 0.0f);
    vec3 t1(0.0f, 0.0f, 0.0f);
    vec3 t2(10.0f, 20.0f, 30.0f);
    
    CameraExtrinsics a(q, t1, 1, 100, 0);
    CameraExtrinsics b(q, t2, 2, 200, 0);
    
    CameraExtrinsics result = interpolate(a, b, 0.3f);
    
    EXPECT_NEAR(result.m_t.x, 3.0f, 1e-5f);
    EXPECT_NEAR(result.m_t.y, 6.0f, 1e-5f);
    EXPECT_NEAR(result.m_t.z, 9.0f, 1e-5f);
}

TEST_F(CameraExtrinsicsInterpolationTest, InterpolateClampsBelowZero) {
    CameraExtrinsics result = interpolate(ext_a, ext_b, -0.5f);
    
    EXPECT_NEAR(result.m_t.x, ext_a.m_t.x, 1e-6f);
    EXPECT_NEAR(result.m_t.y, ext_a.m_t.y, 1e-6f);
    EXPECT_NEAR(result.m_t.z, ext_a.m_t.z, 1e-6f);
}

TEST_F(CameraExtrinsicsInterpolationTest, InterpolateClampsAboveOne) {
    CameraExtrinsics result = interpolate(ext_a, ext_b, 1.5f);
    
    EXPECT_NEAR(result.m_t.x, ext_b.m_t.x, 1e-6f);
    EXPECT_NEAR(result.m_t.y, ext_b.m_t.y, 1e-6f);
    EXPECT_NEAR(result.m_t.z, ext_b.m_t.z, 1e-6f);
}

TEST_F(CameraExtrinsicsInterpolationTest, InterpolateRotation) {
    quat q1(1.0f, 0.0f, 0.0f, 0.0f);
    quat q2(0.0f, 1.0f, 0.0f, 0.0f);
    vec3 t(0.0f, 0.0f, 0.0f);
    
    CameraExtrinsics a(q1, t, 1, 100, 0);
    CameraExtrinsics b(q2, t, 2, 200, 0);
    
    CameraExtrinsics result = interpolate(a, b, 0.5f);
    
    float dot = glm::dot(glm::normalize(result.m_q), glm::normalize(q1));
    EXPECT_GT(dot, 0.0f);
    EXPECT_LT(dot, 1.0f);
}

TEST_F(CameraExtrinsicsInterpolationTest, InterpolatePreservesFrameIdx) {
    CameraExtrinsics result = interpolate(ext_a, ext_b, 0.5f);
    
    EXPECT_EQ(result.frame_idx, ext_a.frame_idx);
    EXPECT_EQ(result.timestamp, ext_a.timestamp);
    EXPECT_EQ(result.cam_uid, ext_a.cam_uid);
}

TEST_F(CameraExtrinsicsInterpolationTest, InterpolateShortestPath) {
    quat q1(1.0f, 0.0f, 0.0f, 0.0f);
    quat q2(-1.0f, 0.0f, 0.0f, 0.0f);
    vec3 t(0.0f, 0.0f, 0.0f);
    
    CameraExtrinsics a(q1, t, 1, 100, 0);
    CameraExtrinsics b(q2, t, 2, 200, 0);
    
    CameraExtrinsics result = interpolate(a, b, 0.5f);
    
    EXPECT_NEAR(result.m_q.w, 1.0f, 1e-6f);
    EXPECT_NEAR(result.m_q.x, 0.0f, 1e-6f);
    EXPECT_NEAR(result.m_q.y, 0.0f, 1e-6f);
    EXPECT_NEAR(result.m_q.z, 0.0f, 1e-6f);
}

TEST_F(CameraExtrinsicsInterpolationTest, InterpolateIdenticalExtrinsics) {
    CameraExtrinsics result = interpolate(ext_a, ext_a, 0.5f);
    
    EXPECT_NEAR(result.m_q.x, ext_a.m_q.x, 1e-6f);
    EXPECT_NEAR(result.m_q.y, ext_a.m_q.y, 1e-6f);
    EXPECT_NEAR(result.m_q.z, ext_a.m_q.z, 1e-6f);
    EXPECT_NEAR(result.m_q.w, ext_a.m_q.w, 1e-6f);
    
    EXPECT_NEAR(result.m_t.x, ext_a.m_t.x, 1e-6f);
    EXPECT_NEAR(result.m_t.y, ext_a.m_t.y, 1e-6f);
    EXPECT_NEAR(result.m_t.z, ext_a.m_t.z, 1e-6f);
}

}
