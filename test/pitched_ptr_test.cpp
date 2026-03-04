#include <gtest/gtest.h>
#include <tinygs/common.hpp>
#include <vector>
#include <cstdint>

namespace {

using namespace tinygs;

TEST(PitchedPtrTest, DefaultConstructor) {
    PitchedPtr<float> ptr;
    EXPECT_EQ(ptr.ptr, nullptr);
    EXPECT_EQ(ptr.stride_in_bytes, sizeof(float));
}

TEST(PitchedPtrTest, ConstructorWithPtrAndStride) {
    float data[100];
    PitchedPtr<float> ptr(data, 10);
    
    EXPECT_EQ(ptr.ptr, data);
    EXPECT_EQ(ptr.stride_in_bytes, 10 * sizeof(float));
}

TEST(PitchedPtrTest, ConstructorWithOffset) {
    float data[100];
    PitchedPtr<float> ptr(data, 10, 5);
    
    EXPECT_EQ(ptr.ptr, data + 5);
    EXPECT_EQ(ptr.stride_in_bytes, 10 * sizeof(float));
}

TEST(PitchedPtrTest, ConstructorWithExtraStride) {
    float data[100];
    PitchedPtr<float> ptr(data, 10, 0, 16);
    
    EXPECT_EQ(ptr.ptr, data);
    EXPECT_EQ(ptr.stride_in_bytes, 10 * sizeof(float) + 16);
}

TEST(PitchedPtrTest, ConversionConstructor) {
    float data[100];
    PitchedPtr<float> float_ptr(data, 10);
    PitchedPtr<const float> const_ptr(float_ptr);
    
    EXPECT_EQ(const_ptr.ptr, data);
    EXPECT_EQ(const_ptr.stride_in_bytes, 10 * sizeof(float));
}

TEST(PitchedPtrTest, OperatorCall) {
    std::vector<float> data(100, 0.0f);
    PitchedPtr<float> ptr(data.data(), 10);
    
    ptr(0)[0] = 1.0f;
    ptr(1)[0] = 2.0f;
    ptr(2)[5] = 3.0f;
    
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[10], 2.0f);
    EXPECT_FLOAT_EQ(data[25], 3.0f);
}

TEST(PitchedPtrTest, OperatorPlusEquals) {
    std::vector<float> data(100, 0.0f);
    PitchedPtr<float> ptr(data.data(), 10);
    
    ptr += 3;
    
    EXPECT_EQ(ptr.ptr, data.data() + 30);
}

TEST(PitchedPtrTest, OperatorMinusEquals) {
    std::vector<float> data(100, 0.0f);
    PitchedPtr<float> ptr(data.data() + 50, 10);
    
    ptr -= 3;
    
    EXPECT_EQ(ptr.ptr, data.data() + 20);
}

TEST(PitchedPtrTest, BoolOperator) {
    std::vector<float> data(10);
    PitchedPtr<float> valid_ptr(data.data(), 5);
    PitchedPtr<float> null_ptr;
    
    EXPECT_TRUE(static_cast<bool>(valid_ptr));
    EXPECT_FALSE(static_cast<bool>(null_ptr));
}

TEST(PitchedPtrTest, StrideInBytesCalculation) {
    float data[100];
    
    PitchedPtr<float> ptr1(data, 1);
    EXPECT_EQ(ptr1.stride_in_bytes, sizeof(float));
    
    PitchedPtr<float> ptr2(data, 8);
    EXPECT_EQ(ptr2.stride_in_bytes, 8 * sizeof(float));
    
    PitchedPtr<int> ptr3(reinterpret_cast<int*>(data), 4);
    EXPECT_EQ(ptr3.stride_in_bytes, 4 * sizeof(int));
}

TEST(PitchedPtrTest, RowAccessWithCustomStride) {
    std::vector<float> data(100, 0.0f);
    size_t stride = 12;
    PitchedPtr<float> ptr(data.data(), stride);
    
    ptr(0)[0] = 1.0f;
    ptr(1)[0] = 2.0f;
    ptr(2)[0] = 3.0f;
    
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[12], 2.0f);
    EXPECT_FLOAT_EQ(data[24], 3.0f);
}

TEST(PayloadAndIdxTest, Initialization) {
    PayloadAndIdx<float> payload{3.14f, 42};
    EXPECT_FLOAT_EQ(payload.t, 3.14f);
    EXPECT_EQ(payload.idx, 42);
}

TEST(PayloadAndIdxTest, ComparisonOperatorAscending) {
    PayloadAndIdx<float> a{1.0f, 0};
    PayloadAndIdx<float> b{2.0f, 1};
    PayloadAndIdx<float> c{3.0f, 2};
    
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(b < c);
    EXPECT_TRUE(a < c);
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(c < a);
}

TEST(PayloadAndIdxTest, SortingOrder) {
    std::vector<PayloadAndIdx<float>> payloads = {
        {5.0f, 0},
        {2.0f, 1},
        {8.0f, 2},
        {1.0f, 3},
        {3.0f, 4}
    };
    
    std::sort(payloads.begin(), payloads.end());
    
    EXPECT_FLOAT_EQ(payloads[0].t, 1.0f);
    EXPECT_FLOAT_EQ(payloads[1].t, 2.0f);
    EXPECT_FLOAT_EQ(payloads[2].t, 3.0f);
    EXPECT_FLOAT_EQ(payloads[3].t, 5.0f);
    EXPECT_FLOAT_EQ(payloads[4].t, 8.0f);
}

TEST(PayloadAndIdxTest, IntType) {
    PayloadAndIdx<int> payload{100, 5};
    EXPECT_EQ(payload.t, 100);
    EXPECT_EQ(payload.idx, 5);
}

TEST(PayloadAndIdxTest, DoubleType) {
    PayloadAndIdx<double> payload{1.5, 10};
    EXPECT_DOUBLE_EQ(payload.t, 1.5);
    EXPECT_EQ(payload.idx, 10);
}

}
