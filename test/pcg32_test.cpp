#include <gtest/gtest.h>
#include <tinygs/random/pcg32.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace {

using namespace tinygs;

TEST(PCG32Test, DefaultConstructor) {
    pcg32 rng;
    EXPECT_NE(rng.state, 0ULL);
    EXPECT_NE(rng.inc, 0ULL);
}

TEST(PCG32Test, SeededConstructor) {
    pcg32 rng(42ULL, 1ULL);
    pcg32 seeded;
    seeded.seed(42ULL, 1ULL);
    EXPECT_EQ(rng.state, seeded.state);
    EXPECT_EQ(rng.inc, seeded.inc);
}

TEST(PCG32Test, SeedResetsState) {
    pcg32 rng(12345ULL, 1ULL);
    
    uint32_t first = rng.next_uint();
    uint32_t second = rng.next_uint();
    
    rng.seed(12345ULL, 1ULL);
    uint32_t after_reseed = rng.next_uint();
    
    EXPECT_EQ(after_reseed, first);
}

TEST(PCG32Test, NextUintGeneratesValues) {
    pcg32 rng(42ULL);
    
    uint32_t v1 = rng.next_uint();
    uint32_t v2 = rng.next_uint();
    uint32_t v3 = rng.next_uint();
    
    EXPECT_NE(v1, v2);
    EXPECT_NE(v2, v3);
    EXPECT_NE(v1, v3);
}

TEST(PCG32Test, NextUintBoundedInRange) {
    pcg32 rng(42ULL);
    
    for (int i = 0; i < 1000; ++i) {
        uint32_t bound = 100;
        uint32_t value = rng.next_uint(bound);
        EXPECT_LT(value, bound);
    }
}

TEST(PCG32Test, NextUintBoundedSmallBound) {
    pcg32 rng(42ULL);
    
    for (int i = 0; i < 100; ++i) {
        uint32_t bound = 2;
        uint32_t value = rng.next_uint(bound);
        EXPECT_LT(value, bound);
    }
}

TEST(PCG32Test, NextFloatInRange) {
    pcg32 rng(42ULL);
    
    for (int i = 0; i < 1000; ++i) {
        float value = rng.next_float();
        EXPECT_GE(value, 0.0f);
        EXPECT_LT(value, 1.0f);
    }
}

TEST(PCG32Test, NextDoubleInRange) {
    pcg32 rng(42ULL);
    
    for (int i = 0; i < 1000; ++i) {
        double value = rng.next_double();
        EXPECT_GE(value, 0.0);
        EXPECT_LT(value, 1.0);
    }
}

TEST(PCG32Test, NextFloatDistribution) {
    pcg32 rng(42ULL);
    
    std::vector<int> buckets(10, 0);
    const int num_samples = 10000;
    
    for (int i = 0; i < num_samples; ++i) {
        float value = rng.next_float();
        int bucket = static_cast<int>(value * 10.0f);
        bucket = std::min(bucket, 9);
        buckets[bucket]++;
    }
    
    for (int count : buckets) {
        double ratio = static_cast<double>(count) / num_samples;
        EXPECT_GT(ratio, 0.08);
        EXPECT_LT(ratio, 0.12);
    }
}

TEST(PCG32Test, NextDoubleDistribution) {
    pcg32 rng(42ULL);
    
    std::vector<int> buckets(10, 0);
    const int num_samples = 10000;
    
    for (int i = 0; i < num_samples; ++i) {
        double value = rng.next_double();
        int bucket = static_cast<int>(value * 10.0);
        bucket = std::min(bucket, 9);
        buckets[bucket]++;
    }
    
    for (int count : buckets) {
        double ratio = static_cast<double>(count) / num_samples;
        EXPECT_GT(ratio, 0.08);
        EXPECT_LT(ratio, 0.12);
    }
}

TEST(PCG32Test, EqualityOperator) {
    pcg32 a(42ULL, 1ULL);
    pcg32 b(42ULL, 1ULL);
    pcg32 c(43ULL, 1ULL);
    
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    
    a.next_uint();
    EXPECT_FALSE(a == b);
}

TEST(PCG32Test, InequalityOperator) {
    pcg32 a(42ULL, 1ULL);
    pcg32 b(42ULL, 1ULL);
    pcg32 c(43ULL, 1ULL);
    
    EXPECT_FALSE(a != b);
    EXPECT_TRUE(a != c);
}

TEST(PCG32Test, AdvanceForward) {
    pcg32 rng(42ULL);
    
    rng.advance(100);
    
    pcg32 rng2(42ULL);
    for (int i = 0; i < 100; ++i) {
        rng2.next_uint();
    }
    
    EXPECT_EQ(rng.state, rng2.state);
}

TEST(PCG32Test, AdvanceBackward) {
    pcg32 rng(42ULL);
    
    uint32_t v1 = rng.next_uint();
    rng.next_uint();
    rng.next_uint();
    
    rng.advance(-3);
    uint32_t v2 = rng.next_uint();
    
    EXPECT_EQ(v1, v2);
}

TEST(PCG32Test, DistanceBetweenGenerators) {
    pcg32 rng1(42ULL);
    pcg32 rng2(42ULL);
    
    for (int i = 0; i < 100; ++i) {
        rng2.next_uint();
    }
    
    int64_t dist = rng2 - rng1;
    EXPECT_EQ(dist, 100);
}

TEST(PCG32Test, DistanceZero) {
    pcg32 rng1(42ULL);
    pcg32 rng2(42ULL);
    
    int64_t dist = rng2 - rng1;
    EXPECT_EQ(dist, 0);
}

TEST(PCG32Test, Reproducibility) {
    pcg32 rng1(12345ULL, 999ULL);
    pcg32 rng2(12345ULL, 999ULL);
    
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(rng1.next_uint(), rng2.next_uint());
        EXPECT_FLOAT_EQ(rng1.next_float(), rng2.next_float());
        EXPECT_DOUBLE_EQ(rng1.next_double(), rng2.next_double());
    }
}

TEST(PCG32Test, DifferentSeedsProduceDifferentSequences) {
    pcg32 rng1(1ULL);
    pcg32 rng2(2ULL);
    
    bool any_different = false;
    for (int i = 0; i < 100; ++i) {
        if (rng1.next_uint() != rng2.next_uint()) {
            any_different = true;
            break;
        }
    }
    EXPECT_TRUE(any_different);
}

TEST(PCG32Test, DifferentSequencesProduceDifferentResults) {
    pcg32 rng1(42ULL, 1ULL);
    pcg32 rng2(42ULL, 2ULL);
    
    bool any_different = false;
    for (int i = 0; i < 100; ++i) {
        if (rng1.next_uint() != rng2.next_uint()) {
            any_different = true;
            break;
        }
    }
    EXPECT_TRUE(any_different);
}

}
