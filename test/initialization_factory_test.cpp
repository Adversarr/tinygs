#include <gtest/gtest.h>
#include <array>
#include <tinygs/initialization/initialization.hpp>
#include <tinygs/initialization/knn.hpp>
#include <tinygs/initialization/random.hpp>

namespace tinygs {

class InitializationFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(InitializationFactoryTest, CreateKnn) {
    auto init = create_initialization("knn");
    ASSERT_NE(init, nullptr);
    EXPECT_NE(dynamic_cast<KnnInitialization*>(init.get()), nullptr);
}

TEST_F(InitializationFactoryTest, CreateRandom) {
    auto init = create_initialization("random");
    ASSERT_NE(init, nullptr);
    EXPECT_NE(dynamic_cast<RandomInitialization*>(init.get()), nullptr);
}

TEST_F(InitializationFactoryTest, UnknownTypeThrows) {
    EXPECT_THROW(create_initialization("unknown"), std::runtime_error);
}

TEST_F(InitializationFactoryTest, InvalidTypeVariantsThrow) {
    constexpr std::array<const char*, 10> invalid_types = {
        "",
        "invalid_init",
        "kn",
        "randm",
        " knn",
        "knn ",
        "knn1",
        "random2",
        "knn_init",
        "uniform"
    };

    for (const char* type : invalid_types) {
        EXPECT_THROW(create_initialization(type), std::runtime_error)
            << "Expected throw for type: " << type;
    }
}

TEST_F(InitializationFactoryTest, ErrorMessageContainsType) {
    try {
        create_initialization("unknown_type");
        FAIL() << "Expected runtime_error";
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("unknown_type") != std::string::npos);
    }
}

TEST_F(InitializationFactoryTest, ErrorMessageContainsUnknownWord) {
    try {
        create_initialization("test_init");
        FAIL() << "Expected runtime_error";
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("Unknown") != std::string::npos || 
                    msg.find("unknown") != std::string::npos);
    }
}

TEST_F(InitializationFactoryTest, KnnHasValidGaussians) {
    auto init = create_initialization("knn");
    const Gaussian3d& gaussians = init->gaussians();
    EXPECT_TRUE(gaussians.means.empty());
}

TEST_F(InitializationFactoryTest, RandomHasValidGaussians) {
    auto init = create_initialization("random");
    const Gaussian3d& gaussians = init->gaussians();
    EXPECT_TRUE(gaussians.means.empty());
}

TEST_F(InitializationFactoryTest, KnnSupportsGetParams) {
    auto init = create_initialization("knn");
    json params = init->get_params();
    EXPECT_TRUE(params.is_object());
}

TEST_F(InitializationFactoryTest, RandomSupportsGetParams) {
    auto init = create_initialization("random");
    json params = init->get_params();
    EXPECT_TRUE(params.is_object());
}

}
