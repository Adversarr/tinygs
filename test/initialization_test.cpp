#include <gtest/gtest.h>
#include <tinygs/initialization/initialization.hpp>
#include <tinygs/initialization/random.hpp>
#include <nlohmann/json.hpp>

namespace {

using namespace tinygs;

TEST(RandomParametersTest, DefaultValues) {
    RandomParameters params;
    
    EXPECT_EQ(params.num_points, 100000);
    EXPECT_FLOAT_EQ(params.extent, 6.0f);
    EXPECT_FLOAT_EQ(params.init_scaling, 0.01f);
    EXPECT_FLOAT_EQ(params.init_opacity, 0.5f);
    EXPECT_EQ(params.sh_degree, 3);
    EXPECT_FLOAT_EQ(params.min_scale, 1e-7f);
    EXPECT_FLOAT_EQ(params.max_scale, 1.0f);
    EXPECT_TRUE(params.use_uniform_scale);
    EXPECT_EQ(params.seed, 42U);
}

TEST(RandomInitializationTest, InitializeCreatesCorrectNumberOfGaussians) {
    RandomParameters params;
    params.num_points = 100;
    params.seed = 12345;
    
    RandomInitialization init(params);
    PointCloud empty_pc;
    init.initialize(empty_pc);
    
    const auto& gaussians = init.gaussians();
    EXPECT_EQ(gaussians.means.size(), 100U);
    EXPECT_EQ(gaussians.opacities.size(), 100U);
    EXPECT_EQ(gaussians.rotations.size(), 100U);
    EXPECT_EQ(gaussians.scales.size(), 100U);
    EXPECT_EQ(gaussians.sh0.size(), 100U);
}

TEST(RandomInitializationTest, InitializeSetsCorrectOpacity) {
    RandomParameters params;
    params.num_points = 50;
    params.init_opacity = 0.75f;
    
    RandomInitialization init(params);
    PointCloud empty_pc;
    init.initialize(empty_pc);
    
    const auto& gaussians = init.gaussians();
    for (const auto& opacity : gaussians.opacities) {
        EXPECT_FLOAT_EQ(opacity, 0.75f);
    }
}

TEST(RandomInitializationTest, InitializeSetsIdentityRotation) {
    RandomParameters params;
    params.num_points = 50;
    
    RandomInitialization init(params);
    PointCloud empty_pc;
    init.initialize(empty_pc);
    
    const auto& gaussians = init.gaussians();
    for (const auto& rot : gaussians.rotations) {
        EXPECT_FLOAT_EQ(rot.x, 1.0f);
        EXPECT_FLOAT_EQ(rot.y, 0.0f);
        EXPECT_FLOAT_EQ(rot.z, 0.0f);
        EXPECT_FLOAT_EQ(rot.w, 0.0f);
    }
}

TEST(RandomInitializationTest, InitializeUniformScale) {
    RandomParameters params;
    params.num_points = 50;
    params.use_uniform_scale = true;
    params.init_scaling = 0.1f;
    
    RandomInitialization init(params);
    PointCloud empty_pc;
    init.initialize(empty_pc);
    
    const auto& gaussians = init.gaussians();
    float expected_log_scale = std::log(0.1f);
    
    for (const auto& scale : gaussians.scales) {
        EXPECT_NEAR(scale.x, expected_log_scale, 1e-6f);
        EXPECT_NEAR(scale.y, expected_log_scale, 1e-6f);
        EXPECT_NEAR(scale.z, expected_log_scale, 1e-6f);
    }
}

TEST(RandomInitializationTest, InitializeRandomScale) {
    RandomParameters params;
    params.num_points = 50;
    params.use_uniform_scale = false;
    params.min_scale = 0.001f;
    params.max_scale = 0.1f;
    
    RandomInitialization init(params);
    PointCloud empty_pc;
    init.initialize(empty_pc);
    
    const auto& gaussians = init.gaussians();
    bool has_variation = false;
    
    for (size_t i = 1; i < gaussians.scales.size(); ++i) {
        if (gaussians.scales[i].x != gaussians.scales[0].x) {
            has_variation = true;
            break;
        }
    }
    EXPECT_TRUE(has_variation);
}

TEST(RandomInitializationTest, SetParamsFromJson) {
    RandomInitialization init;
    
    json params = R"({
        "num_points": 500,
        "extent": 10.0,
        "init_scaling": 0.05,
        "init_opacity": 0.8,
        "sh_degree": 2,
        "min_scale": 0.0001,
        "max_scale": 0.5,
        "use_uniform_scale": false,
        "seed": 999
    })"_json;
    
    init.set_params(params);
    
    const auto& rp = init.get_random_parameters();
    EXPECT_EQ(rp.num_points, 500);
    EXPECT_FLOAT_EQ(rp.extent, 10.0f);
    EXPECT_FLOAT_EQ(rp.init_scaling, 0.05f);
    EXPECT_FLOAT_EQ(rp.init_opacity, 0.8f);
    EXPECT_EQ(rp.sh_degree, 2);
    EXPECT_FLOAT_EQ(rp.min_scale, 0.0001f);
    EXPECT_FLOAT_EQ(rp.max_scale, 0.5f);
    EXPECT_FALSE(rp.use_uniform_scale);
    EXPECT_EQ(rp.seed, 999U);
}

TEST(RandomInitializationTest, GetParamsReturnsJson) {
    RandomParameters params;
    params.num_points = 200;
    params.extent = 5.0f;
    params.seed = 42;
    
    RandomInitialization init(params);
    json j = init.get_params();
    
    EXPECT_EQ(j["type"], "random");
    EXPECT_EQ(j["num_points"], 200);
    EXPECT_FLOAT_EQ(j["extent"].get<float>(), 5.0f);
    EXPECT_EQ(j["seed"], 42U);
}

TEST(RandomInitializationTest, RgbToShConversion) {
    RandomInitialization init;
    
    vec3 white(1.0f, 1.0f, 1.0f);
    vec3 black(0.0f, 0.0f, 0.0f);
    vec3 gray(0.5f, 0.5f, 0.5f);
    
    constexpr float kInvSH = 0.28209479177387814f;
    
    vec3 sh_white = (white - vec3(0.5f)) / kInvSH;
    vec3 sh_black = (black - vec3(0.5f)) / kInvSH;
    vec3 sh_gray = (gray - vec3(0.5f)) / kInvSH;
    
    EXPECT_GT(sh_white.x, 0.0f);
    EXPECT_LT(sh_black.x, 0.0f);
    EXPECT_NEAR(sh_gray.x, 0.0f, 1e-6f);
}

TEST(CreateInitializationTest, CreatesRandomInitialization) {
    auto init = create_initialization("random");
    EXPECT_NE(init, nullptr);
    
    json params = init->get_params();
    EXPECT_EQ(params["type"], "random");
}

TEST(CreateInitializationTest, CreatesKnnInitialization) {
    auto init = create_initialization("knn");
    EXPECT_NE(init, nullptr);
}

TEST(CreateInitializationTest, CreatesCaseInsensitive) {
    auto init1 = create_initialization("RANDOM");
    auto init2 = create_initialization("Random");
    auto init3 = create_initialization("KNN");
    auto init4 = create_initialization("Knn");
    
    EXPECT_NE(init1, nullptr);
    EXPECT_NE(init2, nullptr);
    EXPECT_NE(init3, nullptr);
    EXPECT_NE(init4, nullptr);
}

TEST(CreateInitializationTest, ThrowsOnUnknownType) {
    EXPECT_THROW(create_initialization("unknown"), std::runtime_error);
    EXPECT_THROW(create_initialization("invalid"), std::runtime_error);
}

TEST(InitializationBaseTest, GaussiansReturnsReference) {
    RandomParameters params;
    params.num_points = 10;
    
    RandomInitialization init(params);
    PointCloud empty_pc;
    init.initialize(empty_pc);
    
    const Gaussian3d& gaussians = init.gaussians();
    EXPECT_EQ(gaussians.means.size(), 10U);
}

}
