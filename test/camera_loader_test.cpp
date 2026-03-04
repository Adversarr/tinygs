#include <gtest/gtest.h>
#include <tinygs/core/camera_loader.hpp>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace {

using namespace tinygs;

class CameraLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir = std::filesystem::temp_directory_path() / "tinygs_camera_loader_test";
        std::filesystem::create_directories(test_dir);
    }

    void TearDown() override {
        std::filesystem::remove_all(test_dir);
    }

    std::filesystem::path test_dir;
};

TEST_F(CameraLoaderTest, LoadCameraExtrinsicsBasic) {
    auto ext_file = test_dir / "extrinsics.txt";
    
    std::ofstream out(ext_file);
    out << "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 100\n";
    out << "2 1.0 0.0 0.0 0.0 1.0 0.0 0.0 1 200\n";
    out.close();

    SingleCameraLoader loader;
    loader.load_camera_extrinsics(ext_file.string());
    
    const auto& extrinsics = loader.get_camera_extrinsics();
    ASSERT_EQ(extrinsics.size(), 2U);
    
    EXPECT_EQ(extrinsics[0].frame_idx, 1U);
    EXPECT_EQ(extrinsics[1].frame_idx, 2U);
}

TEST_F(CameraLoaderTest, LoadCameraExtrinsicsSortsByFrameIdx) {
    auto ext_file = test_dir / "extrinsics.txt";
    
    std::ofstream out(ext_file);
    out << "3 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 300\n";
    out << "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 100\n";
    out << "2 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 200\n";
    out.close();

    SingleCameraLoader loader;
    loader.load_camera_extrinsics(ext_file.string());
    
    const auto& extrinsics = loader.get_camera_extrinsics();
    ASSERT_EQ(extrinsics.size(), 3U);
    
    EXPECT_EQ(extrinsics[0].frame_idx, 1U);
    EXPECT_EQ(extrinsics[1].frame_idx, 2U);
    EXPECT_EQ(extrinsics[2].frame_idx, 3U);
}

TEST_F(CameraLoaderTest, LoadCameraIntrinsicsBasic) {
    auto int_file = test_dir / "intrinsics.txt";
    
    std::ofstream out(int_file);
    out << "1 PINHOLE 1920 1080 1000.0 1000.0 960.0 540.0\n";
    out << "2 PINHOLE 1280 720 800.0 800.0 640.0 360.0\n";
    out.close();

    SingleCameraLoader loader;
    loader.load_camera_intrinsics(int_file.string());
    
    const auto& intrinsics = loader.get_camera_intrinsics();
    ASSERT_EQ(intrinsics.size(), 2U);
    
    EXPECT_EQ(intrinsics[0].uid, 1U);
    EXPECT_EQ(intrinsics[0].width, 1920);
    EXPECT_EQ(intrinsics[0].height, 1080);
    EXPECT_FLOAT_EQ(intrinsics[0].fx, 1000.0f);
    EXPECT_FLOAT_EQ(intrinsics[0].fy, 1000.0f);
    EXPECT_FLOAT_EQ(intrinsics[0].cx, 960.0f);
    EXPECT_FLOAT_EQ(intrinsics[0].cy, 540.0f);
    
    EXPECT_EQ(intrinsics[1].uid, 2U);
    EXPECT_EQ(intrinsics[1].width, 1280);
    EXPECT_EQ(intrinsics[1].height, 720);
}

TEST_F(CameraLoaderTest, LoadCameraIntrinsicsWithDistortion) {
    auto int_file = test_dir / "intrinsics.txt";
    
    std::ofstream out(int_file);
    out << "1 PINHOLE 1920 1080 1000.0 1000.0 960.0 540.0 0.1 0.2 0.3 0.01 0.02\n";
    out.close();

    SingleCameraLoader loader;
    loader.load_camera_intrinsics(int_file.string());
    
    const auto& intrinsics = loader.get_camera_intrinsics();
    ASSERT_EQ(intrinsics.size(), 1U);
    
    EXPECT_FLOAT_EQ(intrinsics[0].k1, 0.1f);
    EXPECT_FLOAT_EQ(intrinsics[0].k2, 0.2f);
    EXPECT_FLOAT_EQ(intrinsics[0].k3, 0.3f);
    EXPECT_FLOAT_EQ(intrinsics[0].p1, 0.01f);
    EXPECT_FLOAT_EQ(intrinsics[0].p2, 0.02f);
}

TEST_F(CameraLoaderTest, LoadCameraIntrinsicsSkipsComments) {
    auto int_file = test_dir / "intrinsics.txt";
    
    std::ofstream out(int_file);
    out << "# Camera intrinsics file\n";
    out << "1 PINHOLE 1920 1080 1000.0 1000.0 960.0 540.0\n";
    out << "# Another comment\n";
    out << "2 PINHOLE 1280 720 800.0 800.0 640.0 360.0\n";
    out.close();

    SingleCameraLoader loader;
    loader.load_camera_intrinsics(int_file.string());
    
    const auto& intrinsics = loader.get_camera_intrinsics();
    EXPECT_EQ(intrinsics.size(), 2U);
}

TEST_F(CameraLoaderTest, LoadCameraIntrinsicsEmptyFileThrows) {
    auto int_file = test_dir / "empty.txt";
    
    std::ofstream out(int_file);
    out << "";
    out.close();

    SingleCameraLoader loader;
    EXPECT_THROW(loader.load_camera_intrinsics(int_file.string()), std::runtime_error);
}

TEST_F(CameraLoaderTest, ConstructorLoadsBothFiles) {
    auto ext_file = test_dir / "extrinsics.txt";
    auto int_file = test_dir / "intrinsics.txt";
    
    std::ofstream ext_out(ext_file);
    ext_out << "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 100\n";
    ext_out.close();
    
    std::ofstream int_out(int_file);
    int_out << "1 PINHOLE 1920 1080 1000.0 1000.0 960.0 540.0\n";
    int_out.close();

    SingleCameraLoader loader(ext_file.string(), int_file.string());
    
    EXPECT_EQ(loader.get_camera_extrinsics().size(), 1U);
    EXPECT_EQ(loader.get_camera_intrinsics().size(), 1U);
}

TEST_F(CameraLoaderTest, ResizeSensorSameSize) {
    auto int_file = test_dir / "intrinsics.txt";
    
    std::ofstream out(int_file);
    out << "1 PINHOLE 1920 1080 1000.0 1000.0 960.0 540.0\n";
    out.close();

    SingleCameraLoader loader;
    loader.load_camera_intrinsics(int_file.string());
    
    loader.resize_sensor(1920, 1080);
    
    const auto& intrinsics = loader.get_camera_intrinsics();
    EXPECT_EQ(intrinsics[0].width, 1920);
    EXPECT_EQ(intrinsics[0].height, 1080);
    EXPECT_FLOAT_EQ(intrinsics[0].fx, 1000.0f);
}

TEST_F(CameraLoaderTest, ResizeSensorSmallerSize) {
    auto int_file = test_dir / "intrinsics.txt";
    
    std::ofstream out(int_file);
    out << "1 PINHOLE 1920 1080 1000.0 1000.0 960.0 540.0\n";
    out.close();

    SingleCameraLoader loader;
    loader.load_camera_intrinsics(int_file.string());
    
    loader.resize_sensor(960, 540);
    
    const auto& intrinsics = loader.get_camera_intrinsics();
    EXPECT_EQ(intrinsics[0].width, 960);
    EXPECT_EQ(intrinsics[0].height, 540);
    EXPECT_FLOAT_EQ(intrinsics[0].fx, 500.0f);
    EXPECT_FLOAT_EQ(intrinsics[0].fy, 500.0f);
    EXPECT_FLOAT_EQ(intrinsics[0].cx, 480.0f);
    EXPECT_FLOAT_EQ(intrinsics[0].cy, 270.0f);
}

TEST_F(CameraLoaderTest, ResizeSensorLargerSize) {
    auto int_file = test_dir / "intrinsics.txt";
    
    std::ofstream out(int_file);
    out << "1 PINHOLE 1920 1080 1000.0 1000.0 960.0 540.0\n";
    out.close();

    SingleCameraLoader loader;
    loader.load_camera_intrinsics(int_file.string());
    
    loader.resize_sensor(3840, 2160);
    
    const auto& intrinsics = loader.get_camera_intrinsics();
    EXPECT_EQ(intrinsics[0].width, 3840);
    EXPECT_EQ(intrinsics[0].height, 2160);
    EXPECT_FLOAT_EQ(intrinsics[0].fx, 2000.0f);
    EXPECT_FLOAT_EQ(intrinsics[0].fy, 2000.0f);
    EXPECT_FLOAT_EQ(intrinsics[0].cx, 1920.0f);
    EXPECT_FLOAT_EQ(intrinsics[0].cy, 1080.0f);
}

TEST_F(CameraLoaderTest, ResizeSensorZeroSizeThrows) {
    auto int_file = test_dir / "intrinsics.txt";
    
    std::ofstream out(int_file);
    out << "1 PINHOLE 1920 1080 1000.0 1000.0 960.0 540.0\n";
    out.close();

    SingleCameraLoader loader;
    loader.load_camera_intrinsics(int_file.string());
    
    EXPECT_THROW(loader.resize_sensor(0, 540), std::runtime_error);
    EXPECT_THROW(loader.resize_sensor(960, 0), std::runtime_error);
}

TEST_F(CameraLoaderTest, LoadFromJsonBasic) {
    auto cameras_file = test_dir / "cameras.json";
    auto poses_file = test_dir / "poses.json";
    
    nlohmann::json cameras = nlohmann::json::array({
        {
            {"camera_id", 1},
            {"model", "PINHOLE"},
            {"width", 1920},
            {"height", 1080},
            {"params", {1000.0, 1000.0, 960.0, 540.0}}
        }
    });
    
    std::ofstream cam_out(cameras_file);
    cam_out << cameras.dump();
    cam_out.close();
    
    nlohmann::json poses = nlohmann::json::array({
        {
            {"image_id", 1},
            {"camera_id", 1},
            {"qvec", {1.0, 0.0, 0.0, 0.0}},
            {"tvec", {0.0, 0.0, 0.0}},
            {"name", "000001.jpg"}
        }
    });
    
    std::ofstream pose_out(poses_file);
    pose_out << poses.dump();
    pose_out.close();

    SingleCameraLoader loader;
    loader.load_from_json(cameras_file.string(), poses_file.string());
    
    EXPECT_EQ(loader.get_camera_intrinsics().size(), 1U);
    EXPECT_EQ(loader.get_camera_extrinsics().size(), 1U);
}

TEST_F(CameraLoaderTest, LoadFromJsonSimplePinhole) {
    auto cameras_file = test_dir / "cameras.json";
    auto poses_file = test_dir / "poses.json";
    
    nlohmann::json cameras = nlohmann::json::array({
        {
            {"camera_id", 1},
            {"model", "SIMPLE_PINHOLE"},
            {"width", 1920},
            {"height", 1080},
            {"params", {1000.0, 960.0, 540.0}}
        }
    });
    
    std::ofstream cam_out(cameras_file);
    cam_out << cameras.dump();
    cam_out.close();
    
    nlohmann::json poses = nlohmann::json::array({
        {
            {"image_id", 1},
            {"camera_id", 1},
            {"qvec", {1.0, 0.0, 0.0, 0.0}},
            {"tvec", {0.0, 0.0, 0.0}},
            {"name", "000001.jpg"}
        }
    });
    
    std::ofstream pose_out(poses_file);
    pose_out << poses.dump();
    pose_out.close();

    SingleCameraLoader loader;
    loader.load_from_json(cameras_file.string(), poses_file.string());
    
    const auto& intrinsics = loader.get_camera_intrinsics();
    ASSERT_EQ(intrinsics.size(), 1U);
    EXPECT_FLOAT_EQ(intrinsics[0].fx, 1000.0f);
    EXPECT_FLOAT_EQ(intrinsics[0].fy, 1000.0f);
    EXPECT_FLOAT_EQ(intrinsics[0].cx, 960.0f);
    EXPECT_FLOAT_EQ(intrinsics[0].cy, 540.0f);
}

TEST_F(CameraLoaderTest, SetCameraIntrinsics) {
    SingleCameraLoader loader;
    
    std::vector<CameraIntrinsics> intrinsics(2);
    intrinsics[0].uid = 1;
    intrinsics[0].width = 1920;
    intrinsics[0].height = 1080;
    intrinsics[1].uid = 2;
    intrinsics[1].width = 1280;
    intrinsics[1].height = 720;
    
    loader.set_camera_intrinsics(intrinsics);
    
    const auto& loaded = loader.get_camera_intrinsics();
    EXPECT_EQ(loaded.size(), 2U);
    EXPECT_EQ(loaded[0].uid, 1U);
    EXPECT_EQ(loaded[1].uid, 2U);
}

}
