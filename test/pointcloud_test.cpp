#include <gtest/gtest.h>
#include <tinygs/core/pointcloud.hpp>
#include <filesystem>
#include <fstream>

namespace {

using namespace tinygs;

class PointCloudTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir = std::filesystem::temp_directory_path() / "tinygs_pointcloud_test";
        std::filesystem::create_directories(test_dir);
    }

    void TearDown() override {
        std::filesystem::remove_all(test_dir);
    }

    std::filesystem::path test_dir;
};

TEST_F(PointCloudTest, LoadFromColmapBasicFormat) {
    auto colmap_file = test_dir / "points3D.txt";
    
    std::ofstream out(colmap_file);
    out << "# 3D point list with one line of data per point:\n";
    out << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n";
    out << "1 0.0 1.0 2.0 128 64 192 1.5\n";
    out << "2 3.0 4.0 5.0 255 0 127 2.0\n";
    out << "3 -1.0 -2.0 -3.0 0 255 128 0.5\n";
    out.close();

    PointCloud pc = load_from_colmap(colmap_file.string());
    
    EXPECT_EQ(pc.points.size(), 3U);
    EXPECT_EQ(pc.colors.size(), 3U);
    
    EXPECT_FLOAT_EQ(pc.points[0].x, 0.0f);
    EXPECT_FLOAT_EQ(pc.points[0].y, 1.0f);
    EXPECT_FLOAT_EQ(pc.points[0].z, 2.0f);
    
    EXPECT_FLOAT_EQ(pc.points[1].x, 3.0f);
    EXPECT_FLOAT_EQ(pc.points[1].y, 4.0f);
    EXPECT_FLOAT_EQ(pc.points[1].z, 5.0f);
    
    EXPECT_FLOAT_EQ(pc.colors[0].x, 128.0f / 255.0f);
    EXPECT_FLOAT_EQ(pc.colors[0].y, 64.0f / 255.0f);
    EXPECT_FLOAT_EQ(pc.colors[0].z, 192.0f / 255.0f);
    
    EXPECT_FLOAT_EQ(pc.colors[1].x, 255.0f / 255.0f);
    EXPECT_FLOAT_EQ(pc.colors[1].y, 0.0f / 255.0f);
    EXPECT_FLOAT_EQ(pc.colors[1].z, 127.0f / 255.0f);
}

TEST_F(PointCloudTest, LoadFromColmapSkipsCommentsAndEmptyLines) {
    auto colmap_file = test_dir / "points3D.txt";
    
    std::ofstream out(colmap_file);
    out << "# Header comment\n";
    out << "\n";
    out << "1 0.0 0.0 0.0 100 100 100 1.0\n";
    out << "\n";
    out << "# Another comment\n";
    out << "2 1.0 1.0 1.0 200 200 200 2.0\n";
    out << "\n";
    out.close();

    PointCloud pc = load_from_colmap(colmap_file.string());
    
    EXPECT_EQ(pc.points.size(), 2U);
    EXPECT_EQ(pc.colors.size(), 2U);
}

TEST_F(PointCloudTest, LoadFromColmapEmptyFile) {
    auto colmap_file = test_dir / "empty.txt";
    
    std::ofstream out(colmap_file);
    out << "# Only comments\n";
    out.close();

    PointCloud pc = load_from_colmap(colmap_file.string());
    
    EXPECT_EQ(pc.points.size(), 0U);
    EXPECT_EQ(pc.colors.size(), 0U);
}

TEST_F(PointCloudTest, LoadFromColmapNonexistentFile) {
    PointCloud pc = load_from_colmap("/nonexistent/path/points3D.txt");
    
    EXPECT_EQ(pc.points.size(), 0U);
    EXPECT_EQ(pc.colors.size(), 0U);
}

TEST_F(PointCloudTest, LoadPlyBasicFormat) {
    auto ply_file = test_dir / "test.ply";
    
    std::ofstream out(ply_file, std::ios::binary);
    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex 2\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "end_header\n";
    out << "0.0 1.0 2.0 128 64 192\n";
    out << "3.0 4.0 5.0 255 0 127\n";
    out.close();

    PointCloud pc = load_ply(ply_file.string());
    
    EXPECT_EQ(pc.points.size(), 2U);
    EXPECT_EQ(pc.colors.size(), 2U);
    
    EXPECT_FLOAT_EQ(pc.points[0].x, 0.0f);
    EXPECT_FLOAT_EQ(pc.points[0].y, 1.0f);
    EXPECT_FLOAT_EQ(pc.points[0].z, 2.0f);
    
    EXPECT_FLOAT_EQ(pc.colors[0].x, 128.0f / 255.0f);
    EXPECT_FLOAT_EQ(pc.colors[0].y, 64.0f / 255.0f);
    EXPECT_FLOAT_EQ(pc.colors[0].z, 192.0f / 255.0f);
}

TEST_F(PointCloudTest, LoadPointCloudDetectsPlyFormat) {
    auto ply_file = test_dir / "test.ply";
    
    std::ofstream out(ply_file, std::ios::binary);
    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex 1\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "end_header\n";
    out << "0.0 0.0 0.0 128 128 128\n";
    out.close();

    PointCloud pc = load_point_cloud(ply_file.string());
    
    EXPECT_EQ(pc.points.size(), 1U);
    EXPECT_EQ(pc.colors.size(), 1U);
}

TEST_F(PointCloudTest, LoadPointCloudDetectsColmapFormat) {
    auto colmap_file = test_dir / "points.txt";
    
    std::ofstream out(colmap_file);
    out << "1 0.0 0.0 0.0 128 128 128 1.0\n";
    out.close();

    PointCloud pc = load_point_cloud(colmap_file.string());
    
    EXPECT_EQ(pc.points.size(), 1U);
    EXPECT_EQ(pc.colors.size(), 1U);
}

TEST_F(PointCloudTest, LoadPointCloudCaseInsensitiveExtension) {
    auto ply_upper = test_dir / "TEST.PLY";
    
    std::ofstream out(ply_upper, std::ios::binary);
    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex 1\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "end_header\n";
    out << "1.0 2.0 3.0 100 100 100\n";
    out.close();

    PointCloud pc = load_point_cloud(ply_upper.string());
    
    EXPECT_EQ(pc.points.size(), 1U);
}

TEST_F(PointCloudTest, LoadPointCloudUnsupportedFormat) {
    auto unsupported = test_dir / "test.xyz";
    
    std::ofstream out(unsupported);
    out << "some data\n";
    out.close();

    PointCloud pc = load_point_cloud(unsupported.string());
    
    EXPECT_EQ(pc.points.size(), 0U);
    EXPECT_EQ(pc.colors.size(), 0U);
}

TEST_F(PointCloudTest, LoadPointCloudFilenameTooShort) {
    PointCloud pc = load_point_cloud("abc");
    
    EXPECT_EQ(pc.points.size(), 0U);
    EXPECT_EQ(pc.colors.size(), 0U);
}

TEST_F(PointCloudTest, SavePlyPartialFeatures) {
    Gaussian3d gs;
    gs.means = {{0.0f, 1.0f, 2.0f}, {3.0f, 4.0f, 5.0f}};
    gs.sh0 = {{0.5f, 0.6f, 0.7f}, {0.8f, 0.9f, 1.0f}};
    gs.opacities = {0.5f, 0.75f};
    gs.scales = {{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}};
    gs.rotations = {{1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}};
    
    auto ply_file = test_dir / "output.ply";
    
    EXPECT_NO_THROW(save_ply(ply_file.string(), gs, false));
    
    EXPECT_TRUE(std::filesystem::exists(ply_file));
    EXPECT_GT(std::filesystem::file_size(ply_file), 0U);
}

TEST_F(PointCloudTest, SavePlyFullFeatures) {
    Gaussian3d gs;
    gs.means = {{0.0f, 1.0f, 2.0f}};
    gs.sh0 = {{0.5f, 0.6f, 0.7f}};
    gs.opacities = {0.5f};
    gs.scales = {{0.1f, 0.2f, 0.3f}};
    gs.rotations = {{1.0f, 0.0f, 0.0f, 0.0f}};
    gs.sh1 = {{0.1f, 0.2f, 0.3f}};
    gs.sh2 = {{0.4f, 0.5f, 0.6f}};
    gs.sh3 = {{0.7f, 0.8f, 0.9f}};
    
    auto ply_file = test_dir / "output_full.ply";
    
    EXPECT_NO_THROW(save_ply(ply_file.string(), gs, true));
    
    EXPECT_TRUE(std::filesystem::exists(ply_file));
    EXPECT_GT(std::filesystem::file_size(ply_file), 0U);
}

TEST_F(PointCloudTest, SaveAndLoadRoundTrip) {
    Gaussian3d gs_original;
    gs_original.means = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    gs_original.sh0 = {{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}};
    gs_original.opacities = {0.8f, 0.9f};
    gs_original.scales = {{0.01f, 0.02f, 0.03f}, {0.04f, 0.05f, 0.06f}};
    gs_original.rotations = {{1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}};
    
    auto ply_file = test_dir / "roundtrip.ply";
    
    save_ply(ply_file.string(), gs_original, false);
    
    PointCloud pc_loaded = load_ply(ply_file.string());
    
    EXPECT_EQ(pc_loaded.points.size(), 2U);
    EXPECT_EQ(pc_loaded.colors.size(), 2U);
    
    for (size_t i = 0; i < gs_original.means.size(); ++i) {
        EXPECT_FLOAT_EQ(pc_loaded.points[i].x, gs_original.means[i].x);
        EXPECT_FLOAT_EQ(pc_loaded.points[i].y, gs_original.means[i].y);
        EXPECT_FLOAT_EQ(pc_loaded.points[i].z, gs_original.means[i].z);
    }
}

}
