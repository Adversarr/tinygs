#include <gtest/gtest.h>
#include <tinygs/common.hpp>
#include <unordered_set>

namespace {

using namespace tinygs;

TEST(TileUtilsTest, GetTileX) {
    EXPECT_EQ(get_tile_x(0), 0);
    EXPECT_EQ(get_tile_x(1), 0);
    EXPECT_EQ(get_tile_x(7), 0);
    EXPECT_EQ(get_tile_x(8), 1);
    EXPECT_EQ(get_tile_x(15), 1);
    EXPECT_EQ(get_tile_x(16), 2);
    EXPECT_EQ(get_tile_x(23), 2);
    EXPECT_EQ(get_tile_x(24), 3);
}

TEST(TileUtilsTest, GetTileY) {
    EXPECT_EQ(get_tile_y(0), 0);
    EXPECT_EQ(get_tile_y(1), 0);
    EXPECT_EQ(get_tile_y(7), 0);
    EXPECT_EQ(get_tile_y(8), 1);
    EXPECT_EQ(get_tile_y(15), 1);
    EXPECT_EQ(get_tile_y(16), 2);
    EXPECT_EQ(get_tile_y(31), 3);
    EXPECT_EQ(get_tile_y(32), 4);
}

TEST(TileUtilsTest, GetIntraX) {
    for (uint32_t i = 0; i < 8; ++i) {
        EXPECT_EQ(get_intra_x(i), i);
    }
    
    for (uint32_t i = 8; i < 16; ++i) {
        EXPECT_EQ(get_intra_x(i), i - 8);
    }
    
    EXPECT_EQ(get_intra_x(16), 0);
    EXPECT_EQ(get_intra_x(17), 1);
    EXPECT_EQ(get_intra_x(23), 7);
}

TEST(TileUtilsTest, GetIntraY) {
    for (uint32_t i = 0; i < 8; ++i) {
        EXPECT_EQ(get_intra_y(i), i);
    }
    
    for (uint32_t i = 8; i < 16; ++i) {
        EXPECT_EQ(get_intra_y(i), i - 8);
    }
    
    EXPECT_EQ(get_intra_y(24), 0);
    EXPECT_EQ(get_intra_y(25), 1);
    EXPECT_EQ(get_intra_y(31), 7);
}

TEST(TileUtilsTest, GetTileIndexTiled) {
    uint32_t tiled_width = 4;
    
    EXPECT_EQ(get_tile_index_tiled(0, 0, tiled_width), 0);
    EXPECT_EQ(get_tile_index_tiled(0, 8, tiled_width), 1);
    EXPECT_EQ(get_tile_index_tiled(8, 0, tiled_width), 4);
    EXPECT_EQ(get_tile_index_tiled(8, 8, tiled_width), 5);
    EXPECT_EQ(get_tile_index_tiled(16, 16, tiled_width), 10);
}

TEST(TileUtilsTest, GetTileIndex) {
    uint32_t width = 32;
    
    EXPECT_EQ(get_tile_index(0, 0, width), 0);
    EXPECT_EQ(get_tile_index(0, 8, width), 1);
    EXPECT_EQ(get_tile_index(0, 16, width), 2);
    EXPECT_EQ(get_tile_index(8, 0, width), 4);
    EXPECT_EQ(get_tile_index(8, 8, width), 5);
}

TEST(TileUtilsTest, GetOffsetInTile) {
    for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
            uint32_t expected = i * 8 + j;
            EXPECT_EQ(get_offset_in_tile(i, j), expected);
        }
    }
}

TEST(TileUtilsTest, GetLinearIndexTiledRoundTrip) {
    uint32_t tiled_width = 4;
    
    for (uint32_t i = 0; i < 32; ++i) {
        for (uint32_t j = 0; j < 32; ++j) {
            uint32_t tile_idx = get_tile_index_tiled(i, j, tiled_width);
            uint32_t offset = get_offset_in_tile(i, j);
            uint32_t linear = (tile_idx << (2 * kImageTileLog2)) + offset;
            EXPECT_EQ(get_linear_index_tiled(i, j, tiled_width), linear);
        }
    }
}

TEST(TileUtilsTest, GetLinearIndexUniqueness) {
    uint32_t width = 32;
    std::unordered_set<uint32_t> indices;
    
    for (uint32_t i = 0; i < 32; ++i) {
        for (uint32_t j = 0; j < 32; ++j) {
            uint32_t idx = get_linear_index(i, j, width);
            EXPECT_TRUE(indices.find(idx) == indices.end());
            indices.insert(idx);
        }
    }
    
    EXPECT_EQ(indices.size(), 32u * 32u);
}

TEST(TileUtilsTest, TileConstantsCorrect) {
    EXPECT_EQ(kImageTile, 8u);
    EXPECT_EQ(kImageTileLog2, 3u);
    EXPECT_EQ(kImageTileMask, 7u);
}

TEST(TileUtilsTest, GetLinearIndexTiledConsistency) {
    uint32_t width = 24;
    uint32_t tiled_width = width >> kImageTileLog2;
    
    for (uint32_t i = 0; i < 24; ++i) {
        for (uint32_t j = 0; j < 24; ++j) {
            uint32_t idx1 = get_linear_index(i, j, width);
            uint32_t idx2 = get_linear_index_tiled(i, j, tiled_width);
            EXPECT_EQ(idx1, idx2);
        }
    }
}

TEST(TileUtilsTest, TileBoundaryBehavior) {
    EXPECT_EQ(get_tile_x(7), 0);
    EXPECT_EQ(get_tile_x(8), 1);
    EXPECT_EQ(get_tile_y(7), 0);
    EXPECT_EQ(get_tile_y(8), 1);
    
    EXPECT_EQ(get_intra_x(7), 7);
    EXPECT_EQ(get_intra_x(8), 0);
    EXPECT_EQ(get_intra_y(7), 7);
    EXPECT_EQ(get_intra_y(8), 0);
}

TEST(TileUtilsTest, LargeCoordinates) {
    uint32_t width = 1920;
    uint32_t tiled_width = width >> kImageTileLog2;
    
    uint32_t idx1 = get_linear_index(0, 0, width);
    uint32_t idx2 = get_linear_index(1079, 1919, width);
    
    EXPECT_LT(idx1, idx2);
    
    uint32_t idx_corner = get_linear_index_tiled(1079, 1919, tiled_width);
    EXPECT_GT(idx_corner, 0u);
}

TEST(TileUtilsTest, TileIndexSequential) {
    uint32_t width = 64;
    uint32_t prev_idx = 0;
    
    for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
            uint32_t idx = get_linear_index(i, j, width);
            if (i > 0 || j > 0) {
                EXPECT_GT(idx, prev_idx);
            }
            prev_idx = idx;
        }
    }
}

}
