/**
 * @file test_main.cpp
 * @brief Main test runner for tinygs tests
 */

#include <gtest/gtest.h>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}