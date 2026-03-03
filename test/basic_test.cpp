#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

#include "tinygs/utils/file.hpp"

namespace {

TEST(FileUtilsTest, ReadlinesTrimsWhitespaceAndSkipsEmptyLines) {
  const std::filesystem::path base =
      std::filesystem::temp_directory_path() / "tinygs_gtest_readlines";
  std::filesystem::create_directories(base);
  const std::filesystem::path file_path = base / "input.txt";

  {
    std::ofstream out(file_path);
    ASSERT_TRUE(out.is_open());
    out << "  alpha  \n";
    out << "\n";
    out << "\tbeta\t\n";
    out << "   \n";
  }

  const auto lines = tinygs::readlines(file_path.string());
  ASSERT_EQ(lines.size(), 2U);
  EXPECT_EQ(lines[0], "alpha");
  EXPECT_EQ(lines[1], "beta");

  std::filesystem::remove_all(base);
}

TEST(FileUtilsTest, EnsureCreatesDirectory) {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "tinygs_gtest_ensure";
  std::filesystem::remove_all(dir);

  tinygs::ensure(dir.string());

  EXPECT_TRUE(std::filesystem::exists(dir));
  EXPECT_TRUE(std::filesystem::is_directory(dir));

  std::filesystem::remove_all(dir);
}

}  // namespace
