#pragma once

#include <fstream>
#include <string>
#include <vector>

namespace tinygs {

/// @brief Read all non-empty lines from a text file
/// @param path Path to the file to read
std::vector<std::string> readlines(const std::string& path);

/// @brief List all files in a directory
/// @param path Path to the directory to list
/// @param relative Whether to return relative paths
std::vector<std::string> list_folder(const std::string& path, bool relative = false);

/// @brief Ensures the folder exists
void ensure(const std::string& path);

} // namespace tinygs