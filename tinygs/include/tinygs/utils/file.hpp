#pragma once

#include <fstream>
#include <string>
#include <vector>

namespace tinygs {

/// @brief Read all non-empty lines from a text file
/// @param path Path to the file to read
std::vector<std::string> readlines(const std::string& path);

/// @brief Ensures the folder exists
void ensure(const std::string& path);

} // namespace tinygs