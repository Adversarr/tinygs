#pragma once

#include <fstream>
#include <string>
#include <vector>

namespace tinygs {

/**
 * @brief Read all non-empty lines from a text file
 * @param path Path to the file to read
 * @return Vector of strings containing all non-empty lines
 * @throws std::runtime_error if file cannot be opened
 */
std::vector<std::string> readlines(const std::string& path);

} // namespace tinygs