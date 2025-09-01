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

/**
 * @brief List all files in a directory
 * @param path Path to the directory to list
 * @param relative Whether to return relative paths
 * @return Vector of strings containing all file names
 * @throws std::runtime_error if directory cannot be listed
 */
std::vector<std::string> list_folder(const std::string& path, bool relative = false);

} // namespace tinygs