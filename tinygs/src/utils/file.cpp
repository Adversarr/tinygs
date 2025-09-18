#include "tinygs/utils/file.hpp"
#include "tinygs/cuda/common_host.hpp"

#include <stdexcept>
#include <filesystem>
#include <algorithm>

namespace tinygs {

std::vector<std::string> readlines(const std::string& path) {
    // Check if file exists
    if (!std::filesystem::exists(path)) {
        const std::string error_msg = fmt::format("File does not exist: {}", path);
        log_error(error_msg);
        throw std::runtime_error(error_msg);
    }
    
    log_success("Reading file: {}", path);
    
    std::vector<std::string> lines;
    std::ifstream file(path);
    
    if (!file.is_open()) {
        const std::string error_msg = fmt::format("Failed to open file: {}", path);
        log_error(error_msg);
        throw std::runtime_error(error_msg);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Trim whitespace and skip empty lines
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    
    if (file.bad()) {
        const std::string error_msg = fmt::format("Error occurred while reading file: {}", path);
        log_error(error_msg);
        throw std::runtime_error(error_msg);
    }
    
    log_success("Successfully read {} lines from file: {}", lines.size(), path);
    return lines;
}

std::vector<std::string> list_folder(const std::string& path, bool relative) {
    // Check if directory exists
    if (!std::filesystem::exists(path)) {
        const std::string error_msg = fmt::format("Directory does not exist: {}", path);
        log_error(error_msg);
        throw std::runtime_error(error_msg);
    }
    
    // Check if path is actually a directory
    if (!std::filesystem::is_directory(path)) {
        const std::string error_msg = fmt::format("Path is not a directory: {}", path);
        log_error(error_msg);
        throw std::runtime_error(error_msg);
    }
    
    log_success("Listing directory: {}", path);
    
    std::vector<std::string> entries;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (relative) {
                entries.push_back(entry.path().string());
            } else {
                entries.push_back(entry.path().filename().string());
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        const std::string error_msg = fmt::format("Error occurred while listing directory {}: {}", path, e.what());
        log_error(error_msg);
        throw std::runtime_error(error_msg);
    }
    
    // Sort entries for consistent ordering
    std::sort(entries.begin(), entries.end());
    
    log_success("Successfully listed {} entries from directory: {}", entries.size(), path);
    return entries;
}

void ensure(const std::string& path) {
    // Validate input path
    if (path.empty()) {
        const std::string error_msg = "Path cannot be empty";
        log_error(error_msg);
        throw std::invalid_argument(error_msg);
    }
    
    // Check if path already exists
    if (std::filesystem::exists(path)) {
        // If it exists but is not a directory, throw an error
        if (!std::filesystem::is_directory(path)) {
            const std::string error_msg = fmt::format("Path exists but is not a directory: {}", path);
            log_error(error_msg);
            throw std::runtime_error(error_msg);
        }
        
        // Directory already exists, nothing to do
        log_success("Directory already exists: {}", path);
        return;
    }
    
    // Create the directory recursively
    try {
        std::filesystem::create_directories(path);
        log_success("Successfully created directory: {}", path);
    } catch (const std::filesystem::filesystem_error& e) {
        const std::string error_msg = fmt::format("Failed to create directory {}: {}", path, e.what());
        log_error(error_msg);
        throw std::runtime_error(error_msg);
    }
}

}