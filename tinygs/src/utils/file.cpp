#include "tinygs/utils/file.hpp"
#include "tinygs/cuda/common_host.hpp"

#include <stdexcept>
#include <filesystem>

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

}