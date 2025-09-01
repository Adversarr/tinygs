/**
 * @file basic_example.cpp
 * @brief Basic example demonstrating tinygs library usage
 */

#include <tinygs/common.hpp>
#include <iostream>

int main() {
    std::cout << "TinyGS Basic Example" << std::endl;
    std::cout << "Library Version: " << tinygs::getVersion() << std::endl;
    std::cout << "CUDA Available: " << (tinygs::isCudaAvailable() ? "Yes" : "No") << std::endl;
    
    // Create TinyGS instance
    tinygs::TinyGS gs;
    
    // Initialize the system
    if (!gs.initialize()) {
        std::cerr << "Failed to initialize TinyGS" << std::endl;
        return -1;
    }
    
    std::cout << "TinyGS initialized successfully!" << std::endl;
    
    // Cleanup is handled automatically by destructor
    return 0;
}