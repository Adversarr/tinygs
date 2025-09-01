# Tests CMakeLists.txt for tinygs

# Find required packages
find_package(tinygs REQUIRED)

# Try to find Google Test
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    # If GTest is not found, fetch it
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50f33f9142a332e68d798063b2dee5.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Include GoogleTest
include(GoogleTest)

# Test executable
add_executable(tinygs_tests
    test_main.cpp
    test_basic.cpp
    test_initialization.cpp
)

# Link libraries
target_link_libraries(tinygs_tests
    tinygs::tinygs
    gtest
    gtest_main
    gmock
    gmock_main
)

# Set properties
set_target_properties(tinygs_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(tinygs_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Discover tests
gtest_discover_tests(tinygs_tests)

# CUDA tests (if CUDA is available)
if(CMAKE_CUDA_COMPILER)
    add_executable(tinygs_cuda_tests
        test_cuda.cu
        test_main.cpp
    )
    
    target_link_libraries(tinygs_cuda_tests
        tinygs::tinygs
        gtest
        gtest_main
    )
    
    set_target_properties(tinygs_cuda_tests PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    
    gtest_discover_tests(tinygs_cuda_tests)
endif()