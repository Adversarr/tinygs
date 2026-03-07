include(GoogleTest)

if(TINYGS_BACKEND STREQUAL "CUDA")
    add_executable(tinygs_basic_test
        basic_test.cpp
    )

target_link_libraries(tinygs_basic_test
    PRIVATE
        tinygs::tinygs
        GTest::gtest_main
)

set_target_properties(tinygs_basic_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(tinygs_basic_test)

add_executable(tinygs_camera_ext_test
    camera_ext_test.cpp
)

target_link_libraries(tinygs_camera_ext_test
    PRIVATE
        tinygs::tinygs
        GTest::gtest_main
)

set_target_properties(tinygs_camera_ext_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(tinygs_camera_ext_test)

add_executable(tinygs_camera_loader_test
    camera_loader_test.cpp
)

target_link_libraries(tinygs_camera_loader_test
    PRIVATE
        tinygs::tinygs
        GTest::gtest_main
)

set_target_properties(tinygs_camera_loader_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(tinygs_camera_loader_test)

add_executable(tinygs_camera_test
    camera_test.cpp
)

target_link_libraries(tinygs_camera_test
    PRIVATE
        tinygs::tinygs
        GTest::gtest_main
)

set_target_properties(tinygs_camera_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(tinygs_camera_test)

add_executable(tinygs_factory_test
    factory_test.cpp
)

target_link_libraries(tinygs_factory_test
    PRIVATE
        tinygs::tinygs
        GTest::gtest_main
)

set_target_properties(tinygs_factory_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(tinygs_factory_test)

add_executable(tinygs_gaussian_test
    gaussian_test.cpp
)

target_link_libraries(tinygs_gaussian_test
    PRIVATE
        tinygs::tinygs
        GTest::gtest_main
)

set_target_properties(tinygs_gaussian_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(tinygs_gaussian_test)

add_executable(tinygs_initialization_factory_test
    initialization_factory_test.cpp
)

target_link_libraries(tinygs_initialization_factory_test
    PRIVATE
        tinygs::tinygs
        GTest::gtest_main
)

set_target_properties(tinygs_initialization_factory_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(tinygs_initialization_factory_test)

add_executable(tinygs_initialization_test
    initialization_test.cpp
)

target_link_libraries(tinygs_initialization_test
    PRIVATE
        tinygs::tinygs
        GTest::gtest_main
)

set_target_properties(tinygs_initialization_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(tinygs_initialization_test)

add_executable(tinygs_knn_algorithm_test
    knn_algorithm_test.cpp
)

target_link_libraries(tinygs_knn_algorithm_test
    PRIVATE
        tinygs::tinygs
        GTest::gtest_main
)

set_target_properties(tinygs_knn_algorithm_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(tinygs_knn_algorithm_test)

add_executable(tinygs_lr_scheduler_factory_test
    lr_scheduler_factory_test.cpp
)

target_link_libraries(tinygs_lr_scheduler_factory_test
    PRIVATE
        tinygs::tinygs
        GTest::gtest_main
)

set_target_properties(tinygs_lr_scheduler_factory_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(tinygs_lr_scheduler_factory_test)

add_executable(tinygs_lr_scheduler_test
    lr_scheduler_test.cpp
)

target_link_libraries(tinygs_lr_scheduler_test
    PRIVATE
        tinygs::tinygs
        GTest::gtest_main
)

set_target_properties(tinygs_lr_scheduler_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(tinygs_lr_scheduler_test)

add_executable(tinygs_backend_types_test
    backend_types_test.cpp
)

target_link_libraries(tinygs_backend_types_test
    PRIVATE
        tinygs::tinygs
        GTest::gtest_main
)

set_target_properties(tinygs_backend_types_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(tinygs_backend_types_test)

add_executable(tinygs_backend_runtime_contract_test
    backend_runtime_contract_test.cpp
)

target_link_libraries(tinygs_backend_runtime_contract_test
    PRIVATE
        tinygs::tinygs
        GTest::gtest_main
)

set_target_properties(tinygs_backend_runtime_contract_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(tinygs_backend_runtime_contract_test)

add_executable(tinygs_buffer_utils_test
    buffer_utils_test.cpp
)

target_link_libraries(tinygs_buffer_utils_test
    PRIVATE
        tinygs::tinygs
        GTest::gtest_main
)

set_target_properties(tinygs_buffer_utils_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(tinygs_buffer_utils_test)

add_executable(tinygs_optimizer_factory_test
    optimizer_factory_test.cpp
)

target_link_libraries(tinygs_optimizer_factory_test
    PRIVATE
        tinygs::tinygs
        GTest::gtest_main
)

set_target_properties(tinygs_optimizer_factory_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(tinygs_optimizer_factory_test)

add_executable(tinygs_pcg32_test
    pcg32_test.cpp
)

target_link_libraries(tinygs_pcg32_test
    PRIVATE
        tinygs::tinygs
        GTest::gtest_main
)

set_target_properties(tinygs_pcg32_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(tinygs_pcg32_test)

add_executable(tinygs_pitched_ptr_test
    pitched_ptr_test.cpp
)

target_link_libraries(tinygs_pitched_ptr_test
    PRIVATE
        tinygs::tinygs
        GTest::gtest_main
)

set_target_properties(tinygs_pitched_ptr_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(tinygs_pitched_ptr_test)

add_executable(tinygs_pointcloud_test
    pointcloud_test.cpp
)

target_link_libraries(tinygs_pointcloud_test
    PRIVATE
        tinygs::tinygs
        GTest::gtest_main
)

set_target_properties(tinygs_pointcloud_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(tinygs_pointcloud_test)

add_executable(tinygs_pose_opt_test
    pose_opt_test.cpp
)

target_link_libraries(tinygs_pose_opt_test
    PRIVATE
        tinygs::tinygs
        GTest::gtest_main
)

set_target_properties(tinygs_pose_opt_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(tinygs_pose_opt_test)

add_executable(tinygs_strategy_factory_test
    strategy_factory_test.cpp
)

target_link_libraries(tinygs_strategy_factory_test
    PRIVATE
        tinygs::tinygs
        GTest::gtest_main
)

set_target_properties(tinygs_strategy_factory_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(tinygs_strategy_factory_test)

add_executable(tinygs_tile_utils_test
    tile_utils_test.cpp
)

target_link_libraries(tinygs_tile_utils_test
    PRIVATE
        tinygs::tinygs
        GTest::gtest_main
)

set_target_properties(tinygs_tile_utils_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

    gtest_discover_tests(tinygs_tile_utils_test)
endif()

if(TINYGS_BACKEND STREQUAL "METAL")
    add_executable(tinygs_metal_runtime_test
        metal_runtime_test.cpp
    )

    target_link_libraries(tinygs_metal_runtime_test
        PRIVATE
            tinygs::tinygs
            GTest::gtest_main
    )

    set_target_properties(tinygs_metal_runtime_test PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    add_test(NAME tinygs_metal_runtime_test COMMAND tinygs_metal_runtime_test)
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_test(
    NAME tinygs_phase0_backend_boundaries
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/check_backend_boundaries.py check
)
