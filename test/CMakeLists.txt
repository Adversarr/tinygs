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

include(GoogleTest)
gtest_discover_tests(tinygs_basic_test)
