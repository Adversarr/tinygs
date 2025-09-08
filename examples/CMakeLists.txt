# Examples CMakeLists.txt for tinygs

function(tinygs_add_example name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE tinygs::tinygs)
    set_target_properties(${name} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )
endfunction()

tinygs_add_example(check_load_camera)

tinygs_add_example(vis_dataloader)
target_link_libraries(vis_dataloader PRIVATE ${OpenCV_LIBS})
target_include_directories(vis_dataloader PRIVATE ${OpenCV_INCLUDE_DIRS})

tinygs_add_example(vis_init)
target_link_libraries(vis_init PRIVATE ${OpenCV_LIBS})
target_include_directories(vis_init PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(manual_train manual_train.cu)
target_link_libraries(manual_train PRIVATE tinygs::tinygs)
set_target_properties(manual_train PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
target_link_libraries(manual_train PRIVATE ${OpenCV_LIBS})
target_include_directories(manual_train PRIVATE ${OpenCV_INCLUDE_DIRS})
