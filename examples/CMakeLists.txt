# Examples CMakeLists.txt for tinygs

function(tinygs_add_example name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} 
        PRIVATE 
            tinygs::tinygs
            cxxopts::cxxopts
    )
    set_target_properties(${name} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )
endfunction()

tinygs_add_example(check_load_camera)
tinygs_add_example(vis_dataloader)
tinygs_add_example(vis_init)
tinygs_add_example(vis_video)
tinygs_add_example(manual_train)
tinygs_add_example(export_default)

add_executable(manual_train_old manual_train.cu)
target_link_libraries(manual_train_old PRIVATE tinygs::tinygs)