# Application binaries CMakeLists.txt for tinygs

function(tinygs_add_app name)
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

tinygs_add_app(export_default)
tinygs_add_app(config_train)
tinygs_add_app(single_gs)
tinygs_add_app(loss_psnr_check)
