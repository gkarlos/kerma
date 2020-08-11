file(GLOB_RECURSE KermadSources "*.cpp")

add_executable(kermad ${KermadSources})

set_target_properties(kermad PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_compile_options(kermad PUBLIC "-fPIC")

target_link_libraries(kermad KermaSupport)

target_link_libraries(kermad
    LLVMOption
)