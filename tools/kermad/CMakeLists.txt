file(GLOB_RECURSE KermadSources "*.cpp")

add_executable(kermad ${KermadSources})

target_compile_options(kermad PUBLIC "-fPIC")

target_link_libraries(kermad ${Boost_LIBRARIES}
                             cxxtools
                             cxxtools-json
                             KermaCompile
                             KermaSourceInfo)