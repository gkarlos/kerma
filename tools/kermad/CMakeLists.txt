file(GLOB_RECURSE KermadSources "*.cpp")

add_executable(kermad ${KermadSources})

target_link_libraries(kermad PUBLIC ${Boost_LIBRARIES}
                                    cxxtools
                                    cxxtools-json
                                    Kerma
                                    )