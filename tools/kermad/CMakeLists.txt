file(GLOB_RECURSE KermadSources "*.cpp")

add_executable(kermad ${KermadSources})

# target_compile_options(kermad PUBLIC "-fPIC")

# message(${LLVMIRReader})
target_link_libraries(kermad PUBLIC ${Boost_LIBRARIES}
                                    cxxtools
                                    cxxtools-json
                                    Kerma
                                    )
# target_link_libraries(kermad PRIVATE KermaAnalysis
#                                      KermaBase
#                                      KermaCompile
#                                      KermaSourceInfo
#                                      KermaSupport)