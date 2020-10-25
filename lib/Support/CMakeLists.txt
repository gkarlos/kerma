configure_file(config.h.in ${CMAKE_SOURCE_DIR}/include/kerma/Support/Config.h @ONLY)
configure_file(version.h.in ${CMAKE_SOURCE_DIR}/include/kerma/Support/Version.h @ONLY)

set(Support_SOURCES Demangle.cpp
                    FileSystem.cpp
                    FileUtils.cpp
                    Parse.cpp
                    Version.cpp )

add_library(KermaSupportOpt SHARED ${Support_SOURCES})

set_target_properties(KermaSupportOpt PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(KermaSupportOpt LLVMDemangle stdc++fs)

add_library(KermaSupport SHARED ${Support_SOURCES})
set_target_properties(KermaSupport PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(KermaSupport PUBLIC stdc++fs)

# Clean up config auto-generated files
set_property(TARGET KermaSupport
    APPEND
    PROPERTY ADDITIONAL_CLEAN_FILES ${CMAKE_SOURCE_DIR}/include/kerma/Support/Config.h
                                    ${CMAKE_SOURCE_DIR}/include/kerma/Support/Version.h)