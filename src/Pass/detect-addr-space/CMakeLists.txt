set(DETECT_ADDR_SPACE_PASS_SOURCES DetectAddrSpace.cpp)
add_library(KermaDetectAddrSpace SHARED ${DETECT_ADDR_SPACE_PASS_SOURCES})
set_target_properties(KermaDetectAddrSpace PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(KermaDetectAddrSpace KermaCuda KermaSupport)


