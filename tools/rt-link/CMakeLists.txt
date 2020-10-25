
add_executable(rt-link rt-link.cpp)
set_target_properties(rt-link PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(rt-link KermaRTUtils)
target_link_libraries(rt-link KermaTransforms)