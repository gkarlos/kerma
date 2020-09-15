
add_executable(rt-check rt-check.cpp)
set_target_properties(rt-check PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(rt-check KermaRTUtils)
target_link_libraries(rt-check LLVMIRReader)