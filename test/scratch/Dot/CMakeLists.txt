add_executable(DotTest DotTest.cpp)
# llvm_map_components_to_libnames(DOT_TEST_LLVM_COMPONENTS Support Passes)
target_link_libraries(DotTest ${DOT_TEST_LLVM_COMPONENTS} KermaDG)
set_target_properties(DotTest PROPERTIES COMPILE_FLAGS "-fno-rtti")