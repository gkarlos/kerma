
add_executable(TestCuda TestCuda.cpp)
target_link_libraries(TestCuda gtest gtest_main KermaCuda KermaSupport) 
set_target_properties(TestCuda PROPERTIES COMPILE_FLAGS "-fno-rtti")
add_test(NAME libKermaCuda.Cuda COMMAND TestCuda)