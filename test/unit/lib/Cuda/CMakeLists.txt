
### TestCuda
add_executable(TestCuda TestCuda.cpp)
llvm_map_components_to_libnames(TestCuda_LLVM_COMPONENTS Support)
target_link_libraries(TestCuda gtest gtest_main KermaCuda KermaSupport ${TestCuda_LLVM_COMPONENTS})
set_target_properties(TestCuda PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_compile_options(TestCuda PUBLIC "-fPIC")
add_test(NAME libKermaCuda.Cuda COMMAND TestCuda)

### TestCudaKernel
add_executable(TestCudaKernel TestCudaKernel.cpp)
llvm_map_components_to_libnames(TestCudaKernel_LLVM_COMPONENTS Support Passes)
target_link_libraries(TestCudaKernel gtest gtest_main 
                        KermaCuda KermaSupport KermaDetectKernelsPass
                        ${TestCudaKernel_LLVM_COMPONENTS})
set_target_properties(TestCudaKernel PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_compile_options(TestCudaKernel PUBLIC "-fPIC")
add_test(NAME libKermaCuda.CudaKernel COMMAND TestCudaKernel)
