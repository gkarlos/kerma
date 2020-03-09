
## TODO Create a loop to to build the tests instead of 1 by 1

### TestCuda
add_executable(TestCuda TestCuda.cpp)
llvm_map_components_to_libnames(TestCuda_LLVM_COMPONENTS Support)
target_link_libraries(TestCuda gtest gtest_main KermaCuda KermaSupport ${TestCuda_LLVM_COMPONENTS})
set_target_properties(TestCuda PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_compile_options(TestCuda PUBLIC "-fPIC")
add_test(NAME libKermaCuda.Cuda COMMAND TestCuda)

### TestCudaDim
add_executable(TestCudaDim TestCudaDim.cpp)
llvm_map_components_to_libnames(TestCudaDim_LLVM_COMPONENTS Support)
target_link_libraries(TestCudaDim gtest gtest_main KermaCuda KermaSupport ${TestCudaDim_LLVM_COMPONENTS})
set_target_properties(TestCudaDim PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_compile_options(TestCudaDim PUBLIC "-fPIC")
add_test(NAME libKermaCuda.CudaDim COMMAND TestCudaDim)

### TestCudaModule
add_executable(TestCudaModule TestCudaModule.cpp)
llvm_map_components_to_libnames(TestCudaModule_LLVM_COMPONENTS Core Support Passes)
target_link_libraries(TestCudaModule gtest gtest_main KermaCuda KermaDetectKernelsPass
                        ${TestCudaModule_LLVM_COMPONENTS})
set_target_properties(TestCudaModule PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_compile_options(TestCudaModule PUBLIC "-fPIC")
add_test(NAME libKermaCuda.CudaProgram COMMAND TestCudaModule)

### TestCudaKernel
add_executable(TestCudaKernel TestCudaKernel.cpp)
llvm_map_components_to_libnames(TestCudaKernel_LLVM_COMPONENTS Core Support Passes)
target_link_libraries(TestCudaKernel gtest gtest_main 
                        KermaCuda KermaSupport KermaDetectKernelsPass
                        ${TestCudaKernel_LLVM_COMPONENTS})
set_target_properties(TestCudaKernel PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_compile_options(TestCudaKernel PUBLIC "-fPIC")
add_test(NAME libKermaCuda.CudaKernel COMMAND TestCudaKernel)