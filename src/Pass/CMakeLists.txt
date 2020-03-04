# add_subdirectory(test-pass)
# add_subdirectory(detect-addr-space)
add_subdirectory(DetectKernels)
add_subdirectory(memop-graph)
add_subdirectory(dg)


## Build LLVM Module

set(KERMA_SA_LLVM_MODULE_SOURCES DetectKernels/DetectKernels.cpp)

add_library(LLVMKermaStaticAnalysis MODULE ${KERMA_SA_LLVM_MODULE_SOURCES})
set_target_properties(LLVMKermaStaticAnalysis PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(LLVMKermaStaticAnalysis KermaCuda KermaSupport)
target_compile_options(LLVMKermaStaticAnalysis PUBLIC "-fPIC")
