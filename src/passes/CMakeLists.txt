# add_subdirectory(test-pass)
add_subdirectory(detect-addr-space)
add_subdirectory(detect-kernels)
add_subdirectory(memop-graph)
add_subdirectory(dg)


## Build LLVM Module

set(KERMA_SA_LLVM_MODULE_SOURCES detect-addr-space/DetectAddrSpace.cpp
                                 detect-kernels/DetectKernels.cpp
                                 memop-graph/MemOpGraphPass.cpp
                                 dg/DGDotPass.cpp )

add_library(LLVMKermaStaticAnalysis MODULE ${KERMA_SA_LLVM_MODULE_SOURCES})
set_target_properties(LLVMKermaStaticAnalysis PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(LLVMKermaStaticAnalysis KermaCuda KermaSupport KermaDG)
target_compile_options(LLVMKermaStaticAnalysis PUBLIC "-fPIC")
