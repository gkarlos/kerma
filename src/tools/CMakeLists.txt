set(KERMA_OPT_SOURCES kerma-opt.cpp)
#../cuda/CudaProgram.cpp ../../include/kerma/passes/memop-graph/MemOpDeps.h ../../include/kerma/cuda/CudaKernel.h ../../include/kerma/passes/Util.h)

llvm_map_components_to_libnames(KERMA_OPT_LLVM_COMPONENTS Support Passes)

add_executable(kerma-opt ${KERMA_OPT_SOURCES} )
target_link_libraries(kerma-opt ${KERMA_OPT_LLVM_COMPONENTS} KermaDetectKernelsPass KermaDG)
set_target_properties(kerma-opt PROPERTIES COMPILE_FLAGS "-fno-rtti")
#target_link_libraries(kerma-opt ${KERMA_OPT_LLVM_LIBS})
#target_link_libraries(kerma-opt LLVMKermaDetectKernelsPass)
#
#set_target_properties(LLVMKermaDetectKernelsPass PROPERTIES COMPILE_FLAGS "${KERMA_OPT_LLVM_LIBS_FLAGS}")