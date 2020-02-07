set(KERMA_LIB_CUDA_SOURCES CudaProgram.cpp
                           CudaKernel.cpp
                           Util.cpp)

llvm_map_components_to_libnames(KERMA_LIB_CUDA_LLVM_COMPONENTS Demangle)

add_library(KermaCuda ${KERMA_LIB_CUDA_SOURCES})
set_target_properties(KermaCuda PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_compile_options(KermaCuda PRIVATE "-fPIC")

target_link_libraries(KermaCuda ${KERMA_LIB_CUDA_LLVM_COMPONENTS})