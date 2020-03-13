file(GLOB KERMA_LIB_CUDA_SOURCES "*.cpp")
llvm_map_components_to_libnames(KERMA_LIB_CUDA Support)
add_library(KermaCuda ${KERMA_LIB_CUDA_SOURCES})
target_link_libraries(KermaCuda KermaSupport ${KERMA_LIB_CUDA})
set_target_properties(KermaCuda PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_compile_options(KermaCuda PUBLIC "-fPIC")

# Kerma LLVM Module Sources
set(Cuda_LLVM_SRC ${KERMA_LIB_CUDA_SOURCES} PARENT_SCOPE)