set(KERMA_CFG_SOURCES kerma-cfg.cpp)

llvm_map_components_to_libnames(KERMA_CFG_LLVM_COMPONENTS Core Support IRReader Analysis Demangle)

add_executable(kerma-cfg ${KERMA_CFG_SOURCES})
set_target_properties(kerma-cfg PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(kerma-cfg KermaDetectKernelsPass KermaCuda KermaSupport ${KERMA_CFG_LLVM_COMPONENTS})

#target_compile_options(KermaDetectKernelsPass PUBLIC "-fPIC")

