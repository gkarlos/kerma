set(DETECT_KERNELS_PASS_SOURCES DetectKernels.cpp)

add_library(KermaDetectKernelsPass SHARED ${DETECT_KERNELS_PASS_SOURCES})
#llvm_map_components_to_libnames(DETECT_KERNELS_PASS_COMPONENTS Demangle)
set_target_properties(KermaDetectKernelsPass PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(KermaDetectKernelsPass KermaCuda KermaSupport)
#target_compile_options(KermaDetectKernelsPass PUBLIC "-fPIC")

