set(DETECT_KERNELS_PASS_SOURCES DetectKernels.cpp)

add_library(KermaDetectKernelsPass SHARED ${DETECT_KERNELS_PASS_SOURCES} ${KERMA_COMMON_SOURCES})
set_target_properties(KermaDetectKernelsPass PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(KermaDetectKernelsPass KermaCuda KermaSupport)

set(DetectKernels_LLVM_Sources
        ${CMAKE_CURRENT_SOURCE_DIR}/DetectKernels.cpp
        PARENT_SCOPE)