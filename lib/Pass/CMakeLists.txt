set(DETECT_KERNELS_PASS_SOURCES DetectKernels.cpp)

llvm_map_components_to_libnames(mbt_llvm_deps Demangle)

add_library(mbt SHARED MaterializeBlockThread.cpp)
target_link_libraries(mbt ${mbt_llvm_deps})
set_target_properties(mbt PROPERTIES COMPILE_FLAGS "-fno-rtti")


# target_link_libraries(KermaDetectKernelsPass KermaCuda KermaSupport)

# set(DetectKernels_LLVM_Sources
#         ${CMAKE_CURRENT_SOURCE_DIR}/DetectKernels.cpp
#         PARENT_SCOPE)