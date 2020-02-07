set(MEM_OP_GRAPH_PASS_SOURCES MemOpGraphPass.cpp)
add_library(KermaMemOpGraphPass SHARED ${MEM_OP_GRAPH_PASS_SOURCES})
#llvm_map_components_to_libnames(DETECT_KERNELS_PASS_COMPONENTS Demangle)
set_target_properties(KermaMemOpGraphPass PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(KermaMemOpGraphPass KermaCuda KermaSupport)
target_compile_options(KermaMemOpGraphPass PUBLIC "-fPIC")

