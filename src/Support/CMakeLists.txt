file(GLOB KERMA_LIB_SUPPORT_SOURCES "*.cpp")
llvm_map_components_to_libnames(KERMA_LIB_SUPPORT_LLVM_COMPONENTS Core Demangle)
add_library(KermaSupport ${KERMA_LIB_SUPPORT_SOURCES})
target_link_libraries(KermaSupport ${KERMA_LIB_SUPPORT_LLVM_COMPONENTS} ${Boost_LIBRARIES})
set_target_properties(KermaSupport PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_compile_options(KermaSupport PUBLIC "-fPIC")

## Kerma LLVM Module Sources
set(Support_LLVM_Sources ${KERMA_LIB_SUPPORT_SOURCES} PARENT_SCOPE)
set(Support_LLVM_Components Core Demangle PARENT_SCOPE)