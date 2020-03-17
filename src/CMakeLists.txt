file(GLOB KERMA_COMMON_SOURCES *.cpp)

add_subdirectory(Support)
add_subdirectory(Cuda)
add_subdirectory(Pass)

### Build the Kerma LLVM Module ###
add_library(KermaLLVMModule MODULE ${Pass_LLVM_Sources}
                                   ${Cuda_LLVM_Sources}
                                   ${Support_LLVM_Sources}
                                   GenericOpts.cpp)

set(LLVM_Components "")
list(APPEND LLVM_Components ${Pass_LLVM_Components})
list(APPEND LLVM_Components ${Cuda_LLVM_Components})
list(APPEND LLVM_Components ${Support_LLVM_Components})
list(REMOVE_DUPLICATES LLVM_Components)
message(STATUS "Using LLVM Components: ${LLVM_Components}")

llvm_map_components_to_libnames(KermaLLVMLibs Demangle )
target_link_libraries(KermaLLVMModule ${KermaLLVMLibs})
set_target_properties(KermaLLVMModule PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_compile_options(KermaLLVMModule PUBLIC "-fPIC")