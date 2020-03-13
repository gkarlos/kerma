add_subdirectory(Support)
add_subdirectory(Cuda)
add_subdirectory(Pass)

### Build the Kerma LLVM Module ###
add_library(KermaLLVMModule MODULE ${Pass_LLVM_SRC}
                                   ${Cuda_LLVM_SRC}
                                   ${Support_LLVM_SRC})

set_target_properties(KermaLLVMModule PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_compile_options(KermaLLVMModule PUBLIC "-fPIC")