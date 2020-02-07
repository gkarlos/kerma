add_library(LLVMKermaTestPass MODULE kerma-test-pass.cpp)

set_target_properties(LLVMKermaTestPass PROPERTIES COMPILE_FLAGS "-fno-rtti")

