set(SOURCE_INFO_PASS_SOURCES SourceInfo.cpp)
# llvm_map_components_to_libnames(KERMA_OPT_LLVM_COMPONENTS Support Passes)
add_library(SourceInfoPass MODULE ${SOURCE_INFO_PASS_SOURCES})
set_target_properties(SourceInfoPass PROPERTIES COMPILE_FLAGS "-fno-rtti")