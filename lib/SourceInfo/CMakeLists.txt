set(SourceInfo_SOURCES SourceLoc.cpp
                       SourceRange.cpp
                       SourceInfo.cpp)
add_library(KermaSourceInfo ${SourceInfo_SOURCES})
set_target_properties(KermaSourceInfo PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(KermaSourceInfo LLVMSupport)

