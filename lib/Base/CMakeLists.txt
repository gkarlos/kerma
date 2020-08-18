file(GLOB_RECURSE BaseSources "*.cpp")
add_library(KermaBase SHARED ${BaseSources})
target_link_libraries(KermaBase 
    LLVMSupport
)