file(GLOB_RECURSE BaseSources "*.cpp")
add_library(KermaBase SHARED ${BaseSources})
# set_target_properties(KermaBase PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(KermaBase PRIVATE
    LLVMSupport
)

add_library(KermaBaseOpt SHARED ${BaseSources})
set_target_properties(KermaBase PROPERTIES COMPILE_FLAGS "-fno-rtti")