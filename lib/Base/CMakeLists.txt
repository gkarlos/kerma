file(GLOB_RECURSE BaseSources "*.cpp")
add_library(KermaBase SHARED ${BaseSources})
# set_target_properties(KermaBase PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(KermaBase PUBLIC KermaNVVM KermaSupport)

add_library(KermaBaseOpt SHARED ${BaseSources})
set_target_properties(KermaBaseOpt PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(KermaBaseOpt KermaSupportOpt)