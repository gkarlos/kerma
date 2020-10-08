set(Utils_SOURCES LLVMMetadata.cpp
                  LLVMShorthands.cpp)

##===--------------------------------------------------------------------------
## Utils (plugin)
##===--------------------------------------------------------------------------
add_library(KermaUtilsPlugin SHARED ${Utils_SOURCES})
target_include_directories(KermaUtilsPlugin PUBLIC ${KERMA_INCLUDE_DIR})
target_compile_definitions(KermaUtilsPlugin PUBLIC KERMA_OPT_PLUGIN)
set_target_properties(KermaUtilsPlugin PROPERTIES COMPILE_FLAGS "-fno-rtti")


##===--------------------------------------------------------------------------
## Utils
##===--------------------------------------------------------------------------
add_library(KermaUtils SHARED ${Utils_SOURCES})
target_include_directories(KermaUtils PUBLIC ${KERMA_INCLUDE_DIR})
set_target_properties(KermaUtils PROPERTIES COMPILE_FLAGS "-fno-rtti")