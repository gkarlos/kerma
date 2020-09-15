set(KermaRTUtils_SOURCES Util.cpp)
set(KermaRT KermaRT.cpp)


##===--------------------------------------------------------------------------
## KermaRTUtilsOpt
##===--------------------------------------------------------------------------
add_library(KermaRTUtilsOpt SHARED ${KermaRTUtils_SOURCES})
target_compile_definitions(KermaRTUtilsOpt PUBLIC -DLLVM_ENABLE_STATS KERMA_OPT_PLUGIN)
target_include_directories(KermaRTUtilsOpt PUBLIC ${KERMA_INCLUDE_DIR})
set_target_properties(KermaRTUtilsOpt PROPERTIES COMPILE_FLAGS "-fno-rtti")

##===--------------------------------------------------------------------------
## KermaRTUtilsOpt
##===--------------------------------------------------------------------------
add_library(KermaRTUtils SHARED ${KermaRTUtils_SOURCES})
target_compile_definitions(KermaRTUtils PUBLIC -DLLVM_ENABLE_STATS KERMA_OPT_PLUGIN)
target_include_directories(KermaRTUtils PUBLIC ${KERMA_INCLUDE_DIR})
set_target_properties(KermaRTUtils PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(KermaRTUtils LLVMCore)


##
## TODO Create .bc of KermaRT
##


