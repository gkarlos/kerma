set(MaterializeDims_SOURCES MaterializeDims.cpp)
set(MaterializeIdx_SOURCES MaterializeDims.cpp)

##===--------------------------------------------------------------------------
## Build standalone Opt plugins for each pass so tha we can easily test a 
## specific pass in opt. It is important that these libraries are not linked 
## with any LLVM libraries to avoid duplicate cmdline opts errors in Opt
##===--------------------------------------------------------------------------

##===--------------------------------------------------------------------------
## MaterializeDims (plugin)
##===--------------------------------------------------------------------------
add_library(MaterializeDimsPlugin SHARED ${MaterializeDims_SOURCES})
target_compile_definitions(MaterializeDimsPlugin PUBLIC -DLLVM_ENABLE_STATS KERMA_OPT_PLUGIN)
target_include_directories(MaterializeDimsPlugin PUBLIC ${KERMA_INCLUDE_DIR})
set_target_properties(MaterializeDimsPlugin PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(MaterializeDimsPlugin DetectKernelsPlugin
                                            KermaNVVMOpt 
                                            KermaBaseOpt 
                                            KermaSupportOpt 
                                            )

##===--------------------------------------------------------------------------
## MaterializeIdx (plugin)
##===--------------------------------------------------------------------------
add_library(MaterializeIdxPlugin SHARED ${MaterializeIdx_SOURCES})
target_compile_definitions(MaterializeIdxPlugin PUBLIC -DLLVM_ENABLE_STATS KERMA_OPT_PLUGIN)
target_include_directories(MaterializeIdxPlugin PUBLIC ${KERMA_INCLUDE_DIR})
set_target_properties(MaterializeIdxPlugin PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(MaterializeIdxPlugin DetectKernelsPlugin
                                           KermaNVVMOpt 
                                           KermaBaseOpt 
                                           KermaSupportOpt 
                                           )

##===--------------------------------------------------------------------------
## libAnalysis
##===--------------------------------------------------------------------------
add_library(Analysis SHARED ${MaterializeDims_SOURCES} ${MaterializeIdx_SOURCES})
target_include_directories(Analysis PUBLIC ${KERMA_INCLUDE_DIR})
set_target_properties(Analysis PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(Analysis KermaBase
                               KermaSupport)