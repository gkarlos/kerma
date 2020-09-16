set(MaterializeDims_SOURCES MaterializeDims.cpp)
set(MaterializeIdx_SOURCES MaterializeIdx.cpp)
set(InstruMemOpPrintf_SOURCES InstruMemOpPrintf.cpp)
set(LinkDeviceRT_SOURCES LinkDeviceRT.cpp)

##===--------------------------------------------------------------------------
## Build standalone Opt plugins for each pass so that we can easily test a
## specific pass in opt. It is important that these libraries are not linked
## with any LLVM libraries to avoid duplicate cmdline opts errors in Opt
##===--------------------------------------------------------------------------

##===--------------------------------------------------------------------------
## MaterializeDims (plugin)
##===--------------------------------------------------------------------------
add_library(MaterializeDimsPlugin SHARED ${MaterializeDims_SOURCES})
target_compile_definitions(MaterializeDimsPlugin PUBLIC KERMA_OPT_PLUGIN)
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
target_compile_definitions(MaterializeIdxPlugin PUBLIC KERMA_OPT_PLUGIN)
target_include_directories(MaterializeIdxPlugin PUBLIC ${KERMA_INCLUDE_DIR})
set_target_properties(MaterializeIdxPlugin PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(MaterializeIdxPlugin DetectKernelsPlugin
                                           KermaNVVMOpt
                                           KermaBaseOpt
                                           KermaSupportOpt
                                           )

##===--------------------------------------------------------------------------
## MemOpInstrumentation (plugin)
##===--------------------------------------------------------------------------
add_library(InstruMemOpPrintf SHARED ${InstruMemOpPrintf_SOURCES})
target_compile_definitions(InstruMemOpPrintf PUBLIC KERMA_OPT_PLUGIN)
target_include_directories(InstruMemOpPrintf PUBLIC ${KERMA_INCLUDE_DIR})
set_target_properties(InstruMemOpPrintf PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(InstruMemOpPrintf
                      DetectKernelsPlugin
                      KermaRTUtilsOpt)

##===--------------------------------------------------------------------------
## KermaTransforms
##===--------------------------------------------------------------------------
add_library(KermaTransforms SHARED ${MaterializeDims_SOURCES}
                                   ${MaterializeIdx_SOURCES}
                                   ${LinkDeviceRT_SOURCES}
                                   ${InstrMemOpPrintf_SOURCES})
target_include_directories(KermaTransforms PUBLIC ${KERMA_INCLUDE_DIR})
set_target_properties(KermaTransforms PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(KermaTransforms KermaAnalysis
                                      KermaBase
                                      KermaNVVM
                                      KermaSupport
                                      LLVMIRReader
                                      LLVMLinker)