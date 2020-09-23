set(MaterializeDims_SOURCES MaterializeDims.cpp)
set(MaterializeIdx_SOURCES MaterializeIdx.cpp)
set(InstruMemOpPrintf_SOURCES InstruMemOpPrintf.cpp)
set(CanonLoadsAndStores_SOURCES CanonLoadsAndStores.cpp)
set(NormConstantGEPs_SOURCES NormConstantGEPs.cpp)
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
## CanonLoadsAndStores (plugin)
##===--------------------------------------------------------------------------
add_library(CanonLoadsAndStoresPlugin SHARED ${CanonLoadsAndStores_SOURCES})
target_compile_definitions(CanonLoadsAndStoresPlugin PUBLIC KERMA_OPT_PLUGIN)
target_include_directories(CanonLoadsAndStoresPlugin PUBLIC ${KERMA_INCLUDE_DIR})
set_target_properties(CanonLoadsAndStoresPlugin PROPERTIES COMPILE_FLAGS "-fno-rtti")

##===--------------------------------------------------------------------------
## NormConstantGEPs (plugin)
##===--------------------------------------------------------------------------
add_library(NormConstantGEPsPlugin SHARED ${NormConstantGEPs_SOURCES})
target_compile_definitions(NormConstantGEPsPlugin PUBLIC KERMA_OPT_PLUGIN)
target_include_directories(NormConstantGEPsPlugin PUBLIC ${KERMA_INCLUDE_DIR})
set_target_properties(NormConstantGEPsPlugin PROPERTIES COMPILE_FLAGS "-fno-rtti")

##===--------------------------------------------------------------------------
## InstruMemOpPrintf (plugin)
##===--------------------------------------------------------------------------
add_library(InstruMemOpPrintfPlugin SHARED ${InstruMemOpPrintf_SOURCES})
target_compile_definitions(InstruMemOpPrintfPlugin PUBLIC KERMA_OPT_PLUGIN)
target_include_directories(InstruMemOpPrintfPlugin PUBLIC ${KERMA_INCLUDE_DIR})
set_target_properties(InstruMemOpPrintfPlugin PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(InstruMemOpPrintfPlugin
                      CanonLoadsAndStoresPlugin
                      NormConstantGEPsPlugin
                      KermaAnalysis
                      KermaRTUtilsOpt)



add_library(LinkRT SHARED ${LinkDeviceRT_SOURCES})
target_compile_definitions(LinkRT PUBLIC KERMA_OPT_PLUGIN)
target_include_directories(LinkRT PUBLIC ${KERMA_INCLUDE_DIR})
set_target_properties(LinkRT PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(LinkRT LLVMLinker)


##===--------------------------------------------------------------------------
## KermaTransforms
##===--------------------------------------------------------------------------
add_library(KermaTransforms SHARED ${MaterializeDims_SOURCES}
                                   ${MaterializeIdx_SOURCES}
                                   ${LinkDeviceRT_SOURCES}
                                   ${InstruMemOpPrintf_SOURCES}
                                   ${CanonLoadsAndStores_SOURCES}
                                   ${NormConstantGEPs_SOURCES})
target_include_directories(KermaTransforms PUBLIC ${KERMA_INCLUDE_DIR})
set_target_properties(KermaTransforms PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(KermaTransforms KermaAnalysis
                                      KermaBase
                                      KermaNVVM
                                      KermaSupport
                                      LLVMIRReader
                                      LLVMLinker)