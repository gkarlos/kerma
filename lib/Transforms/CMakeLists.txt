
set(Canonicalize_SOURCES Canonicalize/BreakConstantGEP.cpp
                         Canonicalize/Canonicalizer.cpp
                         Canonicalize/GepifyMem.cpp
                         Canonicalize/SimplifyGEP.cpp)
set(MaterializeDims_SOURCES MaterializeDims.cpp)
set(MaterializeIdx_SOURCES MaterializeIdx.cpp)
set(InstruMemOpPrintf_SOURCES InstruMemOpPrintf.cpp)
set(LinkDeviceRT_SOURCES LinkDeviceRT.cpp)

##===--------------------------------------------------------------------------
## KermaTransforms (plugin)
##===--------------------------------------------------------------------------
add_library(KermaTransformsPlugin SHARED ${Canonicalize_SOURCES}
                                         ${MaterializeDims_SOURCES}
                                         ${MaterializeIdx_SOURCES}
                                         ${InstruMemOpPrintf_SOURCES})
target_include_directories(KermaTransformsPlugin PUBLIC ${KERMA_INCLUDE_DIR})
target_compile_definitions(KermaTransformsPlugin PUBLIC KERMA_OPT_PLUGIN)
set_target_properties(KermaTransformsPlugin PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(KermaTransformsPlugin KermaAnalysisPlugin
                                            KermaBase
                                            KermaNVVM
                                            KermaUtilsPlugin
                                            KermaRTUtils)

##===--------------------------------------------------------------------------
## KermaTransforms
##===--------------------------------------------------------------------------
add_library(KermaTransforms SHARED ${Canonicalize_SOURCES}
                                   ${MaterializeDims_SOURCES}
                                   ${MaterializeIdx_SOURCES}
                                   ${InstruMemOpPrintf_SOURCES}
                                   ${LinkDeviceRT_SOURCES})
target_include_directories(KermaTransforms PUBLIC ${KERMA_INCLUDE_DIR})
set_target_properties(KermaTransforms PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(KermaTransforms KermaAnalysis
                                      KermaBase
                                      KermaNVVM
                                      KermaSupport
                                      KermaUtils
                                      LLVMIRReader
                                      LLVMLinker)