set(DetectKernels_SOURCES DetectKernels.cpp)
set(InferAddressSpaces_SOURCES InferAddressSpaces.cpp)
set(Symbolize_SOURCES Symbolize.cpp)
set(Typecheck_SOURCES Typecheck.cpp)

##===--------------------------------------------------------------------------
## Analysis (plugin)
##===--------------------------------------------------------------------------
add_library(KermaAnalysisPlugin SHARED ${DetectKernels_SOURCES}
                                       ${InferAddressSpaces_SOURCES}
                                       ${Symbolize_SOURCES}
                                       ${Typecheck_SOURCES})
target_include_directories(KermaAnalysisPlugin PUBLIC ${KERMA_INCLUDE_DIR})
target_compile_definitions(KermaAnalysisPlugin PUBLIC KERMA_OPT_PLUGIN)
set_target_properties(KermaAnalysisPlugin PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(KermaAnalysisPlugin KermaSupport KermaUtilsPlugin KermaNVVM)


##===--------------------------------------------------------------------------
## Analysis
##===--------------------------------------------------------------------------
add_library(KermaAnalysis SHARED ${DetectKernels_SOURCES}
                                 ${InferAddressSpaces_SOURCES}
                                 ${Symbolize_SOURCES}
                                 ${Typecheck_SOURCES})
target_include_directories(KermaAnalysis PUBLIC ${KERMA_INCLUDE_DIR})
set_target_properties(KermaAnalysis PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(KermaAnalysis KermaSupport KermaUtils KermaNVVM)