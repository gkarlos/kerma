set(DetectKernels_SOURCES DetectKernels.cpp)
set(InferAddressSpaces_SOURCES InferAddressSpaces.cpp)
set(Names_SOURCES Names.cpp)
set(Typecheck_SOURCES Typecheck.cpp)
set(DataDep_SOURCES DataDependency.cpp)

file(GLOB_RECURSE Analysis_SOURCES DetectKernels.cpp
                                   InferAddressSpaces.cpp
                                   Names.cpp
                                   Typecheck.cpp)

##===--------------------------------------------------------------------------
## Analysis (plugin)
##===--------------------------------------------------------------------------
add_library(KermaAnalysisPlugin SHARED ${Analysis_SOURCES})
target_include_directories(KermaAnalysisPlugin PUBLIC ${KERMA_INCLUDE_DIR})
target_compile_definitions(KermaAnalysisPlugin PUBLIC KERMA_OPT_PLUGIN)
set_target_properties(KermaAnalysisPlugin PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(KermaAnalysisPlugin KermaSupportOpt KermaUtilsPlugin KermaBaseOpt KermaNVVMOpt)


##===--------------------------------------------------------------------------
## Analysis
##===--------------------------------------------------------------------------
add_library(KermaAnalysis SHARED ${Analysis_SOURCES})
target_include_directories(KermaAnalysis PUBLIC ${KERMA_INCLUDE_DIR})
# set_target_properties(KermaAnalysis PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(KermaAnalysis PRIVATE KermaUtils
                                            KermaNVVM
                                            KermaBase)