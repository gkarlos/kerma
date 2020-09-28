set(DetectKernels_SOURCES DetectKernels.cpp)
set(InferAddressSpaces_SOURCES InferAddressSpaces.cpp)
set(Delinearize_SOURCES Delinearize.cpp)

##===--------------------------------------------------------------------------
## Build standalone Opt plugins for each pass so tha we can easily test a
## specific pass in opt. It is important that these libraries are not linked
## with any LLVM libraries to avoid duplicate cmdline opts errors in Opt
##===--------------------------------------------------------------------------

##===--------------------------------------------------------------------------
## DetectKernels (plugin)
##===--------------------------------------------------------------------------
add_library(DetectKernelsPlugin SHARED ${DetectKernels_SOURCES})
target_include_directories(DetectKernelsPlugin PRIVATE ${KERMA_INCLUDE_DIR})
set_target_properties(DetectKernelsPlugin PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(DetectKernelsPlugin KermaSupportOpt)

##===--------------------------------------------------------------------------
## Delinearize (plugin)
##===--------------------------------------------------------------------------
add_library(DelinearizePlugin SHARED ${Delinearize_SOURCES})
target_include_directories(DelinearizePlugin PRIVATE ${KERMA_INCLUDE_DIR})
set_target_properties(DelinearizePlugin PROPERTIES COMPILE_FLAGS "-fno-rtti")

##===--------------------------------------------------------------------------
## InferAddressSpaces (plugin)
##===--------------------------------------------------------------------------
add_library(InferAddressSpacesPlugin SHARED ${InferAddressSpaces_SOURCES})
target_include_directories(InferAddressSpacesPlugin PRIVATE ${KERMA_INCLUDE_DIR})
set_target_properties(InferAddressSpacesPlugin PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(InferAddressSpacesPlugin DetectKernelsPlugin)

##===--------------------------------------------------------------------------
## Analysis
##===--------------------------------------------------------------------------
add_library(KermaAnalysis SHARED ${DetectKernels_SOURCES}
                                 ValueName.cpp
                                 InferDimensions.cpp
                                 ${InferAddressSpaces_SOURCES}
                                 ${Delinearize_SOURCES})
target_include_directories(KermaAnalysis PRIVATE ${KERMA_INCLUDE_DIR})
set_target_properties(KermaAnalysis PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(KermaAnalysis KermaSupport KermaNVVM)