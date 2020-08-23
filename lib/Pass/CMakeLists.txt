##===--------------------------------------------------------------------------
## Build standalone Opt plugins for each pass so that
## we can easily test a specific pass in opt
## 
## It is important that these libraries are not linked with any LLVM
## libraries to avoid duplicate cmdline opts errors as Opt is already
## linked with LLVM
##===--------------------------------------------------------------------------

# DetectKernels
set(DetectKernelsPlugin_SOURCES DetectKernels.cpp)
add_library(DetectKernelsPlugin SHARED ${DetectKernelsPlugin_SOURCES})
target_include_directories(DetectKernelsPlugin PRIVATE ${KERMA_INCLUDE_DIR})
target_link_libraries(DetectKernelsPlugin KermaSupportOpt)
set_target_properties(DetectKernelsPlugin PROPERTIES COMPILE_FLAGS "-fno-rtti")
# set_property(TARGET DetectKernelsPlugin PROPERTY POSITION_INDEPENDENT_CODE ON)

#MaterializeDims
set(MaterializeDimsPlugin_Sources MaterializeDims.cpp DetectKernels.cpp)
add_library(MaterializeDimsPlugin SHARED ${MaterializeDimsPlugin_Sources})
target_compile_definitions(MaterializeDimsPlugin PUBLIC -DLLVM_ENABLE_STATS KERMA_OPT_PLUGIN)
target_include_directories(MaterializeDimsPlugin PUBLIC ${KERMA_INCLUDE_DIR})
target_link_libraries(MaterializeDimsPlugin KermaNVVMOpt KermaBaseOpt 
                                            KermaSupportOpt 
                                            )
set_target_properties(MaterializeDimsPlugin PROPERTIES COMPILE_FLAGS "-fno-rtti")
# set_property(TARGET MaterializeDimsPlugin PROPERTY POSITION_INDEPENDENT_CODE ON)

##===--------------------------------------------------------------------------
## Build libPassOpt
## Contains all the Kerma passes and can be loaded in opt
##===--------------------------------------------------------------------------

# file(GLOB_RECURSE KermaPassOpt_Sources "*.cpp")
# add_library(KermaPassOpt SHARED ${KermaPassOpt_Sources})
# target_link_libraries(KermaPassOpt PRIVATE KermaBaseOpt)
# set_target_properties(KermaPass PROPERTIES COMPILE_FLAGS "-fno-rtti")

##===--------------------------------------------------------------------------
## Build libPass                                                               
## Contains all the Kerma passes                                               
## This is the library that is meant to be linked by the Kerma tools           
##===--------------------------------------------------------------------------

set(MaterializeDims_Sources MaterializeDims.cpp DetectKernels.cpp)
add_library(MaterializeDims SHARED ${MaterializeDims_Sources})
# target_compile_definitions(MaterializeDims PUBLIC KERMA_OPT_PLUGIN)
target_include_directories(MaterializeDims PRIVATE ${KERMA_INCLUDE_DIR})
target_link_libraries(MaterializeDims KermaBaseOpt KermaSupportOpt)
set_target_properties(MaterializeDims PROPERTIES COMPILE_FLAGS "-fno-rtti")