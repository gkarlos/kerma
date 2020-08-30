set(DetectKernels_SOURCES DetectKernels.cpp)
set(MaterializeDims_SOURCES MaterializeDims.cpp)
set(MaterializeIdx_SOURCES MaterializeDims.cpp)
##===--------------------------------------------------------------------------
## Build standalone Opt plugins for each pass so that
## we can easily test a specific pass in opt
## 
## It is important that these libraries are not linked with any LLVM
## libraries to avoid duplicate cmdline opts errors as Opt is already
## linked with LLVM
##===--------------------------------------------------------------------------

# DetectKernels
add_library(DetectKernelsPlugin SHARED ${DetectKernels_SOURCES})
target_include_directories(DetectKernelsPlugin PRIVATE ${KERMA_INCLUDE_DIR})
set_target_properties(DetectKernelsPlugin PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(DetectKernelsPlugin KermaSupportOpt)
# set_property(TARGET DetectKernelsPlugin PROPERTY POSITION_INDEPENDENT_CODE ON)

#MaterializeDims
add_library(MaterializeDimsPlugin SHARED ${MaterializeDims_SOURCES})
target_compile_definitions(MaterializeDimsPlugin PUBLIC -DLLVM_ENABLE_STATS KERMA_OPT_PLUGIN)
target_include_directories(MaterializeDimsPlugin PUBLIC ${KERMA_INCLUDE_DIR})
set_target_properties(MaterializeDimsPlugin PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(MaterializeDimsPlugin DetectKernelsPlugin
                                            KermaNVVMOpt 
                                            KermaBaseOpt 
                                            KermaSupportOpt 
                                            )

#MaterializeIdx
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

add_library(MaterializeDims SHARED ${MaterializeDims_SOURCES})
# target_compile_definitions(MaterializeDims PUBLIC KERMA_OPT_PLUGIN)
target_include_directories(MaterializeDims PRIVATE ${KERMA_INCLUDE_DIR})
target_link_libraries(MaterializeDims KermaBaseOpt KermaSupportOpt)
set_target_properties(MaterializeDims PROPERTIES COMPILE_FLAGS "-fno-rtti")