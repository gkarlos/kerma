set(DetectKernels_SOURCES DetectKernels.cpp)

##===--------------------------------------------------------------------------
## Build standalone Opt plugins for each pass so tha we can easily test a 
## specific pass in opt. It is important that these libraries are not linked 
## with any LLVM libraries to avoid duplicate cmdline opts errors in Opt
##===--------------------------------------------------------------------------

##===--------------------------------------------------------------------------
## DetectKernels
##===--------------------------------------------------------------------------
add_library(DetectKernelsPlugin SHARED ${DetectKernels_SOURCES})
target_include_directories(DetectKernelsPlugin PRIVATE ${KERMA_INCLUDE_DIR})
set_target_properties(DetectKernelsPlugin PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(DetectKernelsPlugin KermaSupportOpt)