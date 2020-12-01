set(KermaRTUtils_SOURCES Util.cpp)
set(KermaRT KermaRT.cpp)


##===--------------------------------------------------------------------------
## KermaRTUtilsOpt
##===--------------------------------------------------------------------------
add_library(KermaRTUtilsOpt SHARED ${KermaRTUtils_SOURCES})
target_compile_definitions(KermaRTUtilsOpt PUBLIC -DLLVM_ENABLE_STATS KERMA_OPT_PLUGIN)
target_include_directories(KermaRTUtilsOpt PUBLIC ${KERMA_INCLUDE_DIR})
set_target_properties(KermaRTUtilsOpt PROPERTIES COMPILE_FLAGS "-fno-rtti")

##===--------------------------------------------------------------------------
## KermaRTUtils
##===--------------------------------------------------------------------------
add_library(KermaRTUtils SHARED ${KermaRTUtils_SOURCES})
target_compile_definitions(KermaRTUtils PUBLIC -DLLVM_ENABLE_STATS KERMA_OPT_PLUGIN)
target_include_directories(KermaRTUtils PUBLIC ${KERMA_INCLUDE_DIR})
set_target_properties(KermaRTUtils PROPERTIES COMPILE_FLAGS "-fno-rtti")




if (${KERMA_CI_BUILD})
    message("[+] Skipping DeviceRT and MetaDriver in TRAVIS build")
else()
##===--------------------------------------------------------------------------
## DeviceRT
##===--------------------------------------------------------------------------
    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libKermaDeviceRT.bc"
        COMMAND $ENV{LLVM_HOME}/bin/clang++ -S -x cuda --cuda-device-only -emit-llvm --cuda-gpu-arch=${KERMA_CUDA_ARCH} -o libKermaDeviceRT.bc -std=c++11 "${CMAKE_CURRENT_SOURCE_DIR}/DeviceRT.cpp"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        DEPENDS DeviceRT.cpp
        VERBATIM
        )
    add_custom_target(KermaRT ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/libKermaDeviceRT.bc")

    ## Copy to both lib and out
    add_custom_command(TARGET KermaRT COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/libKermaDeviceRT.bc ${KERMA_LIBRARY_OUTPUT_DIRECTORY})
    add_custom_command(TARGET KermaRT COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/libKermaDeviceRT.bc ${KERMA_RUNTIME_OUTPUT_DIRECTORY})

##===--------------------------------------------------------------------------
## MetaDriver
##===--------------------------------------------------------------------------
    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/MetaDriver.ll"
        COMMAND $ENV{LLVM_HOME}/bin/clang++ -S -O0 -c -emit-llvm -o MetaDriver.ll -std=c++11  "${CMAKE_CURRENT_SOURCE_DIR}/MetaDriver.cpp"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        DEPENDS MetaDriver.cpp
        VERBATIM
    )
    add_custom_target(MetaDriver ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/MetaDriver.ll")
    add_custom_command(TARGET MetaDriver COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/MetaDriver.ll ${KERMA_RUNTIME_OUTPUT_DIRECTORY})
endif()
