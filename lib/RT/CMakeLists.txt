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

##
## TODO Create .bc of DeviceRT
##
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libKermaDeviceRT.bc"
    COMMAND $ENV{LLVM_HOME}/bin/clang++ -S -x cuda --cuda-device-only -emit-llvm --cuda-gpu-arch=sm_52 -o libKermaDeviceRT.bc -std=c++11 "${CMAKE_CURRENT_SOURCE_DIR}/DeviceRT.cpp"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS DeviceRT.cpp
    VERBATIM
    )
add_custom_target(KermaRT ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/libKermaDeviceRT.bc")

##
## TODO Create .bc of DeviceRTMemTrace
##
# add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libKermaDeviceRTCuMemtrace.bc"
#     COMMAND $ENV{LLVM_HOME}/bin/clang++ -S -x cuda --cuda-device-only -emit-llvm --cuda-gpu-arch=sm_52 -o libKermaDeviceRTCuMemtrace.bc -std=c++11 "${CMAKE_CURRENT_SOURCE_DIR}/DeviceRTCuMemtrace.cpp"
#     WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
#     DEPENDS DeviceRTCuMemtrace.cpp
#     VERBATIM
#     )
# add_custom_target(CuMemtraceRT ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/libKermaDeviceRTCuMemtrace.bc")