#
# Build individual libraries for each pass so that
# we can easily run a specific pass with opt
# 

# DetectKernels
set(DetectKernelsPass_Sources DetectKernels.cpp)
add_library(DetectKernelsPass SHARED ${DetectKernelsPass_Sources})
set_target_properties(DetectKernelsPass PROPERTIES COMPILE_FLAGS "-fno-rtti")

# MeterializeBlockThread
set(MaterializeBlockThread_Sources MaterializeBlockThread.cpp)
add_library(MaterializeBlockThread SHARED ${MaterializeBlockThread_Sources})
set_target_properties(MaterializeBlockThread PROPERTIES COMPILE_FLAGS "-fno-rtti")

# LoopInfoTestPass
set(LoopInfoTestPass_Sources LoopInfoTestPass.cpp)
add_library(LoopInfoTestPass SHARED ${LoopInfoTestPass_Sources})
set_target_properties(LoopInfoTestPass PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_compile_options(LoopInfoTestPass PUBLIC "-fPIC")

#
# Build libPass
# 

file(GLOB_RECURSE LibPassSources "*.cpp")

# llvm_map_components_to_libnames(mbt_llvm_deps Demangle)

# add_library(mbt SHARED MaterializeBlockThread.cpp)
# target_link_libraries(mbt ${mbt_llvm_deps})
# set_target_properties(mbt PROPERTIES COMPILE_FLAGS "-fno-rtti")


# target_link_libraries(KermaDetectKernelsPass KermaCuda KermaSupport)

# set(DetectKernels_LLVM_Sources
#         ${CMAKE_CURRENT_SOURCE_DIR}/DetectKernels.cpp
#         PARENT_SCOPE)