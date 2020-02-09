cmake_minimum_required(VERSION 3.2)
project("kerma-static-analysis")

# set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/llvm.cmake)

set(KERMA_INCLUDE_DIRS include)

include_directories(${KERMA_INCLUDE_DIRS})

add_subdirectory(src)
# if ( NOT EXISTS ${LLVM})

# add_subdirectory(test)