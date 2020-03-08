cmake_minimum_required(VERSION 3.11)
project("kerma-static-analysis")

# set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(ENABLE_TESTS ON)
set(BUILD_EXAMPLES OFF)

include(cmake/llvm.cmake)
include(cmake/boost.cmake)
include(cmake/gtest.cmake)

set(KERMA_INCLUDE_DIRS include)

include_directories(${KERMA_INCLUDE_DIRS})

add_subdirectory(src)


if(ENABLE_TESTS)
    set(BUILD_EXAMPLES ON)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(ENABLE_TESTS)
    message(STATUS "Building Tests")
    enable_testing()
    add_subdirectory(test)
endif()

