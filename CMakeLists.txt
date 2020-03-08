cmake_minimum_required(VERSION 3.11)
project("Kerma")

set(KERMA_MAJOR_VERSION 1)
set(KERMA_MINOR_VERSION 0)
set(KERMA_VERSION ${KERMA_MAJOR_VERSION}.${KERMA_MINOR_VERSION})

# set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(ENABLE_TESTS ON)
set(BUILD_EXAMPLES OFF)
set(BUILD_DOCS ON)

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
    enable_testing()
    add_subdirectory(test)
endif()

if(BUILD_DOCS)
    add_subdirectory(docs)
endif()

