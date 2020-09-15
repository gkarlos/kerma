cmake_minimum_required(VERSION 3.12)
project("Kerma")

set(KERMA_MAJOR_VERSION 0)
set(KERMA_MINOR_VERSION 1)
set(KERMA_VERSION ${KERMA_MAJOR_VERSION}.${KERMA_MINOR_VERSION})
set(KERMA_HOME ${CMAKE_CURRENT_SOURCE_DIR}/build)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(ENABLE_TESTS ON)
set(BUILD_EXAMPLES ON)
set(BUILD_DOCS ON)
set(KERMA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

include(cmake/llvm.cmake)
include(cmake/boost.cmake)
include(cmake/gtest.cmake)

set(CMAKE_C_COMPILER ${LLVM_BIN}/clang)
set(CMAKE_CXX_COMPILER ${LLVM_BIN}/clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

## Config
## TODO Move to separete .cmake file
if(${ENABLE_TESTS})
    set(BUILD_EXAMPLES ON)
endif()
set(KERMA_TESTS_ENABLED ${ENABLE_TESTS})
set(KERMA_EXAMPLES_ENABLED ${BUILD_EXAMPLES})

include_directories(${KERMA_INCLUDE_DIR})

add_subdirectory(lib)
add_subdirectory(tools)

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

