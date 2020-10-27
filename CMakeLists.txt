cmake_minimum_required(VERSION 3.12)
project("Kerma")

if ( DEFINED "$ENV{TRAVIS}")
    set(KERMA_CI_BUILD TRUE)
else()
    set(KERMA_CI_BUILD FALSE)
endif()

set(KERMA_VER_MAJOR 0)
set(KERMA_VER_MINOR 1)
set(KERMA_VER_PATCH 0)
set(KERMA_VERSION "${KERMA_VER_MAJOR}.${KERMA_VER_MINOR}.${KERMA_VER_PATCH}")
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

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out/bin)

# we "install" in these directories
set(KERMA_BINARY_OUTPUT_DIRECTORY  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(KERMA_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(KERMA_INCLUDE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out/include)
set(KERMA_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out/rt)
add_custom_target(CreateRTDirectory ALL COMMAND ${CMAKE_COMMAND} -E make_directory ${KERMA_RUNTIME_OUTPUT_DIRECTORY})
set(KERMA_CUDA_ARCH sm_30)

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

# install the headers
file(COPY ${KERMA_INCLUDE_DIR}/kerma DESTINATION ${KERMA_INCLUDE_OUTPUT_DIRECTORY} PATTERN .h)

