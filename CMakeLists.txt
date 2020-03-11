cmake_minimum_required(VERSION 3.12)
project("Kerma")

set(KERMA_MAJOR_VERSION 0)
set(KERMA_MINOR_VERSION 1)
set(KERMA_VERSION ${KERMA_MAJOR_VERSION}.${KERMA_MINOR_VERSION})

# set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(ENABLE_TESTS ON)
set(BUILD_EXAMPLES OFF)
set(BUILD_DOCS ON)
set(KERMA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

include(cmake/llvm.cmake)
include(cmake/boost.cmake)
include(cmake/gtest.cmake)

## Config
set(KERMA_TESTS_ENABLED ${ENABLE_TESTS})
if(${KERMA_TESTS_ENABLED})
    set(BUILD_EXAMPLES ON)
endif()

configure_file(
    ${KERMA_INCLUDE_DIR}/kerma/Config/version.h.in
    ${KERMA_INCLUDE_DIR}/kerma/Config/version.h
)

configure_file(
    ${KERMA_INCLUDE_DIR}/kerma/Config/config.h.in
    ${KERMA_INCLUDE_DIR}/kerma/Config/config.h
)
##-- 

include_directories(${KERMA_INCLUDE_DIR})

add_subdirectory(src)

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

