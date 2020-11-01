message(STATUS "Configuring spdlog")

include(FetchContent)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.8.1
)

FetchContent_GetProperties(spdlog)
if(NOT spdlog_POPULATED)
  FetchContent_Populate(spdlog)
  add_subdirectory(${spdlog_SOURCE_DIR} ${spdlog_BINARY_DIR})
endif()

set(SPDLOG_INCLUDE_DIR ${spdlog_SOURCE_DIR}/include)
include_directories(${SPDLOG_INCLUDE_DIR})
message(STATUS "Configuring spdlog -- done")
