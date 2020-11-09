message(STATUS "Configuring nlohmann/json")
include(FetchContent)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.9.1
)

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
  add_custom_target(exclude_json_tests ALL
    COMMAND rm -f "${json_BINARY_DIR}/CTestTestfile.cmake")
endif()

set(json_INCLUDE_DIR ${json_SOURCE_DIR}/include)
include_directories(${json_INCLUDE_DIR})
message(${json_INCLUDE_DIR})
message(STATUS "Configuring nlohmann/json -- done")
