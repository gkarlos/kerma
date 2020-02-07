if ( NOT DEFINED ENV{LLVM_HOME})
  message(FATAL_ERROR "$LLVM_HOME is not defined")
else()
  set(LLVM_HOME $ENV{LLVM_HOME} CACHE PATH "Root of LLVM install")

  set(LLVM_INC   ${LLVM_HOME}/include)
  set(LLVM_LIB   ${LLVM_HOME}/lib)
  set(LLVM_BIN   ${LLVM_HOME}/bin)
  set(LLVM_CMAKE ${LLVM_LIB}/cmake/llvm)
  set(LLVM_DIR ${LLVM_CMAKE})

  find_package(LLVM REQUIRED CONFIG)
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "LLVM Directories: ")
  message("    *  HOME - " ${LLVM_HOME})
  message("    *   INC - " ${LLVM_INC})
  message("    *   LIB - " ${LLVM_LIB})
  message("    * CMAKE - " ${LLVM_CMAKE})

  message(STATUS "Found LLVM defs ${LLVM_DEFINITIONS}")
  add_definitions(${LLVM_DEFINITIONS})

  include_directories(${LLVM_INC})
  link_directories(${LLVM_LIB})

  set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin/${CMAKE_CFG_INT_DIR}")
  set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/lib/${CMAKE_CFG_INT_DIR}")
endif()