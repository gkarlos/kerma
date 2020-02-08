message(STATUS "Configuring LLVM")

if ( NOT DEFINED ENV{LLVM_HOME})
  message(FATAL_ERROR "$LLVM_HOME is not defined")
else()
  set(LLVM_HOME $ENV{LLVM_HOME} CACHE PATH "Root of LLVM installation")
  message("   [-] Home: ${LLVM_HOME}")

  if (DEFINED ENV{LLVM_INC})
    set(LLVM_INC $ENV{LLVM_INC})
    message("\tFound $LLVM_INC = ${LLVM_INC}")
  else()
    set(LLVM_INC   ${LLVM_HOME}/include)
    message("\tUsing $LLVM_HOME/include")
  endif()
  
  if (DEFINED ENV{LLVM_LIB})
    set(LLVM_LIB $ENV{LLVM_LIB})
    message("\tFound $LLVM_LIB = ${LLVM_LIB}")
  else()
    set(LLVM_LIB ${LLVM_HOME}/lib)
    message("\tUsing $LLVM_HOME/lib")
  endif()
  
  if (DEFINED ENV{LLVM_BIN})
    set(LLVM_BIN $ENV{LLVM_BIN})
    message("\tFound $LLVM_BIN = ${LLVM_BIN}")
  else()
    set(LLVM_BIN ${LLVM_HOME}/bin)
    message("\tUsing $LLVM_HOME/bin")
  endif()

  set(LLVM_CMAKE ${LLVM_LIB}/cmake/llvm)
  set(LLVM_DIR ${LLVM_CMAKE})

  find_package(LLVM REQUIRED CONFIG PATHS ${LLVM_DIR})
  
  message("   [-] Vers: ${LLVM_PACKAGE_VERSION}")
  message("   [-] Using LLVMConfig.cmake in ${LLVM_DIR}")

  add_definitions(${LLVM_DEFINITIONS})

  include_directories(${LLVM_INC})
  link_directories(${LLVM_LIB})

  set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin/${CMAKE_CFG_INT_DIR}")
  set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/lib/${CMAKE_CFG_INT_DIR}")

  message(STATUS "Configuring LLVM - done")
  
endif()