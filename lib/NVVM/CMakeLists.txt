file(GLOB_RECURSE KermaNVVM_SOURCES "*.cpp")

add_library(KermaNVVMOpt SHARED ${KermaNVVM_SOURCES})
target_include_directories(KermaNVVMOpt PRIVATE ${KERMA_INCLUDE_DIR})
set_target_properties(KermaNVVMOpt PROPERTIES COMPILE_FLAGS "-fno-rtti")

add_library(KermaNVVM SHARED ${KermaNVVM_SOURCES})
target_include_directories(KermaNVVM PRIVATE ${KERMA_INCLUDE_DIR})
set_target_properties(KermaNVVM PROPERTIES COMPILE_FLAGS "-fno-rtti")