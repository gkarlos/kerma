file(GLOB_RECURSE KermaCuda_SOURCES "*.cpp")

add_library(KermaCudaOpt SHARED ${KermaCuda_SOURCES})
target_include_directories(KermaCudaOpt PRIVATE ${KERMA_INCLUDE_DIR})
set_target_properties(KermaCudaOpt PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(KermaCudaOpt KermaSupportOpt)

add_library(KermaCuda SHARED ${KermaCuda_SOURCES})
target_include_directories(KermaCuda PRIVATE ${KERMA_INCLUDE_DIR})
set_target_properties(KermaCuda PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(KermaCuda KermaSupport)