set(KermaCompile_SOURCES CompileDBAdjuster.cpp)

add_library(KermaCompile SHARED ${KermaCompile_SOURCES})

target_link_libraries(KermaCompile clangTooling)