set(KermaCompile_SOURCES CompilationAdjuster.cpp)

add_library(KermaCompile SHARED ${KermaCompile_SOURCES})

target_link_libraries(KermaCompile clangTooling)