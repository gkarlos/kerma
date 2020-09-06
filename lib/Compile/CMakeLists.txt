set(KermaCompile_SOURCES CompilationAdjuster.cpp
                         DiagnosticConsumers.cpp)

add_library(KermaCompile SHARED ${KermaCompile_SOURCES})
set_target_properties(KermaCompile PROPERTIES COMPILE_FLAGS "-fno-rtti")

target_link_libraries(KermaCompile clangTooling clangBasic)