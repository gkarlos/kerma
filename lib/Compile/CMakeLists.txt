set(KermaCompile_SOURCES Compiler.cpp
                         CompilationAdjuster.cpp
                         DiagnosticConsumers.cpp)

add_library(KermaCompile SHARED ${KermaCompile_SOURCES})
# set_target_properties(KermaCompile PROPERTIES COMPILE_FLAGS "-fno-rtti")

target_link_libraries(KermaCompile PUBLIC ${Boost_LIBRARIES} clangBasic clangDriver)