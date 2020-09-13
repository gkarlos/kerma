set(SourceInfo_SOURCES SourceLoc.cpp
                       SourceRange.cpp
                       SourceInfo.cpp
                       SourceInfoExtractor.cpp
                       FunctionRangeExtractor.cpp
                       Util.cpp)

add_library(KermaSourceInfo SHARED ${SourceInfo_SOURCES})

add_definitions(${LLVM_CXXFLAGS})
target_include_directories(KermaSourceInfo PRIVATE ${LLVM_LIB}/clang/${CLANG_VERSION_STRING}/include) 
include_directories(${LLVM_INCLUDE_DIRS})

set_target_properties(KermaSourceInfo PROPERTIES COMPILE_FLAGS "-fno-rtti")

target_link_libraries(KermaSourceInfo PRIVATE
    clangFrontend
    clangSerialization
    clangDriver
    clangParse
    clangSema
    clangAnalysis
    clangAST
    clangBasic
    clangEdit
    clangLex
    clangTooling
    LLVMCore
    LLVMFrontendOpenMP
    LLVMIRReader
    LLVMSupport
    LLVMMC
    LLVMMCParser
    LLVMOption
    LLVMProfileData
)

target_link_libraries(KermaSourceInfo PRIVATE KermaCompile KermaSupport)

