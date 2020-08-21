file(GLOB_RECURSE SupportSources "*.cpp")

# add_library(KermaSupport SHARED ${SupportSources})

# set_target_properties(KermaSupport PROPERTIES COMPILE_FLAGS "-fno-rtti")

# add_definitions(${LLVM_CXXFLAGS})
# target_compile_options(KermaSupport PUBLIC "-fPIC")
# include_directories(${LLVM_LIB}/clang/${CLANG_VERSION_STRING}/include) 
# include_directories(${LLVM_INCLUDE_DIRS})

# llvm_map_components_to_libnames(KermaSupportLLVMLibs 
#     Core
#     IRReader
#     Support
#     Option
# )

# target_link_libraries(KermaSupport
#     clangFrontend
#     clangSerialization
#     clangDriver
#     clangParse
#     clangSema
#     clangAnalysis
#     clangAST
#     clangBasic
#     clangEdit
#     clangLex
#     clangTooling
# )

# target_link_libraries(KermaSupport
#     LLVMCore
#     LLVMSupport
#     LLVMIRReader
#     LLVMOption
#     LLVMMC
#     LLVMMCParser
#     LLVMProfileData
# )

# target_link_libraries(KermaSupport 
#     stdc++fs
# )

add_library(KermaSupportOpt SHARED Demangle.cpp FileSystem.cpp Version.cpp)
set_target_properties(KermaSupportOpt PROPERTIES COMPILE_FLAGS "-fno-rtti")
# set_property(TARGET KermaSupportOpt PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(KermaSupportOpt LLVMDemangle)

