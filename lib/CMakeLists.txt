add_subdirectory(Support)
# add_subdirectory(Utils)
# add_subdirectory(Base)
# add_subdirectory(Compile)
# add_subdirectory(NVVM)
# add_subdirectory(Analysis)
add_subdirectory(RT)
# add_subdirectory(Transforms)
# add_subdirectory(SourceInfo)


set(Kerma_SOURCES
    Analysis/DetectAssumptions.cpp
    Analysis/DetectKernels.cpp
    Analysis/DetectMemories.cpp
    Analysis/DetectMemoryAccesses.cpp
    Analysis/InferAddressSpaces.cpp
    Analysis/Loops.cpp
    Analysis/MemoryAccessTree.cpp
    Analysis/MemoryAccessTreeBuilder.cpp
    Analysis/Names.cpp
    Analysis/Nests.cpp
    Analysis/Typecheck.cpp

    Base/Assumption.cpp
    Base/Dim.cpp
    Base/If.cpp
    Base/Index.cpp
    Base/Kernel.cpp
    Base/Loop.cpp
    Base/Memory.cpp
    Base/MemoryAccess.cpp
    Base/Mode.cpp
    Base/Node.cpp
    Base/Stmt.cpp

    Compile/CompilationAdjuster.cpp
    Compile/Compiler.cpp
    Compile/DiagnosticConsumers.cpp

    NVVM/NVVM.cpp
    NVVM/NVVMUtilities.cpp

    RT/Util.cpp

    SourceInfo/SourceInfo.cpp
    SourceInfo/SourceInfoAction.cpp
    SourceInfo/SourceInfoBuilder.cpp
    SourceInfo/SourceLoc.cpp
    SourceInfo/SourceRange.cpp
    SourceInfo/Util.cpp

    Support/Demangle.cpp
    Support/FileSystem.cpp
    Support/FileUtils.cpp
    Support/Parse.cpp
    Support/Version.cpp

    Transforms/Canonicalize/BreakConstantGEP.cpp
    Transforms/Canonicalize/Canonicalizer.cpp
    Transforms/Canonicalize/DeviceFunctionInliner.cpp
    Transforms/Canonicalize/GepifyMem.cpp
    Transforms/Canonicalize/SimplifyGEP.cpp
    Transforms/Canonicalize/StripPrintf.cpp
    Transforms/Instrument/MatInstrumenter.cpp
    Transforms/Instrument/InstrumentPrintf.cpp
    Transforms/Instrument/LinkDeviceRT.cpp
    Transforms/Materializer.cpp
    Transforms/MetaKernels.cpp
    Transforms/PrepareMetaDriver.cpp
    Transforms/StripAnnotations.cpp
    # Transforms/MaterializeDims.cpp
    # Transforms/MaterializeIdx.cpp

    Utils/LLVMMetadata.cpp
    Utils/LLVMShorthands.cpp
 )

add_library(Kerma SHARED ${Kerma_SOURCES})
target_link_libraries(Kerma  ${Boost_LIBRARIES}
                             LLVM
                             LLVMTransformUtils
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
                             stdc++fs)

set(KermaOpt_SOURCES
    Analysis/DetectKernels.cpp
    Analysis/InferAddressSpaces.cpp
    Analysis/Names.cpp
    Analysis/Typecheck.cpp

    Base/Assumption.cpp
    Base/Dim.cpp
    Base/Index.cpp
    Base/Kernel.cpp
    Base/Memory.cpp
    Base/MemoryAccess.cpp

    # Compile/CompilationAdjuster.cpp
    # Compile/Compiler.cpp
    # Compile/DiagnosticConsumers.cpp

    NVVM/NVVM.cpp
    NVVM/NVVMUtilities.cpp

    RT/Util.cpp

    # SourceInfo/FunctionRangeExtractor.cpp
    # SourceInfo/SourceInfo.cpp
    # SourceInfo/SourceInfoExtractor.cpp
    SourceInfo/SourceLoc.cpp
    SourceInfo/SourceRange.cpp
    # SourceInfo/Util.cpp

    Support/Demangle.cpp
    Support/FileSystem.cpp
    Support/FileUtils.cpp
    Support/Parse.cpp
    Support/Version.cpp

    Transforms/Canonicalize/BreakConstantGEP.cpp
    Transforms/Canonicalize/Canonicalizer.cpp
    Transforms/Canonicalize/DeviceFunctionInliner.cpp
    Transforms/Canonicalize/GepifyMem.cpp
    Transforms/Canonicalize/SimplifyGEP.cpp
    Transforms/Instrument/InstrumentPrintf.cpp
    Transforms/Instrument/LinkDeviceRT.cpp
    # Transforms/MaterializeDims.cpp
    # Transforms/MaterializeIdx.cpp

    Utils/LLVMMetadata.cpp
    Utils/LLVMShorthands.cpp
 )

 add_library(KermaOpt SHARED ${KermaOpt_SOURCES})
 target_compile_definitions(KermaOpt PUBLIC KERMA_OPT_PLUGIN)
 target_link_libraries(KermaOpt ${Boost_LIBRARIES})