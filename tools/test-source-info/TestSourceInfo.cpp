#include "kerma/SourceInfo/SourceInfoExtractor.h"


#include <iostream>

using namespace clang;
using namespace kerma;

/*
  - LLVM location taken from $LLVM_HOME
  - We assume Clang is under $LLVM_HOME
      * We need to add $LLVM_HOME/lib/clang/10.0.0/include as an include path
        because clang tools apparently dont have it (unlike the clang binary)
*/

static std::string ERR_READ_COMPILE_DB = "Failed to read compile_commands.json";

int main(int argc, const char** argv) {

  if ( argc < 2) {
    std::cout << "usage: ./test-source-info <BuildDir>\n";
    return 0;
  }

  // look for compile_commands.json
  // auto CompileDB = tooling::CompilationDatabase::loadFromDirectory(argv[1], ERR_READ_COMPILE_DB);

  // CompileDBAdjuster::appendClangIncludes(*CompileDB);

  SourceInfoExtractor SIE(argv[1]);

  std::string Target = "mm2_kernel1";

  auto ranges = SIE.getFunctionRange(Target);

  if ( ranges.empty())
    std::cout << "No ranges found for " << Target << '\n';
  else
    for ( auto& entry : ranges) {
      std::cout << entry.first << '\n';
      for ( auto& range : entry.second)
        std::cout << " \\_ " << range << '\n';
    }


  // std::cout << "Running tool on: " << SIE.getSource() << '\n';
  // for ( auto CC : CompileDB->getAllCompileCommands()) {

  //   for ( auto arg : CC.CommandLine) {
  //     std::cout << arg << "\n";
  //   }
  // }

  // SIE.getFunctionRanges("asd");

  return 0;
}