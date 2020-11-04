#include "kerma/SourceInfo/SourceInfoBuilder.h"

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

  SourceInfoBuilder SIB(argv[1]);

  // SourceInfoLocator SIE()

  std::string Target = "bpnn_layerforward_CUDA";

  // auto ranges = SIE.getFunctionRange(Target);

  auto SI = SIB.getSourceInfo();
  auto Range = SI.getFunctionRange(Target);

  std::cout << Target << '\n';
  std::cout << " \\_ " << Range << '\n';

  return 0;
}