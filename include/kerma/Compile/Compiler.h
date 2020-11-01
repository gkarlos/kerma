#ifndef KERMA_COMPILE_COMPILER_H
#define KERMA_COMPILE_COMPILER_H

// This header is needed because DiagnosticIDs
// has some forward declarations from it
#include <clang/Basic/FileManager.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/DiagnosticIDs.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/LLVM.h>
#include <clang/Driver/Driver.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <llvm/Support/Path.h>
#include <memory>

namespace kerma {

#define DEVICE_IR "device.ll"
#define HOST_IR "host.ll"

class Compiler {
private:
  std::string ClangPath;
  clang::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOptions;
  clang::TextDiagnosticPrinter DiagPrinter;
  clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagID;
  clang::DiagnosticsEngine Diags;
  clang::driver::Driver Driver;
public:
  Compiler(const std::string& ClangPath);

  bool getDeviceIR(const std::string& SourcePath);
};

} // namespace kerma

#endif // KERMA_COMPILE_COMPILER_H