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


#define DEVICE_IR "device.ll"
#define HOST_IR "host.ll"

namespace kerma {

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

  /// Generate device side LLVM IR.
  /// On success a file named \p Out will be created in the current dir
  /// @returns true on success. false otherwise
  bool EmitDeviceIR(const std::string& SourcePath, const std::string& Out=DEVICE_IR);
  bool EmitHostIR(const std::string& SourcePath, const std::string& Out=HOST_IR);
};

} // namespace kerma

#endif // KERMA_COMPILE_COMPILER_H