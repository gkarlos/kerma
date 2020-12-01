#ifndef KERMA_COMPILE_COMPILER_H
#define KERMA_COMPILE_COMPILER_H

// This header is needed because DiagnosticIDs
// has some forward declarations from it
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/DiagnosticIDs.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/FileManager.h>
#include <clang/Basic/LLVM.h>
#include <clang/Driver/Driver.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <llvm/Support/Path.h>
#include <memory>

namespace kerma {

class Compiler {
private:
  std::string ClangPath;
  clang::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOptions;
  clang::TextDiagnosticPrinter DiagPrinter;
  clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagIDs;
  clang::DiagnosticsEngine Diags;
  clang::driver::Driver Driver;

public:
  static const std::string DefaultDeviceIRFile;
  static const std::string DefaultHostIRFile;
  static const std::string DefaultDeviceBCFile;
  static const std::string DefaultHostBCFile;
  static const std::string DefaultPTXFile;
  static const std::string DefaultDeviceObjFile;
  static const std::string DefaultFatbinFile;
  static const std::string DefaultHostCuiFile;
  static const std::string DefaultLinkedHostIR;

  Compiler(const std::string &ClangPath);

  /// Generate device side LLVM IR.
  /// On success a file named \p Out will be created in the current dir
  /// @returns true on success. false otherwise
  bool EmitDeviceIR(const std::string &SourcePath,
                    const std::string &Out = DefaultDeviceIRFile);
  bool EmitHostIR(const std::string &SourcePath,
                  const std::string &Out = DefaultHostIRFile);
  bool EmitDeviceBC(const std::string &SourcePath,
                    const std::string &Out = DefaultDeviceBCFile);
  bool EmitHostBC(const std::string &SourcePath,
                  const std::string &Out = DefaultHostBCFile);
  bool EmitPTX(const std::string &SourcePath,
               const std::string &Out = DefaultPTXFile);
  bool EmitDeviceObj(const std::string &SourcePath,
                     const std::string &Out = DefaultDeviceObjFile);
  bool EmitFatbin(const std::string &SourcePathDeviceObj,
                  const std::string &SourcePathPtx,
                  const std::string &Out = DefaultFatbinFile);
};

} // namespace kerma

#endif // KERMA_COMPILE_COMPILER_H