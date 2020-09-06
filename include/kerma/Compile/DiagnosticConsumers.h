#ifndef KERMA_COMPILE_DIAGNOSTIC_CONSUMERS_H
#define KERMA_COMPILE_DIAGNOSTIC_CONSUMERS_H

#include "clang/Basic/Diagnostic.h"

namespace kerma {

/// This consumer ignores warnings and only counts errors
class ErrorCountConsumer : public clang::DiagnosticConsumer {
public:
  virtual void HandleDiagnostic(clang::DiagnosticsEngine::Level DiagLevel, const clang::Diagnostic &Info) override;
};

} // namespace kerma

#endif // KERMA_COMPILE_DIAGNOSTICS_CONSUMERS_H