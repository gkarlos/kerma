#include "kerma/Compile/DiagnosticConsumers.h"

#include <iostream>

namespace kerma {

using namespace clang;

void ErrorCountConsumer::HandleDiagnostic( DiagnosticsEngine::Level DiagLevel, const Diagnostic& Info) {
  if ( DiagLevel == DiagnosticsEngine::Level::Error ||
        DiagLevel == DiagnosticsEngine::Level::Fatal ) {
    NumErrors++;
  }
}

} // namespace kerma

