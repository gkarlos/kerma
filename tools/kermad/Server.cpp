#include "Server.h"
#include "Options.h"
#include "Session.h"
#include "kerma/Analysis/DetectKernels.h"
#include "kerma/Support/Json.h"
#include "kerma/Support/Log.h"

#include <cxxtools/json/rpcserver.h>
#include <cxxtools/log/cxxtools.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <spdlog/fmt/bundled/core.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

#define IN "<--"
#define OUT "-->"

namespace kerma {
namespace kermad {

using namespace cxxtools;
using namespace llvm;

Server *Callback(void *thisPtr) {
  Server *self = static_cast<Server *>(thisPtr);
  return self;
}

#define _ERROR_(f, ...)                                                        \
  {                                                                            \
    Log::error(f, ##__VA_ARGS__);                                              \
    throw std::runtime_error(fmt::format(f, ##__VA_ARGS__));                   \
  }

bool Server::killSession() {
  if (hasActiveSession()) {
    CurrSession.reset();
    RB.clearSession();
    return true;
  }
  return false;
}

bool Server::start() {
  if (!isRunning()) {
    Started = true;
    RpcServer->registerMethod("StartSession", *this, &Server::StartSession);
    RpcServer->registerMethod("StopSession", *this, &Server::StopSession);
    Started = true;
    Loop.run();
    return true;
  }
  return false;
}

bool Server::stop() {
  if (isRunning()) {
    CurrSession.reset();
    Loop.exit();
    return true;
  }
  return false;
}

Server::Server(struct Options &Options)
    : Options(Options), Compiler(Options.ClangExePath), Started(false) {
  RpcServer = std::make_unique<json::RpcServer>(Loop, Options.IP, Options.Port);
}


// RPC handlers impl. and utility functions


static void getDeviceIR(Session &Session, Compiler &Compiler) {
  Log::info("Cmpl. {} -> {}", Session.SourcePath, Session.DeviceIRModuleName);

  if (!Compiler.EmitDeviceIR(Session.SourcePath, Session.DeviceIRModuleName)) {
    Log::error("Compilation failed");
    throw std::runtime_error("Failed to compile LLVM IR");
  }

  SMDiagnostic Err;
  Session.DeviceModule =
      llvm::parseIRFile(Session.getDeviceIRModulePath(), Err, Session.Context);
  if (!Session.DeviceModule)
    _ERROR_("Failed to parse device IR: {}", Session.getDeviceIRModulePath());
}

static void getKernels(Session &Session) {
  Session.Kernels = kerma::getKernels(*Session.DeviceModule);
  for (auto &Kernel : Session.Kernels) {
    Kernel.setSourceRange(Session.SI.getFunctionRange(Kernel.getName()));
    Log::info("Kernel #{}. {} @{}", Kernel.getID(), Kernel.getName(),
              Kernel.getSourceRange().getStart().getLine());
  }
}

static void getSourceInfo(Session &Session) {
  try {
    Session.SI = Session.SIB->getSourceInfo();
  } catch (std::exception &e) {
    Log::error("SourceInfo: {}", e.what());
    throw std::runtime_error("Failed to read src info");
  }
}

void Server::initSession(const std::string &SourceDir,
                         const std::string &Source) {
  CurrSession = std::make_unique<Session>(Options, SourceDir, Source);
  RB.setSession(*CurrSession);

  // order is important
  getDeviceIR(*CurrSession, Compiler);
  getSourceInfo(*CurrSession);
  getKernels(*CurrSession);
  // next: extract assumptions
  // next: canonicalize ir
}

#define SYNC std::lock_guard<std::mutex> Guard(Mutex)

/// Start a new session for a specific file
KermaRes Server::StartSession(const std::string &SourceDir,
                              const std::string &Source,
                              const std::string &CompileDb) {
  SYNC;
  if (hasActiveSession())
    _ERROR_("Busy in another session ({})", CurrSession->getID());

  Log::info("{} StartSession({}, {}) ", IN, SourceDir, Source);

  try {
    initSession(SourceDir, Source);
    auto Res = RB.getForStartSession();
    Log::info("{} StartSession({} kernels, {} device functions)", OUT,
              Res["kernels"].size(), Res["device_functions"].size());
    return Res.dump();
  } catch (...) {
    killSession();
    throw;
  }
}

KermaRes Server::StopSession(bool exit) {
  SYNC;
  Log::info("{} StopSession({})", IN, exit);
  if (!killSession())
    throw "No active session to stop";
  return RB.getForStopSession().dump();
}

} // end namespace kermad
} // end namespace kerma
