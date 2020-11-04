
#include <cxxtools/json/rpcserver.h>
#include <cxxtools/log/cxxtools.h>
#include <llvm-10/llvm/Support/SourceMgr.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_ostream.h>
#include <spdlog/spdlog.h>
#include <memory>
#include <stdexcept>
#include "kerma/Analysis/DetectKernels.h"
#include "kerma/Support/Json.h"
#include "kerma/Support/Log.h"
#include "Options.h"
#include "Server.h"
#include "Session.h"
#include "spdlog/fmt/bundled/core.h"




#define IN "<--"
#define OUT "-->"

namespace kerma {
namespace kermad {

using namespace cxxtools;
using namespace llvm;
// using namespace cxxtools::json;

Server *  Callback(void *thisPtr) {
  Server * self = static_cast<Server*>(thisPtr);
  return self;
}

#define _ERROR_(f, ...) {                                     \
  Log::error(f, ##__VA_ARGS__);                               \
  throw std::runtime_error(fmt::format(f, ##__VA_ARGS__));    \
}

bool Server::killSession() {
  if ( hasActiveSession()) {
    CurrSession.reset();
    return true;
  }
  return false;
}


void Server::initSession(const std::string& SourceDir, const std::string& Source) {
  CurrSession = std::make_unique<Session>(Options, SourceDir, Source);
  // SIExtractor = std::make_unique<SourceInfoExtractor>(CurrSession->getSourcePath());
  SIB = std::make_unique<SourceInfoBuilder>(CurrSession->getSourcePath());

  // Generate device IR
  Log::info("Cmpl. {} -> {}", CurrSession->getSourcePath(), CurrSession->DeviceIRModuleName);

  if ( !Compiler.EmitDeviceIR(CurrSession->getSourcePath(),
                              CurrSession->DeviceIRModuleName)) {
    Log::error("Compilation failed");
    throw std::runtime_error("Failed to compile LLVM IR");
    killSession();
  }

  SMDiagnostic Err;
  CurrSession->DeviceModule = llvm::parseIRFile(CurrSession->getDeviceIRModulePath(),
                                                Err, CurrSession->Context);

  if ( !CurrSession->DeviceModule)
    _ERROR_("Failed to parse device IR: {}", CurrSession->getDeviceIRModulePath());

  CurrSession->Kernels = getKernels(*CurrSession->DeviceModule);
  CurrSession->SI = SIB->getSourceInfo();
  for ( auto &Kernel : CurrSession->Kernels) {
    Kernel.setSourceRange(CurrSession->SI.getFunctionRange(Kernel.getName()));
    Log::info("Kernel #{}. {} @{}", Kernel.getID(), Kernel.getName(),
                                   Kernel.getSourceRange().getStart().getLine());
  }
}


/// Start a new session for a specific file
KermaRes
Server::StartSession(const std::string& SourceDir, const std::string& Source,
                     const std::string& CompileDb) {
  if ( hasActiveSession())
    _ERROR_("Busy in another session ({})", CurrSession->getID());

  Log::info("{} StartSession({}, {}) ", IN, SourceDir, Source);

  initSession(SourceDir, Source);

  Json Res;
  Res["kernels"] = Json::array();
  for ( auto& Kernel : CurrSession->Kernels) {
    auto Range = Kernel.getSourceRange();
    Res["kernels"].push_back({
      {"name", Kernel.getName()},
      {"id", Kernel.getID()},
      {"range", {Range.getStart().getLine(),
                 Range.getStart().getCol(),
                 Range.getEnd().getLine(),
                 Range.getEnd().getCol()}}
    });
  }

  Log::info("{} StartSession({} kernels)", OUT, Res["kernels"].size());
  return Res.dump();
}

KermaRes
Server::StopSession(bool exit) {
  if ( !hasActiveSession())
    throw "No active session to stop";

  Log::info("{} StopSession({})", IN, exit);
  CurrSession.reset();
  Json Res;
  Res["status"] = "success";
  return Res.dump();
}

bool Server::start() {
  if ( !isRunning()) {
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
  if ( isRunning()) {
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

} // end namespace kermad
} // end namespace kerma
