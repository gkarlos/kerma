#include "Server.h"

#include <cxxtools/json/rpcserver.h>
#include <cxxtools/log/cxxtools.h>

#include <llvm/Support/raw_ostream.h>

#include <spdlog/spdlog.h>

#include "Session.h"
#include "kerma/SourceInfo/SourceInfoExtractor.h"
#include "kerma/Support/Json.h"
#include "kerma/Support/Log.h"

#include "Options.h"
#include "spdlog/fmt/bundled/core.h"

#include <memory>

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

/// Start a new session for a specific file
KermaRes
Server::StartSession(const std::string& SourceDir, const std::string& Source, const std::string& CompileDb) {
  if ( hasActiveSession())
    throw fmt::format("Busy in another session ({})", CurrSession->getID());

  Log::info("{} StartSession({}, {}) ", IN, SourceDir, Source);

  CurrSession = std::make_unique<Session>(Options, SourceDir, Source);
  SIExtractor = std::make_unique<SourceInfoExtractor>(CurrSession->getSourcePath());

  if ( Compiler.getDeviceIR(CurrSession->getSourcePath()) ) {
    // TODO: set Session.DeviceIR
  }

  Json Res;
  Res["kernels"] = {
    {
      {"name", "my_kernel_1"},
      {"id", 0},
      {"range", {50, 0, 61, 0}}
    },
    {
      {"name", "my_kernel_2"},
      {"id", 1},
      {"range", {70, 0, 81, 0}}
    }
  };
  Log::info("{} {} kernels", OUT, Res["kernels"].size());
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
