#include "Server.h"
#include "Options.h"
#include "Session.h"
#include "kerma/Analysis/DetectKernels.h"
#include "kerma/Analysis/DetectMemories.h"
#include "kerma/Support/Json.h"
#include "kerma/Support/Log.h"
#include "kerma/Transforms/Canonicalize/Canonicalizer.h"

#include <boost/filesystem/path.hpp>
#include <cxxtools/json/rpcserver.h>
#include <cxxtools/log/cxxtools.h>
#include <llvm-10/llvm/IR/Function.h>
#include <llvm-10/llvm/IR/LegacyPassManager.h>
#include <llvm-10/llvm/Transforms/Utils.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <spdlog/fmt/bundled/core.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <system_error>

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

  // Compiler source to an IR file
  if (!Compiler.EmitDeviceIR(Session.SourcePath, Session.DeviceIRModuleName)) {
    Log::error("Compilation failed");
    throw std::runtime_error("Failed to compile LLVM IR");
  }

  // Read the IR file into memory
  SMDiagnostic Err;
  Session.DeviceModule =
      llvm::parseIRFile(Session.getDeviceIRModulePath(), Err, Session.Context);
  if (!Session.DeviceModule)
    _ERROR_("Failed to parse device IR: {}", Session.getDeviceIRModulePath());

  // Canonicalize the IR
  // for now run Mem2Reg here but we should do something better
  llvm::legacy::PassManager PM;
  PM.add(createPromoteMemoryToRegisterPass());
  PM.run(*Session.DeviceModule);

  CanonicalizerPass Canonicalize;
  Log::info("Cano. {}/{} -> {}", Session.WorkingDir, Session.DeviceIRModuleName,
            Session.DeviceIRCanonModuleName);
  Canonicalize.runOnModule(*Session.DeviceModule);

  // Write canon. IR to a file
  Session.DeviceIRCanonModule =
      (fs::path(Session.WorkingDir) / fs::path(Session.DeviceIRCanonModuleName))
          .string();
  std::error_code err;
  llvm::raw_fd_ostream F(Session.DeviceIRCanonModule, err);
  if (err)
    Log::error("Failed to create {}. ({})", Session.DeviceIRCanonModuleName,
               err.message());
  else
    Session.DeviceModule->print(F, nullptr);
}

static void getSourceInfo(Session &Session) {
  try {
    Session.SI = Session.SIB->getSourceInfo();
  } catch (std::exception &e) {
    Log::error("SourceInfo: {}", e.what());
    throw std::runtime_error("Failed to read src info");
  }
}

static void getKernels(Session &Session) {
  Session.KI = KernelInfo(kerma::getKernels(*Session.DeviceModule));
  Log::info("DetKer. Found {} kernels", Session.KI.getKernels().size());
  for (auto &Kernel : Session.KI.getKernels()) {
    Kernel.setSourceRange(Session.SI.getFunctionRange(Kernel.getName()));
    Log::debug("  Kernel #{}. {} @{}", Kernel.getID(), Kernel.getName(),
               Kernel.getSourceRange().getStart().getLine());
  }
}

static void getMemoryInfo(Session &Session) {
  DetectMemoriesPass DMP(Session.KI.getKernels());
  DMP.runOnModule(*Session.DeviceModule);
  Session.MI = DMP.getMemoryInfo();

  unsigned count = 0;
  for ( auto &K : Session.KI.getKernels())
    count += Session.MI.getForKernel(K).size();

  Log::info("DetMem. Found {} memories", count);

  for (auto &K : Session.KI.getKernels()) {
    auto Mem = Session.MI.getForKernel(K);
    Log::debug("  Kernel '{}' uses {} memories:", K.getName(), Mem.size());
    for (auto &M : Mem)
      Log::debug("    {}. {}:{} @{} {}", M.getID(), M.getName(),
                 M.getTypeSize(), M.getAddrSpace(), M.getDim().toString());
  }
}

void Server::initSession(const std::string &SourceDir,
                         const std::string &Source) {
  CurrSession = std::make_unique<Session>(Options, SourceDir, Source);
  RB.setSession(*CurrSession);

  getDeviceIR(*CurrSession, Compiler);

  getSourceInfo(*CurrSession);
  getKernels(*CurrSession);

  // find memories
  getMemoryInfo(*CurrSession);

  // next: extract assumptions
  //       - validate that all unknown values/dims have
  //         been provided with an assumed value/dim
  //       - if an assumption is for a memory whose size
  //         we found in the previous step, validate the
  //         the assumed dims
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
