#include "Server.h"
#include "Options.h"
#include "Session.h"
#include "kerma/Analysis/DetectAssumptions.h"
#include "kerma/Analysis/DetectKernels.h"
#include "kerma/Analysis/DetectMemories.h"
#include "kerma/Support/Json.h"
#include "kerma/Support/Log.h"
#include "kerma/Transforms/Canonicalize/Canonicalizer.h"
#include "kerma/Transforms/StripAnnotations.h"

#include <boost/filesystem/path.hpp>
#include <cxxtools/json/rpcserver.h>
#include <cxxtools/log/cxxtools.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils.h>
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
  Log::info("Cmpl. {} -> {}", Session.SourcePath, Session.DeviceIRName);

  // Compiler source to an IR file
  if (!Compiler.EmitDeviceIR(Session.SourcePath, Session.DeviceIRName)) {
    Log::error("Compilation failed");
    throw std::runtime_error("Failed to compile LLVM IR");
  }

  // Read the IR file into memory
  SMDiagnostic Err;
  Session.DeviceModule =
      llvm::parseIRFile(Session.getDeviceIRPath(), Err, Session.Context);
  if (!Session.DeviceModule)
    _ERROR_("Failed to parse device IR: {}", Session.getDeviceIRPath());
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
  Log::info("DetKern. Found {} kernels", Session.KI.getKernels().size());
  for (auto &Kernel : Session.KI.getKernels()) {
    Kernel.setSourceRange(Session.SI.getFunctionRange(Kernel.getName()));
    Log::debug("  Kernel #{}. {} @{}", Kernel.getID(), Kernel.getName(),
               Kernel.getSourceRange().getStart().getLine());
  }
}

static void getMemoryInfo(Session &Session) {
  DetectMemoriesPass DMP(Session.KI);
  DMP.runOnModule(*Session.DeviceModule);
  Session.MI = DMP.getMemoryInfo();

  auto Args = Session.MI.getArgMemCount();
  auto GVs = Session.MI.getGVMemCount();

  Log::info("DetMem. Found {} memories (A: {}, G: {})", Args + GVs, Args, GVs);

  for (auto &K : Session.KI.getKernels()) {
    auto Mem = Session.MI.getForKernel(K);
    Log::debug("  Kernel '{}' uses {} memories:", K.getName(), Mem.size());
    for (auto &M : Mem)
      Log::debug("    {}. {}:{} @{} {}", M.getID(), M.getName(),
                 M.getTypeSize(), M.getAddrSpace(), M.getDim().toString());
  }
}

static void getAssumptionInfo(Session &Session) {
  DetectAsumptionsPass DAP(&Session.KI, &Session.MI);
  DAP.runOnModule(*Session.DeviceModule);
  Session.AI = DAP.getAssumptionInfo();

  Log::info("Found {} assumptions (D: {}, V: {})", Session.AI.getSize(),
            Session.AI.getDimCount(), Session.AI.getValCount());
}

static void canonicalize(Session &Session) {
  Log::info("Cano. {}/{} -> {}", Session.WorkingDir, Session.DeviceIRName,
            Session.DeviceIRCanonName);

  llvm::legacy::PassManager PM;
  PM.add(new StripAnnotationsPass(Session.KI));
  PM.add(createPromoteMemoryToRegisterPass());
  PM.add(createDeadCodeEliminationPass());
  PM.add(new CanonicalizerPass());
  PM.run(*Session.DeviceModule);

  // Write the canonicalized IR to a file
  Session.DeviceIRCanon =
      (fs::path(Session.WorkingDir) / fs::path(Session.DeviceIRCanonName))
          .string();
  std::error_code err;
  llvm::raw_fd_ostream F(Session.DeviceIRCanon, err);

  if (err)
    Log::error("Failed to create {}. ({})", Session.DeviceIRCanonName,
               err.message());
  else
    Session.DeviceModule->print(F, nullptr);
}

void Server::initSession(const std::string &SourceDir,
                         const std::string &Source) {
  CurrSession = std::make_unique<Session>(Options, SourceDir, Source);
  RB.setSession(*CurrSession);

  getDeviceIR(*CurrSession, Compiler);

  getSourceInfo(*CurrSession);
  getKernels(*CurrSession);

  getMemoryInfo(*CurrSession);
  getAssumptionInfo(*CurrSession);

  canonicalize(*CurrSession);
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
