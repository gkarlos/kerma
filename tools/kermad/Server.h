#ifndef KERMA_TOOLS_KERMAD_SERVER_H
#define KERMA_TOOLS_KERMAD_SERVER_H

#include "Options.h"

#include "kerma/Compile/Compiler.h"
#include "kerma/SourceInfo/SourceInfoBuilder.h"

#include <cxxtools/eventloop.h>
#include <cxxtools/log.h>
#include <cxxtools/json/rpcserver.h>
#include <memory>
#include <mutex>

#include "Session.h"
#include "Response.h"

namespace kerma {
namespace kermad {

using KermaRes = std::string;

class Server {
private:
  Options &Options;
  Compiler Compiler;
  bool Started;
  cxxtools::EventLoop Loop;
  std::unique_ptr<cxxtools::json::RpcServer> RpcServer;
  std::unique_ptr<Session> CurrSession;
  std::unique_ptr<SourceInfoBuilder> SIB;
  ResponseBuilder RB;
  std::mutex Mutex;

  /// Start a new session
  void initSession(const std::string& SourceDir, const std::string& Source);
  /// Kill the active session. no-op if no active session
  /// @returns true - session killed. false otherwise
  bool killSession();
public:
  Server(struct Options &Options);
  /// Start the server
  /// no-op if the server is running
  /// @returns true - server was started. false otherwise
  bool start();

  /// Stop the server
  /// no-op if the server is stopped
  /// @return true - server was stopped. false otherwise
  bool stop();

  /// Check if the server is running
  bool isRunning() const { return Started; }

  /// Check if the server is on an active session currently
  bool hasActiveSession() const { return CurrSession.get(); }


  // Rpc interface below
protected:
  /**
    * @brief Start a new Kerma Session
    *
    * @param Source          path to the .cu file
    * @param CompileCommands path to the compile_commands.json file
    * @return KermaRes
    */
  KermaRes StartSession(const std::string& Dir, const std::string& Source, const std::string& CompileCommands);
  KermaRes StopSession(bool exit);

};

} // end namespace kermad
} // end namespace kerma

#endif