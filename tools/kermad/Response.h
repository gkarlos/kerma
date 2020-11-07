#ifndef KERMA_TOOLS_KERMAD_RESPONSE_H
#define KERMA_TOOLS_KERMAD_RESPONSE_H

#include "Session.h"
#include "nlohmann/json_fwd.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace kerma {
namespace kermad {

using Json = nlohmann::json;

/// This class builds responses to the RPC functions
/// for a given Session.
/// Responses are Json objects and should be used
/// by puting a res.dump() call as the last stmt
/// in an RPC function.
/// If the builder has no Session attached to it,
/// getFor... methods will throw
/// For usage examples see Server.cpp
class ResponseBuilder {
public:
  ResponseBuilder() : Session(nullptr) {}
  ResponseBuilder(Session &Session) : Session(&Session) {}
  void setSession(Session &Session) { this->Session = &Session;}
  void clearSession() { Session = nullptr; }
  Session *getSession() { return Session; }

  Json getForStartSession();
  Json getForStopSession();

private:
  Session *Session;
};

} // namespace kermad
} // namespace kerma

#endif // KERMA_TOOLS_KERMAD_RESPONSE_H