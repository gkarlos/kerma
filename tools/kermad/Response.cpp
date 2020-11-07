#include "Response.h"
#include <stdexcept>

namespace kerma {
namespace kermad {

Json ResponseBuilder::getForStartSession() {
  if (!Session)
    throw new std::runtime_error("SessionBuilder has no session");

  Json Res;
  Res["kernels"] = Json::array();
  for (auto &Kernel : Session->Kernels) {
    auto Range = Kernel.getSourceRange();
    Res["kernels"].push_back(
        {{"name", Kernel.getName()},
         {"id", Kernel.getID()},
         {"range",
          {Range.getStart().getLine(), Range.getStart().getCol(),
           Range.getEnd().getLine(), Range.getEnd().getCol()}}});
  }
  Res["device_functions"] = Json::array();
  for (auto &E : Session->SI.getDeviceFunctionRanges()) {
    auto &Range = E.second;
    Res["device_functions"].push_back(
        {// currently kermaview just uses ranges to highlight
         // the function in the editor. However, to keep the
         // interface consistent lets add name and id fields
         {{"name", "dev_fun"},
          {"id", 0},
          {"range",
           {Range.getStart().getLine(), Range.getStart().getCol(),
            Range.getEnd().getLine(), Range.getEnd().getCol()}}}});
  }

  return Res;
}

Json ResponseBuilder::getForStopSession() {
  Json Res;
  Res["status"] = "success";
  return Res;
}

} // namespace kermad
} // namespace kerma