#ifndef KERMA_SOURCEINFO_SOURCEINFO_H
#define KERMA_SOURCEINFO_SOURCEINFO_H

#include "kerma/Base/Kernel.h"
#include "kerma/SourceInfo/SourceRange.h"
#include <llvm/IR/DebugLoc.h>
#include <optional>
#include <string>
#include <unordered_map>

namespace kerma {

class SourceInfo {
  friend class SourceInfoConsumer;
  friend class SourceInfoVisitor;
  friend class SourceInfoBuilder;

private:
  std::unordered_map<std::string, SourceRange> KernelRanges;
  std::unordered_map<std::string, SourceRange> DeviceFunctionRanges;
  std::vector<SourceRange> IfInitializers;
  std::vector<SourceRange> IfConditions;
  std::vector<SourceRange> DoConditions;
  std::vector<SourceRange> WhileConditions;
  std::vector<SourceRange> ForHeaders;
  std::vector<SourceRange> Stmts;
  std::vector<SourceRange> Exprs;
  std::vector<std::vector<SourceRange> *> Containers{
      &IfInitializers, &IfConditions, &DoConditions, &WhileConditions,
      &ForHeaders,     &Stmts,        &Exprs};

protected:
  void clear() {
    KernelRanges.clear();
    DeviceFunctionRanges.clear();
    IfInitializers.clear();
    IfConditions.clear();
    DoConditions.clear();
    WhileConditions.clear();
    ForHeaders.clear();
    Stmts.clear();
  }

public:
  SourceInfo() = default;
  ~SourceInfo() = default;
  SourceInfo(const SourceInfo &O) { *this = O; }
  SourceInfo &operator=(const SourceInfo &O) {
    KernelRanges = O.KernelRanges;
    DeviceFunctionRanges = O.DeviceFunctionRanges;
    IfInitializers = O.IfInitializers;
    IfConditions = O.IfConditions;
    DoConditions = O.DoConditions;
    WhileConditions = O.WhileConditions;
    ForHeaders = O.ForHeaders;
    Stmts = O.Stmts;
    Exprs = O.Exprs;
    return *this;
  }

  const std::vector<SourceRange> &getIfConditions() { return IfConditions; }
  const std::vector<SourceRange> &getDoConditions() { return DoConditions; }
  const std::vector<SourceRange> &getWhileConditions() {
    return WhileConditions;
  }
  const std::vector<SourceRange> &getForHeaders() { return ForHeaders; }
  const std::vector<SourceRange> &getStmts() { return Stmts; }
  const std::vector<SourceRange> &getExprs() { return Exprs; }

  const std::string &getFunctionOfLine(unsigned int Line) const;

  const SourceRange &getFunctionRange(const std::string &FunctionName) const;

  const std::pair<std::string, SourceRange>
  getFunctionRangePair(const std::string &FunctionName);

  std::vector<SourceRange> getRangesInRange(const SourceRange &R,
                                            bool strict = false);

  const std::unordered_map<std::string, SourceRange> getKernelRanges() const {
    return KernelRanges;
  }

  const std::unordered_map<std::string, SourceRange>
  getDeviceFunctionRanges() const {
    return DeviceFunctionRanges;
  }

  const SourceRange &getRangeForLoc(const SourceLoc &L);
  const SourceRange &getRangeForLoc(const llvm::DebugLoc &DL);
};

} // namespace kerma

#endif // KERMA_SOURCEINFO_SOURCEINFO_H