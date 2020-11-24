#ifndef KERMA_SOURCEINFO_SOURCEINFO_H
#define KERMA_SOURCEINFO_SOURCEINFO_H

#include "kerma/Base/Kernel.h"
#include "kerma/SourceInfo/SourceLoc.h"
#include "kerma/SourceInfo/SourceRange.h"
#include <llvm/IR/DebugLoc.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/raw_ostream.h>
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

  std::vector<std::tuple<SourceRange, SourceRange, SourceRange>> IfStmts;

  std::vector<SourceRange> DoConditions;
  std::vector<SourceRange> WhileConditions;
  std::vector<SourceRange> ForInits;
  std::vector<SourceRange> ForHeaders;
  std::vector<SourceRange> Stmts;
  std::vector<SourceRange> Exprs;
  std::vector<std::vector<SourceRange> *> Containers{
      &IfInitializers, &DoConditions, &WhileConditions, &ForInits, &ForHeaders,
      &Stmts,          &Exprs};

protected:
  void clear() {
    KernelRanges.clear();
    DeviceFunctionRanges.clear();
    IfInitializers.clear();
    IfStmts.clear();
    DoConditions.clear();
    WhileConditions.clear();
    ForInits.clear();
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
    IfStmts = O.IfStmts;
    DoConditions = O.DoConditions;
    WhileConditions = O.WhileConditions;
    ForInits = O.ForInits;
    ForHeaders = O.ForHeaders;
    Stmts = O.Stmts;
    Exprs = O.Exprs;
    return *this;
  }

  const std::vector<std::tuple<SourceRange, SourceRange, SourceRange>> &
  getIfStmts() {
    return IfStmts;
  }
  std::vector<SourceRange> getIfConditionsInRange(SourceRange R);
  std::vector<std::tuple<SourceRange, SourceRange, SourceRange>>
  getIfRangesInRange(SourceRange R);
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
  const SourceRange &getForInitRangeForLoc(const SourceLoc &L);
  const std::tuple<SourceRange, SourceRange, SourceRange> *
  getRangeForBranch(llvm::BranchInst *BI);

  llvm::raw_ostream &
  print(llvm::raw_ostream &OS,
        const std::tuple<SourceRange, SourceRange, SourceRange> &IfRanges);
};

} // namespace kerma

#endif // KERMA_SOURCEINFO_SOURCEINFO_H