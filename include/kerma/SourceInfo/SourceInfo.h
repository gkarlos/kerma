#ifndef KERMA_SOURCEINFO_SOURCEINFO_H
#define KERMA_SOURCEINFO_SOURCEINFO_H

#include "kerma/SourceInfo/SourceRange.h"
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
    return *this;
  }

  const std::string &getFunctionOfLine(unsigned int Line) const;

  const SourceRange &getFunctionRange(const std::string &FunctionName) const;

  const std::pair<std::string, SourceRange>
  getFunctionRangePair(const std::string &FunctionName);

  const std::unordered_map<std::string, SourceRange> getKernelRanges() const {
    return KernelRanges;
  }

  const std::unordered_map<std::string, SourceRange>
  getDeviceFunctionRanges() const {
    return DeviceFunctionRanges;
  }
};

} // namespace kerma

#endif // KERMA_SOURCEINFO_SOURCEINFO_H