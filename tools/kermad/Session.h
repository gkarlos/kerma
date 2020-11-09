#ifndef KERMA_TOOLS_KERMAD_SESSION_H
#define KERMA_TOOLS_KERMAD_SESSION_H

#include "Options.h"
#include "kerma/Analysis/DetectKernels.h"
#include "kerma/Analysis/DetectMemories.h"
#include "kerma/Base/Kernel.h"
#include "kerma/Compile/Compiler.h"
#include "kerma/SourceInfo/SourceInfo.h"
#include "kerma/SourceInfo/SourceInfoBuilder.h"

#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <string>

namespace kerma {
namespace kermad {

namespace fs = boost::filesystem;

class Session {
private:
  unsigned int ID;
  struct Options &Options;
  void setInput(const std::string &SourceDir, const std::string &Source);
  void createWorkingDir();

public:
  Session(struct Options &Options, const std::string &Dir,
          const std::string &Source);

  ~Session();

  std::string WorkingDirName;
  std::string WorkingDir;
  std::string SourceDir;
  std::string Source;
  std::string CompileDb;
  std::string SourcePath;
  std::string CompileDbPath;

  std::string HostIRModuleName = Compiler::DefaultHostIRFile;
  std::string DeviceIRModuleName = Compiler::DefaultDeviceIRFile;
  std::string DeviceIRCanonModuleName = "canon." + DeviceIRModuleName;
  std::string DeviceIRCanonModule;

  std::unique_ptr<SourceInfoBuilder> SIB;

  llvm::LLVMContext Context;
  std::unique_ptr<llvm::Module> DeviceModule;
  // std::vector<Kernel> Kernels;
  KernelInfo KI;
  SourceInfo SI;
  MemoryInfo MI;

  std::string getDeviceIRModuleName() { return DeviceIRModuleName; }
  std::string getHostIRModuleName() { return HostIRModuleName; }
  std::string getHostIRModulePath() {
    return (fs::path(WorkingDir) / fs::path(HostIRModuleName)).string();
  }
  std::string getDeviceIRModulePath() {
    return (fs::path(WorkingDir) / fs::path(DeviceIRModuleName)).string();
  }

  unsigned int getID() const { return ID; }
  std::string getSource() const { return Source; }
  std::string getSourceDir() const { return Source; }
  std::string getSourcePath() const { return SourcePath; }
  std::string getCompileDb() const { return CompileDb; }
  std::string getCompileDbPath() const { return CompileDbPath; }

  std::string getWorkingDir() const { return WorkingDir; }
};

} // namespace kermad
} // namespace kerma

#endif // KERMA_TOOLS_KERMAD_SESSION_H