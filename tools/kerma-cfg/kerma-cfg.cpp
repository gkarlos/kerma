#include "boost/filesystem/convenience.hpp"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Analysis/CFGPrinter.h"

#include "kerma/Cuda/CudaKernel.h"
#include "kerma/Cuda/CudaModule.h"
#include "kerma/Pass/DetectKernels.h"
#include "kerma/Config/Config.h"
#include "kerma/Support/FileSystem.h"
#include "kerma/Support/Terminal.h"
#include "kerma/Support/Demangle.h"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <chrono>
#include <iomanip>

#define KERMA_CFG_VERSION_MAJ KERMA_MAJOR_VERSION
#define KERMA_CFG_VERSION_MIN "1"
#define KERMA_CFG_VERSION KERMA_MAJOR_VERSION "." KERMA_CFG_VERSION_MIN


using namespace llvm;
using namespace kerma;
using namespace kerma::term;
using namespace std::chrono;

#define PRINT_TOOL      bold("kerma-cfg: ")
#define PRINT_ERROR(m) (bold(red("error: ")) << (m))
#define ERROR(m)       (PRINT_TOOL << PRINT_ERROR(m))

#define VERBOSE_PRINTLN(m)   \
  if ( Verbose.getValue()) { \
    auto now = time_point_cast<milliseconds>(system_clock::now());  \
    auto in_time_t = std::chrono::system_clock::to_time_t(now);     \
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>( now.time_since_epoch()) % 1000; \
    std::stringstream ss;    \
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X")  \
       << "." << now_ms.count();                                    \
    errs() << ss.str() << " [info] " << (m) << "\n";                \
  }

static void PrintVersion(raw_ostream& OS) { OS << "kerma-cfg v" << KERMA_CFG_VERSION << "\n"; }

cl::opt<std::string> IRFileName( cl::Positional, 
                                 cl::desc("<IR file>"), 
                                 cl::value_desc("filename"), 
                                 cl::Required);

cl::opt<std::string> OutputDirectory("out", 
                                     cl::desc("Output directory. Defaults to current directory"), 
                                     cl::value_desc("directory"),
                                     cl::init(""));

cl::opt<std::string> Prefix("prefix",
                     cl::desc("String prepended to each generated filename"),
                     cl::value_desc("string"),
                     cl::init(""));

cl::opt<bool> StructureOnly("structure-only",
                            cl::desc("Display blocks without contents"),
                            cl::init(false));

cl::opt<bool> KernelsOnly("kernel-only",
                          cl::desc("Process only CUDA kernels. Ignore other functions"),
                          cl::init(false));

cl::opt<bool> Verbose("verbose",
                      cl::desc("Verbose"),
                      cl::init(false)); 



inline bool hasOutputDirectory() {
  return !OutputDirectory.getValue().empty();
}

inline void checkFileExtension(const std::string &filename) {
  std::string extension = boost::filesystem::extension(filename);
  if ( extension.compare(".bc") != 0 && extension.compare(".ll") != 0)
    throw std::runtime_error("Unknown LLVM IR file extension. Please use .bc or .ll");
  if ( extension.compare(".bc") == 0)
    throw std::runtime_error("Bitcode files (.bc) not supported yet. Please use .ll");
}

void writeCFGToDotFile(Function &F, bool isKernel)
{
  std::stringstream filename;
  
  if ( OutputDirectory.getValue().empty() )
    filename << "./";
  else
    filename << OutputDirectory.getValue() << "/";

  if ( !Prefix.getValue().empty())
    filename << Prefix.getValue() << ".";
  
  if ( isKernel)
    filename << "kernel.";
  
  filename << demangleFnWithoutArgs(F) << ".dot";

  VERBOSE_PRINTLN("Writting File: " + filename.str());

  std::error_code err;
  raw_fd_ostream File(filename.str(), err, sys::fs::F_Text);

  if (!err)
    WriteGraph(File, (const Function*)&F, StructureOnly.getValue());
  else
    throw std::runtime_error("Could not open file for writting!");
}

void doGenerateCFGs(const std::string &filename, 
                    const std::string &outdir, 
                    bool kernelsOnly, bool summary)
{
  LLVMContext context;
  SMDiagnostic err;
  legacy::PassManager PM;

  auto irModule = parseIRFile( filename, err, context);

  std::set<CudaKernel> kernels;

  VERBOSE_PRINTLN("Running Pass: Detecting Kernel Functions");

  DetectKernelsPass *DKP = new DetectKernelsPass();
  PM.add(DKP);
  PM.run(*irModule.get());
  DKP->getKernels(kernels);
  
  VERBOSE_PRINTLN("Creating CFGPrinter");
  
  auto CfgPrinter = llvm::createCFGPrinterLegacyPassPass();

  if ( kernelsOnly )
    for ( auto kernel : kernels) {
      writeCFGToDotFile(kernel.getFn(), true);
  }
  else {
    for ( auto& F : *irModule.get())
      writeCFGToDotFile(F, DKP->isKernel(F));
  }
}

int main(int argc, char **argv)
{
  int status = 0;

  llvm::InitLLVM LLVM_INIT(argc, argv);
  
  cl::SetVersionPrinter(PrintVersion);
  cl::ParseCommandLineOptions(argc, argv, "This program creates .dot files of the CFG of each function in an IR file");

  try {
    if ( !fileExists(IRFileName.getValue()) )
      throw std::runtime_error("IR file does not exist");

    if ( hasOutputDirectory() && !directoryExists(OutputDirectory.getValue()))
      throw std::runtime_error("Output directory does not exist");

    checkFileExtension(IRFileName.getValue());

    doGenerateCFGs(IRFileName, 
                   hasOutputDirectory()? OutputDirectory.getValue() : boost::filesystem::current_path().string(),
                   KernelsOnly.getValue(),
                   false);

  } catch(std::exception& e) {
    errs() << ERROR(e.what()) << "\n";
    status = 1;
  }

  llvm::llvm_shutdown();
  return status;
}