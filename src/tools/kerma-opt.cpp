//
// Created by gkarlos on 1/3/20.
//

#include <llvm/Support/CommandLine.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/IRReader/IRReader.h>

#include <kerma/cuda/CudaProgram.h>
#include <kerma/cuda/CudaSupport.h>
#include <kerma/passes/detect-kernels/DetectKernels.h>

#include <kerma/passes/dg/Dot.h>

#include <string>

using namespace llvm;
using namespace kerma;
using namespace kerma::cuda;

static void PrintVersion(raw_ostream& OS) {
  OS << "Kerma Static Analyzer v0.1\n";
}

static cl::OptionCategory KermaOptCategory("Kerma-opt options");
static cl::opt<std::string> InputHostIR("host-ir",
    cl::desc("Specify Host-Side IR (.ll or .bc)"),
    cl::value_desc("filename"), cl::Required,
    cl::cat(KermaOptCategory));
static cl::opt<std::string> InputDeviceIR("device-ir",
    cl::desc("Specify Device-Side IR (.ll or .bc)"),
    cl::value_desc("filename"), cl::Required,
    cl::cat(KermaOptCategory));

static void filenameCheck(const std::string& IRFile, cuda::CudaSide side)
{
  if ( IRFile.size() < 4
       || IRFile.at(IRFile.size() - 3) != '.'
       || ( IRFile.substr(IRFile.size() - 3, IRFile.size()) != ".ll" &&
            IRFile.substr(IRFile.size() - 3, IRFile.size()) != ".bc") )
  {
    throw std::runtime_error( std::string("Invalid ")
                              + std::string((side == CudaSide::HOST)? "host" : "device")
                              + std::string(" IR file. Provide an .ll or .bc file."));
  }
}

static void extensionCheck( const std::string& hostIRFile,
                            const std::string& deviceIRFile) {
  std::string hostIRFileExtension =
      hostIRFile.substr(hostIRFile.size() - 3, hostIRFile.size());
  std::string deviceIRFileExtension =
      deviceIRFile.substr(deviceIRFile.size() - 3, deviceIRFile.size());

  if ( hostIRFileExtension != deviceIRFileExtension)
    throw std::runtime_error("IR files mismatch. Use either .ll or .bc files");
}

int main(int argc, char **argv) {
  llvm::InitLLVM LLVM_INIT(argc, argv);

  cl::HideUnrelatedOptions(KermaOptCategory);
  cl::SetVersionPrinter(PrintVersion);
  cl::ParseCommandLineOptions(argc, argv);

  try {
    filenameCheck(InputHostIR, CudaSide::HOST);
    filenameCheck(InputDeviceIR, CudaSide::DEVICE);
    extensionCheck(InputHostIR, InputDeviceIR);
    if ( InputHostIR == InputDeviceIR)
      throw std::runtime_error("Same file specified for both host and device IR");
  } catch(std::exception& e) {
    errs() << "kerma-opt: " << e.what() << "\n";
    return 1;
  }

  LLVMContext context;
  SMDiagnostic error;
  auto hostModule = llvm::parseIRFile(InputHostIR, error, context);
  auto deviceModule = llvm::parseIRFile(InputDeviceIR, error, context);

  CudaProgram program(hostModule.get(), deviceModule.get());

  legacy::PassManager PM;
  DetectKernelsPass *detectKernels = new DetectKernelsPass(&program);

  PM.add(detectKernels);
  PM.run(*program.getDeviceModule());

  for ( auto kernel : program.getKernels())
    kernel->pp(llvm::errs());



}