#include <gtest/gtest.h>

#include "kerma/Cuda/Cuda.h"
#include "kerma/Cuda/CudaKernel.h"
#include "kerma/Cuda/CudaProgram.h"
#include "kerma/Pass/DetectKernels.h"
#include "kerma/Support/Demangle.h"

#include "llvm/IR/Function.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <llvm/IR/LegacyPassManager.h>
// #include <llvm/Support/SourceMgr.h>
#include <llvm/IRReader/IRReader.h>
#include <memory>
#include <stdio.h>  // defines FILENAME_MAX
#include <unistd.h> // for getcwd()

using namespace llvm;
using namespace kerma;

#define POLYBENCH std::string("../../../../../examples/polybench/cuda/")

DetectKernelsPass *
runDetectKernelsPass(const std::string& hostIR, const std::string& deviceIR)
{
  LLVMContext context;
  SMDiagnostic error;
  auto hostModule = llvm::parseIRFile(hostIR, error, context);
  auto deviceModule = llvm::parseIRFile(deviceIR, error, context);
  CudaProgram program(*hostModule.get(), *deviceModule.get());

  legacy::PassManager PM;
  DetectKernelsPass *dkp = new DetectKernelsPass(program);
  PM.add(dkp);
  PM.run(program.getDeviceModule());
  return dkp;
}

void cleanupPass(DetectKernelsPass **dkp)
{
  delete *dkp;
  *dkp = nullptr;
}

std::unique_ptr<llvm::Module>
readLLVMModule(const std::string& IRFile)
{
  LLVMContext context;
  SMDiagnostic error;
  auto deviceModule = parseIRFile(IRFile, error, context);
  return deviceModule;
}

std::string GetCurrentWorkingDir()
{
  std::string cwd("\0",FILENAME_MAX+1);
  return getcwd(&cwd[0],cwd.capacity());
}

TEST(Constructor, Function)
{
  std::string deviceIR = POLYBENCH + "2MM/2mm-cuda-nvptx64-nvidia-cuda-sm_52.ll";
  std::string hostIR = POLYBENCH + "2MM/2m.ll";

  LLVMContext context;
  SMDiagnostic error;
  auto deviceModule = parseIRFile(deviceIR, error, context);
  CudaProgram program(*deviceModule.get(), *deviceModule.get());
  legacy::PassManager PM;
  DetectKernelsPass *dkp = new DetectKernelsPass(program);
  PM.add(dkp);
  PM.run(program.getDeviceModule());

  for ( auto &F : *deviceModule.get()) {
    if ( dkp->isKernel(F)) {
      CudaKernel kernel(F);
      ASSERT_EQ(kernel.getIRModuleSide(), CudaSide::Unknown);

      kernel.setIRModuleSide(CudaSide::DEVICE);
      ASSERT_EQ(kernel.getIRModuleSide(), CudaSide::DEVICE);

      ASSERT_EQ(kernel.getNumArgs(), F.arg_size());
      ASSERT_EQ(kernel.getFn().getName().compare(F.getName()), 0);
      ASSERT_EQ(kernel.getName().compare(demangleFnWithoutArgs(F)), 0);
      ASSERT_EQ(kernel.getMangledName().compare(F.getName()), 0);
      ASSERT_EQ(kernel.getNumArgs(), F.arg_size());
    }
  }
}

TEST(Constructor, Function_CudaSide)
{
  std::string deviceIR = POLYBENCH + "2MM/2mm-cuda-nvptx64-nvidia-cuda-sm_52.ll";
  std::string hostIR = POLYBENCH + "2MM/2m.ll";
  LLVMContext context;
  SMDiagnostic error;
  auto deviceModule = parseIRFile(deviceIR, error, context);
  CudaProgram program(*deviceModule.get(), *deviceModule.get());

  legacy::PassManager PM;
  DetectKernelsPass *dkp = new DetectKernelsPass(program);
  PM.add(dkp);
  PM.run(program.getDeviceModule());

  for ( auto kernel : dkp->getKernels()) {
    ASSERT_EQ(kernel.getIRModuleSide(), CudaSide::DEVICE);

    bool kernelFnFoundInIR = false;

    llvm::Function *kernelF = nullptr;

    for ( auto it = deviceModule.get()->functions().begin();
          it != deviceModule.get()->functions().end(); it++ ) {
      if ( static_cast<Value*>(&*it) == static_cast<Value*>(&kernel.getFn())) {
        kernelF = &*it;
        kernelFnFoundInIR = true;
        break;
      }
    }

    ASSERT_TRUE(kernelFnFoundInIR);

    EXPECT_NE(kernelF, nullptr);
    
    ASSERT_EQ(kernel.getNumArgs(), kernelF->arg_size());
    ASSERT_EQ(kernel.getFn().getName().compare(kernelF->getName()), 0);
    ASSERT_EQ(kernel.getName().compare(demangleFnWithoutArgs(*kernelF)), 0);
    ASSERT_EQ(kernel.getMangledName().compare(kernelF->getName()), 0);
    ASSERT_EQ(kernel.getNumArgs(), kernelF->arg_size());
  }
    
  // cleanupPass(&dkp);
}

TEST(LineNumber, all )
{
  std::string deviceIR = POLYBENCH + "2MM/2mm-cuda-nvptx64-nvidia-cuda-sm_52.ll";
  LLVMContext context;
  SMDiagnostic error;
  auto deviceModule = parseIRFile(deviceIR, error, context);

  for ( auto &F : *deviceModule.get()) {
    CudaKernel f(F, CudaSide::DEVICE);
    ASSERT_EQ(f.getLineStart(), 0);
    ASSERT_EQ(f.getLineEnd(), 0);
    ASSERT_EQ(f.getNumLines(), 0);

    f.setLineStart(10);
    ASSERT_EQ(f.getLineStart(), 10);

    f.setLineEnd(15);
    ASSERT_EQ(f.getLineEnd(), 15);

    ASSERT_EQ(f.getNumLines(), 5);

    f.setLineEnd(0);
    ASSERT_EQ(f.getNumLines(), 0);
  }
}