#include <gtest/gtest.h>

#include "kerma/Cuda/Cuda.h"
#include "kerma/Cuda/CudaKernel.h"
#include "kerma/Cuda/CudaModule.h"
#include "kerma/Pass/DetectKernels.h"
#include "kerma/Support/Demangle.h"
#include "kerma/Support/SourceCode.h"

#include "llvm/IR/Function.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IRReader/IRReader.h>
#include <memory>
#include <stdio.h>  // defines FILENAME_MAX
#include <unistd.h> // for getcwd()

using namespace llvm;
using namespace kerma;

#define POLYBENCH std::string("../../../../../examples/polybench/cuda/")

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
  std::string hostIR = POLYBENCH + "2MM/2mm.ll";

  LLVMContext context;
  SMDiagnostic error;
  auto deviceModule = parseIRFile(deviceIR, error, context);
  CudaModule program(*deviceModule.get(), *deviceModule.get());
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
  std::string hostIR = POLYBENCH + "2MM/2mm.ll";
  LLVMContext context;
  SMDiagnostic error;
  auto deviceModule = parseIRFile(deviceIR, error, context);
  CudaModule program(*deviceModule.get(), *deviceModule.get());

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
}

TEST(LineNumbers, Default )
{
  std::string deviceIR = POLYBENCH + "2MM/2mm-cuda-nvptx64-nvidia-cuda-sm_52.ll";
  LLVMContext context;
  SMDiagnostic error;
  auto deviceModule = parseIRFile(deviceIR, error, context);

  for ( auto &F : *deviceModule.get()) {
    CudaKernel A(F, CudaSide::DEVICE);
    ASSERT_EQ(A.getSignatureLineStart(), SRC_LINE_UNKNOWN);
    ASSERT_EQ(A.getSignatureLineEnd(), SRC_LINE_UNKNOWN);
    ASSERT_EQ(A.getSignatureNumLines(), 0);
    ASSERT_EQ(A.getBodyLineStart(), SRC_LINE_UNKNOWN);
    ASSERT_EQ(A.getBodyLineEnd(), SRC_LINE_UNKNOWN);
    ASSERT_EQ(A.getBodyNumLines(), 0);
    ASSERT_EQ(A.getLineStart(), A.getSignatureLineStart());
    ASSERT_EQ(A.getLineStart(), SRC_LINE_UNKNOWN);
    ASSERT_EQ(A.getLineEnd(), A.getBodyLineEnd());
    ASSERT_EQ(A.getLineEnd(), SRC_LINE_UNKNOWN);
    ASSERT_EQ(A.getNumLines(), 0);
  }
}

TEST( LineNumbers, SetSignatureValidStartEnd )
{
  std::string deviceIR = POLYBENCH + "2MM/2mm-cuda-nvptx64-nvidia-cuda-sm_52.ll";
  LLVMContext context;
  SMDiagnostic error;
  auto deviceModule = parseIRFile(deviceIR, error, context);

  for ( auto &F : *deviceModule.get()) {
    CudaKernel A(F, CudaSide::DEVICE);

    A.setSignatureLineStart(0);
    A.setSignatureLineEnd(1);
    ASSERT_EQ(A.getSignatureLineStart(), 0);
    ASSERT_EQ(A.getSignatureLineEnd(), 1);
    ASSERT_EQ(A.getSignatureNumLines(), 2);
    ASSERT_EQ(A.getBodyLineStart(), SRC_LINE_UNKNOWN);
    ASSERT_EQ(A.getBodyLineEnd(), SRC_LINE_UNKNOWN);
    ASSERT_EQ(A.getBodyNumLines(), 0);
    ASSERT_EQ(A.getLineStart(), 0);
    ASSERT_EQ(A.getLineEnd(), SRC_LINE_UNKNOWN);
    ASSERT_EQ(A.getNumLines(), 0);
  }
}

TEST( LineNumbers, SetLineNumbersFromFile )
{
  std::string deviceIR = POLYBENCH + "2MM/2mm-cuda-nvptx64-nvidia-cuda-sm_52.ll";
  LLVMContext context;
  SMDiagnostic error;
  auto deviceModule = parseIRFile(deviceIR, error, context);

  for ( auto &F : *deviceModule.get()) {
    if ( demangleFnWithoutArgs(F).compare("mm2_kernel1") != 0 ) {
      CudaKernel A(F, CudaSide::DEVICE);
      A.setSignatureLineStart(108);
      A.setSignatureLineEnd(108);
      A.setBodyLineStart(108);
      A.setBodyLineEnd(120);

      ASSERT_EQ( A.getSignatureLineStart(), 108);
      ASSERT_EQ( A.getSignatureLineEnd(), 108);
      ASSERT_EQ( A.getBodyLineStart(), 108);
      ASSERT_EQ( A.getBodyLineEnd(), 120);
      ASSERT_EQ( A.getSignatureNumLines(), 1);
      ASSERT_EQ( A.getBodyNumLines(), 13);
      ASSERT_EQ( A.getLineStart(), A.getSignatureLineStart());
      ASSERT_EQ( A.getLineStart(), 108);
      ASSERT_EQ( A.getLineEnd(), A.getBodyLineEnd());
      ASSERT_EQ( A.getLineEnd(), 120);
      ASSERT_EQ( A.getNumLines(), A.getBodyNumLines());
      ASSERT_EQ( A.getNumLines(), 13);

      CudaKernel B(F, CudaSide::DEVICE);
      A.setSignatureLines(108, 108);
      A.setBodyLines(108, 120);

      ASSERT_EQ( A.getSignatureLineStart(), 108);
      ASSERT_EQ( A.getSignatureLineEnd(), 108);
      ASSERT_EQ( A.getBodyLineStart(), 108);
      ASSERT_EQ( A.getBodyLineEnd(), 120);
      ASSERT_EQ( A.getSignatureNumLines(), 1);
      ASSERT_EQ( A.getBodyNumLines(), 13);
      ASSERT_EQ( A.getLineStart(), A.getSignatureLineStart());
      ASSERT_EQ( A.getLineStart(), 108);
      ASSERT_EQ( A.getLineEnd(), A.getBodyLineEnd());
      ASSERT_EQ( A.getLineEnd(), 120);
      ASSERT_EQ( A.getNumLines(), A.getBodyNumLines());
      ASSERT_EQ( A.getNumLines(), 13);
    }

    if ( demangleFnWithoutArgs(F).compare("mm2_kernel2") != 0 ) {
      CudaKernel A(F, CudaSide::DEVICE);
      A.setSignatureLineStart(123);
      A.setSignatureLineEnd(123);
      A.setBodyLineStart(124);
      A.setBodyLineEnd(136);

      ASSERT_EQ( A.getSignatureLineStart(), 123);
      ASSERT_EQ( A.getSignatureLineEnd(), 123);
      ASSERT_EQ( A.getBodyLineStart(), 124);
      ASSERT_EQ( A.getBodyLineEnd(), 136);
      ASSERT_EQ( A.getSignatureNumLines(), 1);
      ASSERT_EQ( A.getBodyNumLines(), 13);
      ASSERT_EQ( A.getLineStart(), A.getSignatureLineStart());
      ASSERT_EQ( A.getLineStart(), 123);
      ASSERT_EQ( A.getLineEnd(), A.getBodyLineEnd());
      ASSERT_EQ( A.getLineEnd(), 136);
      ASSERT_EQ( A.getNumLines(), A.getBodyNumLines() + 1);
      ASSERT_EQ( A.getNumLines(), 14);

      CudaKernel B(F, CudaSide::DEVICE);
      A.setSignatureLines(123, 123);
      A.setBodyLines(124, 136);

      ASSERT_EQ( A.getSignatureLineStart(), 123);
      ASSERT_EQ( A.getSignatureLineEnd(), 123);
      ASSERT_EQ( A.getBodyLineStart(), 124);
      ASSERT_EQ( A.getBodyLineEnd(), 136);
      ASSERT_EQ( A.getSignatureNumLines(), 1);
      ASSERT_EQ( A.getBodyNumLines(), 13);
      ASSERT_EQ( A.getLineStart(), A.getSignatureLineStart());
      ASSERT_EQ( A.getLineStart(), 123);
      ASSERT_EQ( A.getLineEnd(), A.getBodyLineEnd());
      ASSERT_EQ( A.getLineEnd(), 136);
      ASSERT_EQ( A.getNumLines(), A.getBodyNumLines() + 1);
      ASSERT_EQ( A.getNumLines(), 14);
    }
  }
}

TEST( LineNumbers, SignatureInconsistencies )
{
  std::string deviceIR = POLYBENCH + "2MM/2mm-cuda-nvptx64-nvidia-cuda-sm_52.ll";
  LLVMContext context;
  SMDiagnostic error;
  auto deviceModule = parseIRFile(deviceIR, error, context);

  for ( auto &F : *deviceModule.get()) {
    CudaKernel A(F, CudaSide::DEVICE);
    A.setSignatureLines(123, 123);
    A.setBodyLines(124, 136);

    ASSERT_EQ( A.getSignatureLineStart(), 123);
    ASSERT_EQ( A.getSignatureLineEnd(), 123);
    ASSERT_EQ( A.getBodyLineStart(), 124);
    ASSERT_EQ( A.getBodyLineEnd(), 136);
    ASSERT_EQ( A.getSignatureNumLines(), 1);
    ASSERT_EQ( A.getBodyNumLines(), 13);
    ASSERT_EQ( A.getLineStart(), A.getSignatureLineStart());
    ASSERT_EQ( A.getLineStart(), 123);
    ASSERT_EQ( A.getLineEnd(), A.getBodyLineEnd());
    ASSERT_EQ( A.getLineEnd(), 136);
    ASSERT_EQ( A.getNumLines(), A.getBodyNumLines() + 1);
    ASSERT_EQ( A.getNumLines(), 14);

    A.setSignatureLineStart(125);
    ASSERT_EQ( A.getSignatureLineStart(), 125);
    ASSERT_EQ( A.getSignatureLineEnd(), SRC_LINE_UNKNOWN);
    ASSERT_EQ( A.getBodyLineStart(), SRC_LINE_UNKNOWN);
    ASSERT_EQ( A.getBodyLineEnd(), SRC_LINE_UNKNOWN);
    ASSERT_EQ( A.getLineStart(), 125);
    ASSERT_EQ( A.getLineEnd(), SRC_LINE_UNKNOWN);
    ASSERT_EQ( A.getSignatureNumLines(), 0);
    ASSERT_EQ( A.getBodyNumLines(), 0);
    ASSERT_EQ( A.getNumLines(), 0);

    // Reset
    CudaKernel B(F, CudaSide::DEVICE);
    B.setSignatureLines(123, 123);
    B.setBodyLines(124, 136);

    B.setSignatureLineEnd(125);
    ASSERT_EQ( B.getSignatureLineStart(), 123);
    ASSERT_EQ( B.getSignatureLineEnd(), 125);
    ASSERT_EQ( B.getBodyLineStart(), SRC_LINE_UNKNOWN);
    ASSERT_EQ( B.getBodyLineEnd(), SRC_LINE_UNKNOWN);
    ASSERT_EQ( B.getLineStart(), 123);
    ASSERT_EQ( B.getLineEnd(), SRC_LINE_UNKNOWN);
    ASSERT_EQ( B.getSignatureNumLines(), 3);
    ASSERT_EQ( B.getBodyNumLines(), 0);
    ASSERT_EQ( B.getNumLines(), 0);

    // Reset
    CudaKernel C(F, CudaSide::DEVICE);
    C.setSignatureLines(123, 123);
    C.setBodyLines(124, 136);

    C.setSignatureLineStart(150);
    ASSERT_EQ( C.getSignatureLineStart(), 150);
    ASSERT_EQ( C.getSignatureLineEnd(), SRC_LINE_UNKNOWN);
    ASSERT_EQ( C.getBodyLineStart(), SRC_LINE_UNKNOWN);
    ASSERT_EQ( C.getBodyLineEnd(), SRC_LINE_UNKNOWN);
    ASSERT_EQ( C.getLineStart(), 150);
    ASSERT_EQ( C.getLineEnd(), SRC_LINE_UNKNOWN);
    ASSERT_EQ( C.getSignatureNumLines(), 0);
    ASSERT_EQ( C.getBodyNumLines(), 0);
    ASSERT_EQ( C.getNumLines(), 0);

    // Reset
    CudaKernel D(F, CudaSide::DEVICE);
    D.setSignatureLines(123, 123);
    D.setBodyLines(124, 136);

    D.setSignatureLineEnd(150);
    ASSERT_EQ( D.getSignatureLineStart(), 123);
    ASSERT_EQ( D.getSignatureLineEnd(), 150);
    ASSERT_EQ( D.getBodyLineStart(), SRC_LINE_UNKNOWN);
    ASSERT_EQ( D.getBodyLineEnd(), SRC_LINE_UNKNOWN);
    ASSERT_EQ( D.getLineStart(), 123);
    ASSERT_EQ( D.getLineEnd(), SRC_LINE_UNKNOWN);
    ASSERT_EQ( D.getSignatureNumLines(), 28);
    ASSERT_EQ( D.getBodyNumLines(), 0);
    ASSERT_EQ( D.getNumLines(), 0);
  }
}

TEST( LineNumbers, BodyInconsistencies )
{
  std::string deviceIR = POLYBENCH + "2MM/2mm-cuda-nvptx64-nvidia-cuda-sm_52.ll";
  LLVMContext context;
  SMDiagnostic error;
  auto deviceModule = parseIRFile(deviceIR, error, context);

  for ( auto &F : *deviceModule.get()) {
    CudaKernel A(F, CudaSide::DEVICE);
    A.setSignatureLines(123, 123);
    A.setBodyLines(124, 136);

    A.setBodyLineStart(0);
    ASSERT_EQ( A.getBodyLineStart(), SRC_LINE_UNKNOWN);
    ASSERT_EQ( A.getBodyLineEnd(), SRC_LINE_UNKNOWN);
    ASSERT_EQ( A.getBodyNumLines(), 0);
    ASSERT_EQ( A.getSignatureLineStart(), 123);
    ASSERT_EQ( A.getSignatureLineEnd(), 123);
    ASSERT_EQ( A.getSignatureNumLines(), 1);

    ASSERT_EQ( A.getLineEnd(), SRC_LINE_UNKNOWN);
    ASSERT_EQ( A.getNumLines(), 0);

    CudaKernel B(F, CudaSide::DEVICE);
    B.setSignatureLines(123, 123);
    B.setBodyLines(124, 136);

    B.setBodyLineEnd(123);
    ASSERT_EQ( B.getBodyLineStart(), 124);
    ASSERT_EQ( B.getBodyLineEnd(), SRC_LINE_UNKNOWN);
    ASSERT_EQ( B.getBodyNumLines(), 0);
    ASSERT_EQ( B.getLineEnd(), SRC_LINE_UNKNOWN);
    ASSERT_EQ( B.getNumLines(), 0);

    ASSERT_EQ( B.getSignatureLineStart(), 123);
    ASSERT_EQ( B.getSignatureLineEnd(), 123);
    ASSERT_EQ( B.getSignatureNumLines(), 1);

    CudaKernel C(F, CudaSide::DEVICE);
    C.setSignatureLines(123, 123);
    C.setBodyLines(124, 136);

    C.setBodyLineEnd(130);
    ASSERT_EQ( C.getBodyLineStart(), 124);
    ASSERT_EQ( C.getBodyLineEnd(), 130);
    ASSERT_EQ( C.getBodyNumLines(), 7);
    ASSERT_EQ( C.getLineEnd(), 130);
    ASSERT_EQ( C.getNumLines(), 8);

    ASSERT_EQ( C.getSignatureLineStart(), 123);
    ASSERT_EQ( C.getSignatureLineEnd(), 123);
    ASSERT_EQ( C.getSignatureNumLines(), 1);


  }
}