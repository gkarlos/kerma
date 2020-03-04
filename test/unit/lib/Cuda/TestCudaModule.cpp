#include <gtest/gtest.h>

#include <kerma/Cuda/Cuda.h>
#include <kerma/Cuda/CudaModule.h>
#include <kerma/Pass/DetectKernels.h>

#include <llvm/Support/SourceMgr.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IRReader/IRReader.h>

using namespace llvm;
using namespace kerma;

#define POLYBENCH std::string("../../../../../examples/polybench/cuda/")

TEST(Create, Simple )
{
  std::string deviceIR = POLYBENCH + "2MM/2mm-cuda-nvptx64-nvidia-cuda-sm_52.ll";
  std::string hostIR = POLYBENCH + "2MM/2mm.ll";

  LLVMContext context;
  SMDiagnostic error;
  auto deviceModule = parseIRFile(deviceIR, error, context);
  auto hostModule = parseIRFile(hostIR, error, context);

  EXPECT_NO_THROW(CudaModule program(*deviceModule.get(), *deviceModule.get()));
  
  CudaModule module(*hostModule.get(), *deviceModule.get());

  ASSERT_EQ( module.getDeviceModule().getName().compare( deviceModule.get()->getName()), 0);
  ASSERT_EQ( module.getHostModule().getName().compare( hostModule.get()->getName()), 0);
  ASSERT_EQ( module.getArch(), CudaArch::sm_52);
  ASSERT_TRUE( module.is64bit());
  ASSERT_FALSE( module.is32bit());
  ASSERT_EQ( module.getKernels().size(), 0);
  ASSERT_EQ( module.getNumberOfKernels(), 0);
  ASSERT_EQ( module.getSourceFilename().compare("2mm.cu"), 0);
  ASSERT_EQ( module.getSourceFilename().compare(deviceModule.get()->getSourceFileName()), 0);
  ASSERT_EQ( module.getSourceFilename().compare(hostModule.get()->getSourceFileName()), 0);
}

TEST(Create, thenDetectKernelsPass )
{
  std::string deviceIR = POLYBENCH + "2MM/2mm-cuda-nvptx64-nvidia-cuda-sm_52.ll";
  std::string hostIR = POLYBENCH + "2MM/2mm.ll";

  LLVMContext context;
  SMDiagnostic error;
  auto deviceModule = parseIRFile(deviceIR, error, context);
  auto hostModule = parseIRFile(hostIR, error, context);

  EXPECT_NO_THROW(CudaModule program(*deviceModule.get(), *deviceModule.get()));
  
  CudaModule module(*hostModule.get(), *deviceModule.get());

  ASSERT_EQ( module.getDeviceModule().getName().compare( deviceModule.get()->getName()), 0);
  ASSERT_EQ( module.getHostModule().getName().compare( hostModule.get()->getName()), 0);
  ASSERT_EQ( module.getArch(), CudaArch::sm_52);
  ASSERT_TRUE( module.is64bit());
  ASSERT_FALSE( module.is32bit());
  ASSERT_EQ( module.getKernels().size(), 0);
  ASSERT_EQ( module.getNumberOfKernels(), 0);
  ASSERT_EQ( module.getSourceFilename().compare("2mm.cu"), 0);
  ASSERT_EQ( module.getSourceFilename().compare(deviceModule.get()->getSourceFileName()), 0);
  ASSERT_EQ( module.getSourceFilename().compare(hostModule.get()->getSourceFileName()), 0);

  legacy::PassManager PM;
  DetectKernelsPass *dkp = new DetectKernelsPass(module);
  PM.add(dkp);
  PM.run(module.getDeviceModule());

  ASSERT_EQ( module.getKernels().size(), 2);
  ASSERT_EQ( module.getNumberOfKernels(), 2);
}