#include "kerma/Cuda/Cuda.h"
#include "kerma/Support/SourceCode.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"
#include <gtest/gtest.h>


#include <kerma/Cuda/CudaKernel.h>
#include <kerma/Cuda/CudaModule.h>

#include "llvm/Support/SourceMgr.h"
#include <llvm/IRReader/IRReader.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>

using namespace llvm;
using namespace kerma;

#define POLYBENCH std::string("../../../../../examples/polybench/cuda/")

TEST( Constructor, CudaKernel_int)
{

  std::string deviceIR = POLYBENCH + "2MM/2mm-cuda-nvptx64-nvidia-cuda-sm_52.ll";
  LLVMContext context;
  SMDiagnostic error;
  auto deviceModule = parseIRFile(deviceIR, error, context);
  CudaModule module(*deviceModule.get(), *deviceModule.get());

  llvm::FunctionType *ty = FunctionType::get(Type::getVoidTy(context), false);
  llvm::Twine twine("hello");
  llvm::Function *f = Function::Create( ty,
                                        GlobalValue::InternalLinkage,
                                        twine,
                                        deviceModule.get());

  CudaKernel kernel(*f, CudaSide::DEVICE);
  CudaKernelLaunch launch(kernel);
  CudaKernelLaunchConfiguration emptyConfig;

  ASSERT_EQ(launch.getLine(), SRC_LINE_UNKNOWN);
  ASSERT_EQ(launch.getLaunchConfigutation(), emptyConfig);
  ASSERT_EQ(launch.getKernel(), kernel);
  ASSERT_EQ((bool) launch.inLoop(), false);
  ASSERT_EQ((bool) launch.inBranch(), false);
  ASSERT_EQ((bool) launch.inThenBranch(), false);
  ASSERT_EQ((bool) launch.inElseBranch(), false);

  CudaKernelLaunch launch2(kernel, 152);
  ASSERT_EQ(launch2.getLine(), 152);
}