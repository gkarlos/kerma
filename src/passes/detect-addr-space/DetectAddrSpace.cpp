#include "kerma/Cuda/NVVM.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <kerma/passes/detect-addr-space/DetectAddrSpace.h>
#include <kerma/passes/detect-kernels/DetectKernels.h>

using namespace llvm;


/// Register Pass
char kerma::DetectAddrSpacePass::ID = 1;
static RegisterPass<kerma::DetectAddrSpacePass> INIT_DETECT_ADDR_SPACE("kerma-detect-addr-space", "Detect Address Space of Loads and Stores", false, true);


namespace kerma
{

void DetectAddrSpacePass::getAnalysisUsage(AnalysisUsage& AU) const 
{
  AU.setPreservesAll();
  AU.addRequired<DetectKernelsPass>();
}

bool DetectAddrSpacePass::runOnModule(llvm::Module &M) 
{
  auto kernels = &getAnalysis<DetectKernelsPass>().getKernels();

  for ( auto* kernel : *kernels) {
    outs() << "Detect-Address-Space: " << kernel->getFn()->getName() << "\n";

    for ( auto& BB : *kernel->getFn()) {
      for ( auto& I : BB) {
        bool loadOrStore = false;
        if ( auto* L = dyn_cast<LoadInst>(&I)) {
          loadOrStore = true;
          errs() << getAddrSpace(&I).getName() << " Load";
        } else if ( auto* S = dyn_cast<StoreInst>(&I)) {
          loadOrStore = true;
          errs() << getAddrSpace(&I).getName() << " Store";
        }

        if ( loadOrStore) {
          if ( I.getDebugLoc())
            errs() << " -- @" << I.getDebugLoc().getLine() << ":" << I.getDebugLoc().getCol();
          errs() << " -- " << I << "\n";
        }

        
      }
    }

  }

  return false;
}

std::string getValueName(Value *v) {
  // Constants can always generate themselves
  if(auto C=dyn_cast<ConstantInt>(v)) {
    SmallString<16> cint;
    C->getValue().toString(cint, 10, true);
    return std::string(cint.c_str());
  }

  // Need a function from here on out
  Function* F = nullptr;
  if(auto arg=dyn_cast<Argument>(v))
    F=arg->getParent();
  if(auto i=dyn_cast<Instruction>(v))
    F=i->getParent()->getParent();

  if(F == nullptr)
    return "tmp";

  for(auto i=inst_begin(F),e=inst_end(F); i!=e; ++i) {
    if(auto decl = dyn_cast<DbgDeclareInst>(&*i)) {
      if(decl->getAddress() == v) return decl->getVariable()->getName();
    } else if(auto val = dyn_cast<DbgValueInst>(&*i)) {
      if(val->getValue() == v) return val->getVariable()->getName();
    }
  }

  if(auto GEP=dyn_cast<GetElementPtrInst>(v)) {
    string base = getValueName(GEP->getPointerOperand());
    if(GEP->getNumIndices() > 0) {
      string offset = getValueName(*GEP->idx_begin());
      return base + "[" + offset + "]";
    } else {
      return "*" + base;
    }
  }
  if(auto L=dyn_cast<LoadInst>(v)) {
    return getValueName(L->getPointerOperand());
  }
  if(auto BO=dyn_cast<BinaryOperator>(v)) {
    string left = getValueName(BO->getOperand(0));
    string right = getValueName(BO->getOperand(1));
    switch(BO->getOpcode()) {
    case BinaryOperator::Add:
      return left + "+" + right;
    case BinaryOperator::Sub:
      return left + "-" + right;
    case BinaryOperator::Mul:
      return left + "*" + right;
    case BinaryOperator::SDiv:
    case BinaryOperator::UDiv:
      return left + "/" + right;
    case BinaryOperator::AShr:
    case BinaryOperator::LShr:
      return left + ">>" + right;
    case BinaryOperator::Shl:
      return left + "<<" + right;
    case BinaryOperator::And:
      return left + "&&" + right;
    case BinaryOperator::Or:
      return left + "||" + right;
    case BinaryOperator::Xor:
      return left + "^" + right;
    default:
      break;
    }
  }
  if(auto C=dyn_cast<CastInst>(v)) {
    return getValueName(C->getOperand(0));
  }
  if(auto CI=dyn_cast<CallInst>(v)) {
    if(auto F=CI->getCalledFunction()) {
      switch(F->getIntrinsicID()) {
        case Intrinsic::nvvm_read_ptx_sreg_tid_x:
          return "threadIdx.x";
        case Intrinsic::nvvm_read_ptx_sreg_tid_y:
          return "threadIdx.y";
        case Intrinsic::nvvm_read_ptx_sreg_tid_z:
          return "threadIdx.z";
        case Intrinsic::nvvm_read_ptx_sreg_ntid_x:
          return "threadDim.x";
        case Intrinsic::nvvm_read_ptx_sreg_ntid_y:
          return "threadDim.y";
        case Intrinsic::nvvm_read_ptx_sreg_ntid_z:
          return "threadDim.z";
        case Intrinsic::nvvm_read_ptx_sreg_ctaid_x:
          return "blockIdx.x";
        case Intrinsic::nvvm_read_ptx_sreg_ctaid_y:
          return "blockIdx.y";
        case Intrinsic::nvvm_read_ptx_sreg_ctaid_z:
          return "blockIdx.z";
        case Intrinsic::nvvm_read_ptx_sreg_nctaid_x:
          return "blockDim.x";
        case Intrinsic::nvvm_read_ptx_sreg_nctaid_y:
          return "blockDim.y";
        case Intrinsic::nvvm_read_ptx_sreg_nctaid_z:
          return "blockDim.z";
        case Intrinsic::nvvm_read_ptx_sreg_laneid:
          return "laneID";
        default:
          break;
      }
    }
  }

  DEBUG(errs() << "Unrecognized instruction: "; v->dump(););
  return "tmp";
}

AddressSpace DetectAddrSpacePass::getAddrSpace(llvm::Value *v)
{
  if ( auto load = dyn_cast<LoadInst>(v)) {
    return getAddrSpace(load->getPointerOperand());
  }

  if ( auto store = dyn_cast<StoreInst>(v)) {
    return getAddrSpace(store->getPointerOperand());
  }

  if ( auto gep = dyn_cast<GetElementPtrInst>(v)) {
    return getAddrSpace(gep->getPointerOperand());
  }

  if ( auto op = dyn_cast<Operator>(v)) {
    if ( op->getOpcode() == Instruction::AddrSpaceCast)
      return getAddrSpace(op->getOperand(0));
  }

  if ( auto alloca = dyn_cast<AllocaInst>(v)) {
    if ( auto pty = dyn_cast<PointerType>(alloca->getType()))
      if ( !pty->getElementType()->isPointerTy())
        return AddressSpace::UNKNOWN;
  }

  if ( v->getType()->isPointerTy()) {
    unsigned addr = v->getType()->getPointerAddressSpace();
    switch (addr) {
    case 0:
      return AddressSpace::GENERIC;
    case 1:
      return AddressSpace::GLOBAL;
    case 3:
      return AddressSpace::SHARED;
    case 4:
      return AddressSpace::CONSTANT;
    case 5:
      return AddressSpace::LOCAL;
    default:
      return AddressSpace::UNKNOWN;
    }
  }
}

}




