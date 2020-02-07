#include <kerma/passes/dg/DGDotPass.h>
#include "kerma/passes/dg/Dot.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/Casting.h"
#include <llvm/IR/Function.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DebugLoc.h>
#include <llvm/IR/DebugInfoMetadata.h>

#include <kerma/passes/detect-kernels/DetectKernels.h>
#include <kerma/Support/FileSystem.h>
#include <kerma/Support/LLVMStringUtils.h>
#include <kerma/passes/Util.h>

#include <string>
#include <sstream>
#include <map>
#include <utility>

using namespace llvm;
using namespace kerma;

/* command line args for when the pass is run in opt */
static llvm::cl::opt<std::string> DGOutDir("dg-dir", cl::desc("Specify a file name to write"), cl::value_desc("filename"));
static llvm::cl::opt<bool>        DGMetadata("dg-meta", cl::desc("Include metadata instructions"));

char DGDotPass::ID = 2;
char DGDotKernelPass::ID = 3;

static llvm::RegisterPass<DGDotPass>       INIT_DG_DOT("kerma-dg-dot", 
                                                       "Create a Dot dependency graph", false, true);
static llvm::RegisterPass<DGDotKernelPass> INIT_DG_DOT_KERNEL("kerma-dg-dot-kernel", 
                                                              "Create a Dit dependency graph for kernel functions", false, true);
                                                              
DGDotPass::DGDotPass() : FunctionPass(ID), dotWriter_("TESTOUT.dot")
{}
DGDotKernelPass::DGDotKernelPass() : DGDotPass()
{}

/// DGDotKernelPass
void
DGDotKernelPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const
{
  AU.setPreservesAll();
  AU.addRequired<kerma::DetectKernelsPass>();
}

static bool isKernel(llvm::Function& fn, std::set<kerma::cuda::CudaKernel*>& kernels) {
  for ( auto *kernel: kernels)
    if ( kernel->getFn()->getName() == fn.getName())
      return true;

  return false;
}

bool
DGDotKernelPass::runOnFunction(llvm::Function &F)
{
  if ( kernels_ == nullptr) {
    kernels_ = &getAnalysis<kerma::DetectKernelsPass>().getKernels();
    if ( this->kernels_ == nullptr) {
      llvm::errs() << "[error] DGDotKernelPass::runOnFunction('" + F.getName() + "'): internal error: kernels == nullptr\n";
      return false;
    }
  }

  if ( isKernel(F, *kernels_)) {
    DGDotPass *p = new DGDotPass();
    p->runOnFunction(F);
    delete p;
  }

  return false;
}

void
DGDotKernelPass::print(llvm::raw_ostream &OS, const llvm::Module *M) const
{
  /// TODO
}




/// DGDotPass

void
DGDotPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const
{
  AU.setPreservesAll();
}

void
DGDotPass::print(llvm::raw_ostream &OS, const llvm::Module *M) const
{
  /// TODO implement print
}

DotNode&
DGDotPass::lookupNodeOrNew(Value &V) 
{ 
  auto it = lookup_.find(&V);
  if (  it != lookup_.end()) {
    // errs() << "FOUND: " << it->second << "\n";
    return it->second;
  }
    

  if ( auto *I = dyn_cast<Instruction>(&V)) {
    if ( auto *load = dyn_cast<LoadInst>(I))
      lookup_.insert(std::make_pair(&V, Dot::createNode(*load)));
    else if ( auto *store = dyn_cast<StoreInst>(I))
      lookup_.insert(std::make_pair(&V, Dot::createNode(*store)));
    else if( auto *alloca = dyn_cast<AllocaInst>(I))
      lookup_.insert(std::make_pair(&V, Dot::createNode(*alloca)));
    else
      lookup_.insert(std::make_pair(&V, Dot::createNode(*I)));
  }
  else {
    if ( auto *constant = dyn_cast<Constant>(&V))
      lookup_.insert(std::make_pair(&V, Dot::createNode(*constant)));
    else if ( auto *argument = dyn_cast<Argument>(&V))
      lookup_.insert(std::make_pair(&V, Dot::createNode(*argument)));
    else
        lookup_.insert(std::make_pair(&V, Dot::createNode(V)));
  } 

  // errs() << "NEW: " << lookup_.find(&V)->second << "\n";
  return lookup_.find(&V)->second;
}



bool
DGDotPass::runOnFunction(llvm::Function &F) 
{
  using edge=std::pair<DotNode, DotNode>;
  std::set<edge> edges;

  errs() << "Results for function: " << F.getName() << "\n";
  for ( BasicBlock &BB : F) {
    for ( Instruction &I : BB) {
      auto* UserValue = cast<Value>(&I);
      auto& UserNode = lookupNodeOrNew(*UserValue);

      for ( Use &use : I.operands()) {
        
        Value *UseValue = use.get();

        // if ( auto *UsedInst = dyn_cast<Instruction>(UseValue) ) {
        auto& UseNode = lookupNodeOrNew(*UseValue);
        DotEdge e(UseNode, UserNode);
        // errs() << e << " || " << UseNode << " -> " << UserNode << "\n";
        edges_.insert(e); //std::make_pair(UseNode, UserNode)
        // errs() << "Size: " << edges_.size() << "\n";
        // }
        // else if ( auto *UsedConstant = dyn_cast<Constant>(UseValue)) {
        //   // errs() << "Constant: " << *UseValue << "\n";
        //   auto& UseNode = lookupNodeOrNew(*UseValue);
        //   // errs() << UseNode << "\n";
        //   DotEdge e(UseNode, UserNode);
        //   errs() << e << " || " << UseNode << " -> " << UserNode << "\n";
        //   edges_.insert(e);
        //   errs() << "Size: " << edges_.size() << "\n";
        // }
        // else {
        //   unused_.push_back(UseValue);
        // }
      }
    }
  }

  for ( auto& pair : lookup_)
    nodes_.insert(pair.second);

  // errs() << "========================\n";
  // errs() << "========================\n";

  // for ( auto& edge : edges_)
  //   errs() << "EDGE: " << edge << "\n";
  // // 
  dotWriter_.write(nodes_, edges_);

  // errs() << "========================\n";
  // errs() << "========================\n";

  // for ( auto& node : nodes_)
  //   errs() << node << "\n";

  errs() << "Dot: " << nodes_.size() << " nodes. " << edges_.size() << " edges\n";
  errs() << "Unused Values: " << unused_.size() << "\n";
  // for ( auto *v : unused_)
  //   errs() << *v << "\n";
  errs() << "\n";

  return false;
}