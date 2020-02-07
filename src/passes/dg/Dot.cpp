#include "llvm/IR/Argument.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"

#include <kerma/Support/LLVMStringUtils.h>
#include <kerma/passes/dg/Dot.h>
#include <string>

namespace kerma
{

static unsigned int id = 0;

DotNode Dot::createNode()
{
  return DotNode("_" + std::to_string(id++));
}

/// Instructions

DotNode Dot::createNode(llvm::LoadInst &load)
{
  DotNode node("_" + std::to_string(id++), &load);
  node.setShape("box");
  node.setBorderColor("green");
  return node;
}

DotNode Dot::createNode(llvm::StoreInst &store)
{
  DotNode node("_" + std::to_string(id++), &store);
  node.setShape("box");
  node.setBorderColor("red");
  return node;
}

DotNode Dot::createNode(llvm::AllocaInst &alloca)
{
  DotNode node("_" + std::to_string(id++), &alloca);
  node.setBorderColor("purple");
  return node;
}

DotNode Dot::createNode(llvm::Instruction &instr)
{
  DotNode node("_" + std::to_string(id++), &instr);
  return node;
}

/// Values

DotNode Dot::createNode(llvm::Constant &constant)
{
  DotNode node("_" + std::to_string(id++), &constant);
  node.setBorderColor("blue");
  return node;
}

DotNode Dot::createNode(llvm::Argument &argument)
{
  DotNode node("_" + std::to_string(id++), &argument);
  node.setBorderColor("orange");
  return node;
}

DotNode Dot::createNode(llvm::Value &V) 
{
  DotNode node("_" + std::to_string(id++), &V);
  llvm::errs() << node.getTyStr() << " |" << *node.getLLVMValue() << "\n";
  return node;
}








} /// NAMESPACE kerma